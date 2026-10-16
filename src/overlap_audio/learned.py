"""Multi-seed actual five-second learned heads; validation-only policy selection."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from .transfer import Head,masked_loss,predict_head,write_json,file_hash,RATE
from .postprocess import aligned_cache,component_predictions,macro_ap,convex_mix,apply_policy,prune_bundle,validate_bundle,cached_selected,temporal_filter,lookahead
from .postprocess_experiment import unique_input
from .natural import scores,select_thresholds

SEEDS=[42,1337,2026]
CONFIGS=[{'name':'control','hidden':256,'lr':.0003,'dropout':.3,'weight_decay':.01},
         {'name':'slower','hidden':256,'lr':.0001,'dropout':.3,'weight_decay':.01},
         {'name':'dropout','hidden':256,'lr':.0001,'dropout':.5,'weight_decay':.01},
         {'name':'decay','hidden':256,'lr':.0001,'dropout':.5,'weight_decay':.1}]


def fit_head(x,y,m,vx,vy,vm,classes,config,seed,epochs=80,device='cuda'):
    """Only x/y/m are training inputs; checkpoint moments never use vx."""
    torch.manual_seed(seed)
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.01)
    features=((x-mean)/std).to(device);validation=((vx-mean)/std).to(device)
    target,mask=y.to(device),m.to(device)
    positive=((target>.5)*mask).sum(0);weight=((mask.sum(0)-positive)/positive.clamp_min(1)).clamp(1,20)
    model=Head(len(classes),config['hidden'],config['dropout']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(seed);best=-1.;best_epoch=0;history=[]
    for epoch in range(epochs):
        model.train();total=0.
        for ids in torch.randperm(len(features),generator=generator).split(128):
            loss=masked_loss(model(features[ids]),target[ids],mask[ids],weight)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite learned-head loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ids)
        probability=predict_head(model,validation)
        value=scores(vy,probability,vm,torch.full((len(classes),),.5),classes)['macro_ap_supported']
        if value is None:raise ValueError('No supported validation classes')
        history.append({'epoch':epoch+1,'loss':total/len(features),'validation_ap':value})
        if value>best:
            best=value;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch+1>=20 and epoch+1-best_epoch>=12:break
    return {'state_dict':state,'config':config,'classes':classes,'mean':mean,'std':std,'seed':seed,
            'sample_rate':RATE,'context_seconds':5}, {'seed':seed,'config':config,'best_epoch':best_epoch,'validation_ap':best,
             'history':history,'stop':'validation_plateau' if epoch+1>=20 and epoch+1-best_epoch>=12 else 'epoch_limit'}


def flatten_mixture(old,new_components,new_filter,alpha):
    """Distribute linear temporal operators; compare roundoff before selection."""
    old_count=len(old['components']);new_count=len(new_components)
    components=old['components']+new_components;branches=[];mix=[]
    for weight,branch in zip(old['policy']['mix_weights'],old['policy']['branches']):
        branches.append({'weights':branch['weights']+[0]*new_count,'filters':branch.get('filters',[])+old['policy'].get('filters',[])})
        mix.append((1-alpha)*weight)
    branches.append({'weights':[0]*old_count+[1/new_count]*new_count,'filters':[new_filter]});mix.append(alpha)
    return {'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'classes':old['classes'],
            'encoder_weight_sha256':old['encoder_weight_sha256'],'components':components,'thresholds':old['thresholds'],
            'policy':{'name':f'learned-weight-{alpha}','branches':branches,'mix_weights':mix,'filters':[]}}


def bundle_probabilities(bundle,features):
    validate_bundle(bundle)
    return np.stack([component_predictions(c,features[c['context_seconds']]) for c in bundle['components']])


def run(input_root,output,device='cuda'):
    started=time.time();output=Path(output)
    if device not in ('cpu','cuda'):raise ValueError('Device must be explicit cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no implicit CPU fallback')
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh learned study directory')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    five_cache=unique_input(input_root,'five-second-cache','metadata.json').parent
    one_cache=unique_input(input_root,'private-embeddings','metadata.json').parent
    metadata=json.loads((five_cache/'metadata.json').read_text());one_metadata=json.loads((one_cache/'metadata.json').read_text())
    if metadata['context_seconds']!=5 or metadata['classes']!=one_metadata['classes'] or metadata['assignment']!=one_metadata['assignment'] or metadata['sources']!=one_metadata['sources']:raise ValueError('Context/source contracts differ')
    for root,meta in [(five_cache,metadata),(one_cache,one_metadata)]:
        for split,contract in meta['contracts'].items():
            if file_hash(root/(split+'.npz'))!=contract['sha256']:raise ValueError('Source cache checksum mismatch')
    prior_path=unique_input(input_root,'selected','selected.pt')
    prior=torch.load(prior_path,map_location='cpu',weights_only=True);validate_bundle(prior)
    prior_selection=json.loads((prior_path.parent/'selection.json').read_text())
    reference_path=unique_input(input_root,'transfer-v2','selected.pt')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    tx,ty,tm,tids,tseconds=aligned_cache(five_cache,'train',metadata)
    vx,vy,vm,ids,seconds=aligned_cache(five_cache,'validation',metadata)
    one_x,one_y,one_m,one_ids,one_seconds=aligned_cache(one_cache,'validation',one_metadata)
    if not all(np.array_equal(a,b) for a,b in [(vy,one_y),(vm,one_m),(ids,one_ids),(seconds,one_seconds)]):raise ValueError('Context row/label alignment mismatch')
    reference_ap=macro_ap(vy,component_predictions(reference,vx),vm,metadata['classes'])
    if abs(reference_ap-.47008825693046674)>1e-4:raise ValueError('Original five-second feature replay discrepancy')
    features={1:one_x,5:vx};old_probabilities=bundle_probabilities(prior,features)
    old_p=apply_policy(old_probabilities,prior['policy'],ids,seconds)
    old_ap=macro_ap(vy,old_p,vm,metadata['classes'])
    if abs(old_ap-prior_selection['selected_validation_ap'])>1e-4:raise ValueError('Prior policy replay discrepancy')
    manifest={'configs':CONFIGS,'seeds':SEEDS,'device':device,'reference_ap':reference_ap,'prior_validation_ap':old_ap,'test_labels_loaded':False,
              'input_hashes':{'five_metadata':file_hash(five_cache/'metadata.json'),'one_metadata':file_hash(one_cache/'metadata.json'),'prior_checkpoint':file_hash(prior_path),'reference_checkpoint':file_hash(reference_path)}}
    write_json(output/'protocol.json',manifest)
    x=torch.from_numpy(tx);y=torch.from_numpy(ty);m=torch.from_numpy(tm);validation_x=torch.from_numpy(vx);validation_y=torch.from_numpy(vy);validation_m=torch.from_numpy(vm)
    candidates=[];selected=None;best=-1.
    for config in CONFIGS:
        components=[];histories=[];probabilities=[];folder=output/config['name'];folder.mkdir()
        for seed in SEEDS:
            c,record=fit_head(x,y,m,validation_x,validation_y,validation_m,metadata['classes'],config,seed,device=device)
            c['encoder_weight_sha256']=metadata['encoder_weight_sha256'];torch.save(c,folder/f'seed-{seed}.pt')
            components.append(c);histories.append(record);probabilities.append(component_predictions(c,vx))
            write_json(folder/f'seed-{seed}.json',record)
            print(json.dumps({'config':config['name'],'seed':seed,'best_epoch':record['best_epoch'],'validation_ap':record['validation_ap']}),flush=True)
        p=convex_mix(np.stack(probabilities),[1/len(SEEDS)]*len(SEEDS));value=macro_ap(vy,p,vm,metadata['classes'])
        values=[h['validation_ap'] for h in histories]
        record={'config':config,'equal_seed_ensemble_validation_ap':value,'individual_seed_mean_ap':float(np.mean(values)),'individual_seed_std_ap':float(np.std(values,ddof=1)),'fits':histories}
        candidates.append(record)
        if value>best:best=value;selected=(components,p,record)
    components,p,chosen=selected;filter_results=[];best_filter={'kind':'none'};filter_best=-1
    for spec in [{'kind':'none'},{'kind':'box','width':3},{'kind':'box','width':5}]:
        value=macro_ap(vy,temporal_filter(p,ids,seconds,spec),vm,metadata['classes']);filter_results.append({'filter':spec,'validation_ap':value})
        if value>filter_best:filter_best=value;best_filter=spec
    all_probabilities=np.concatenate([old_probabilities,np.stack([component_predictions(c,vx) for c in components])],axis=0)
    guard=flatten_mixture(prior,components,best_filter,0)
    recovered=apply_policy(all_probabilities,guard['policy'],ids,seconds)
    if float(np.max(np.abs(recovered-old_p)))>1e-6 or abs(macro_ap(vy,recovered,vm,metadata['classes'])-old_ap)>1e-4:raise ValueError('Linear policy flattening changes prior outputs')
    combinations=[];winner=None;global_best=old_ap
    for alpha in [0,.25,.5,.75,1]:
        b=flatten_mixture(prior,components,best_filter,alpha);value=macro_ap(vy,apply_policy(all_probabilities,b['policy'],ids,seconds),vm,metadata['classes'])
        combinations.append({'learned_weight':alpha,'validation_ap':value})
        if value>global_best:global_best=value;winner=b
    if winner is None or global_best<=old_ap+1e-4:
        bundle=prior;final_p=old_p;improved=False;global_best=old_ap
    else:
        bundle=prune_bundle(winner);final_p=cached_selected(all_probabilities,bundle,ids,seconds);improved=True
        bundle['thresholds']=select_thresholds(torch.from_numpy(vy),torch.from_numpy(final_p),torch.from_numpy(vm))
    bundle['lookahead_seconds']=lookahead(bundle['policy'],bundle['components']);validate_bundle(bundle)
    torch.save(bundle,output/'selected.pt')
    selection={'candidates':candidates,'chosen_config':chosen['config'],'filters':filter_results,'chosen_filter':best_filter,'mixtures':combinations,
               'prior_validation_ap':old_ap,'selected_validation_ap':global_best,'improved':improved,'test_labels_loaded':False,'checkpoint_sha256':file_hash(output/'selected.pt'),'lookahead_seconds':bundle['lookahead_seconds']}
    write_json(output/'selection.json',selection);selection_hash=file_hash(output/'selection.json')
    # First observed-test-label access, after complete learned policy freeze.
    fx,fy,fm,fids,fseconds=aligned_cache(five_cache,'test',metadata)
    ox,oy,om,oids,oseconds=aligned_cache(one_cache,'test',one_metadata)
    if not all(np.array_equal(a,b) for a,b in [(fy,oy),(fm,om),(fids,oids),(fseconds,oseconds)]):raise ValueError('Test context alignment mismatch')
    test_components=bundle_probabilities(bundle,{1:ox,5:fx});test_p=apply_policy(test_components,bundle['policy'],fids,fseconds)
    old_test_components=bundle_probabilities(prior,{1:ox,5:fx});old_test=apply_policy(old_test_components,prior['policy'],fids,fseconds)
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    replay=apply_policy(bundle_probabilities(restored,{1:ox,5:fx}),restored['policy'],fids,fseconds)
    if not np.array_equal(replay,test_p):raise RuntimeError('Actual learned checkpoint replay mismatch')
    report={'selected':scores(torch.from_numpy(fy),torch.from_numpy(test_p),torch.from_numpy(fm),bundle['thresholds'],metadata['classes']),
            'prior':scores(torch.from_numpy(fy),torch.from_numpy(old_test),torch.from_numpy(fm),prior['thresholds'],metadata['classes']),
            'status':'Observed test, no untouched claim','selection_unchanged':file_hash(output/'selection.json')==selection_hash,'checkpoint_probability_max_difference':0.}
    write_json(output/'observed-holdout.json',report);write_json(output/'completion.json',{'exit':0,'elapsed_seconds':time.time()-started})
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',default='/kaggle/input');parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    run(**vars(parser.parse_args()))
