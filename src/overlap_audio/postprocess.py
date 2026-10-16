"""Validation-only probability ensembles and recording-safe temporal filtering."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .natural import scores, select_thresholds, file_hash
from .transfer import Head, RATE, encoder, windows, predict_head, write_json, load


def checked_probabilities(probability):
    p=np.asarray(probability,dtype=np.float32)
    if p.ndim!=2 or not len(p) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Expected nonempty finite probability matrix')
    return p


def convex_mix(probabilities,weights):
    values=np.asarray(probabilities,dtype=np.float32)
    w=np.asarray(weights,dtype=np.float64)
    if values.ndim!=3 or w.shape!=(len(values),) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-8,rtol=0):
        raise ValueError('Ensemble weights must be finite, nonnegative and sum to one')
    for p in values:checked_probabilities(p)
    return np.clip(np.einsum('k,knc->nc',w,values),0,1).astype(np.float32)


def temporal_filter(probability,record_ids,seconds,spec):
    """No targets/masks enter filtering. Never cross records or missing seconds."""
    p=checked_probabilities(probability);ids=np.asarray(record_ids);t=np.asarray(seconds)
    if ids.shape!=(len(p),) or t.shape!=(len(p),) or not np.isfinite(t).all() or (t<0).any() or not (t==np.floor(t)).all() or any(not str(x) for x in ids):
        raise ValueError('Invalid recording/time alignment')
    if np.any((ids[1:]==ids[:-1])&(t[1:]<=t[:-1])):raise ValueError('Recording seconds must increase strictly')
    kind=spec.get('kind')
    if kind not in ['none','box','ewma']:raise ValueError('Unknown temporal filter')
    width=spec.get('width',1);alpha=spec.get('alpha',1.)
    if kind=='box' and (not isinstance(width,int) or width<1 or width%2!=1):raise ValueError('Box width must be a positive odd integer')
    if kind=='ewma' and (not math.isfinite(alpha) or not 0<alpha<=1):raise ValueError('EWMA alpha must be in (0,1]')
    if kind=='none':return p.copy()
    edges=np.r_[0,np.where((ids[1:]!=ids[:-1])|(t[1:]!=t[:-1]+1))[0]+1,len(p)]
    output=np.empty_like(p)
    for lo,hi in zip(edges[:-1],edges[1:]):
        block=p[lo:hi].astype(np.float64)
        if kind=='box':
            cumulative=np.vstack([np.zeros((1,p.shape[1])),block.cumsum(0)])
            center=np.arange(len(block));left=np.maximum(0,center-width//2);right=np.minimum(len(block),center+width//2+1)
            output[lo:hi]=(cumulative[right]-cumulative[left])/(right-left)[:,None]
        else:
            state=block[0];output[lo]=state
            for index in range(1,len(block)):
                state=alpha*block[index]+(1-alpha)*state;output[lo+index]=state
    return np.clip(output,0,1)


def apply_policy(component_probabilities,policy,record_ids,seconds):
    branches=[]
    for branch in policy['branches']:
        p=convex_mix(component_probabilities,branch['weights'])
        for spec in branch.get('filters',[]):p=temporal_filter(p,record_ids,seconds,spec)
        branches.append(p)
    p=convex_mix(np.stack(branches),policy['mix_weights'])
    for spec in policy.get('filters',[]):p=temporal_filter(p,record_ids,seconds,spec)
    return p


def lookahead(policy,components):
    future=0
    for mix,branch in zip(policy['mix_weights'],policy['branches']):
        if mix<=0:continue
        active=[c['context_seconds']//2 for c,w in zip(components,branch['weights']) if w>0]
        extra=sum(s['width']//2 for s in branch.get('filters',[]) if s['kind']=='box')
        future=max(future,max(active,default=0)+extra)
    return future+sum(s['width']//2 for s in policy.get('filters',[]) if s['kind']=='box')


def validate_bundle(bundle):
    try:
        if bundle['kind']!='panns_probability_ensemble_v1' or bundle['sample_rate']!=RATE:raise ValueError('Unsupported checkpoint schema')
        classes=bundle['classes'];components=bundle['components'];digest=bundle['encoder_weight_sha256']
        if not isinstance(classes,list) or not classes or any(not isinstance(c,str) or not c for c in classes) or len(set(classes))!=len(classes):raise ValueError('Invalid class order')
        if not isinstance(components,list) or not components or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid components/encoder identity')
        for c in components:
            if c['classes']!=classes or c['sample_rate']!=RATE or c['context_seconds'] not in [1,5] or c['encoder_weight_sha256']!=digest:raise ValueError('Component contract mismatch')
            for key in ['mean','std']:
                if not isinstance(c[key],torch.Tensor) or c[key].shape!=(2048,) or not torch.isfinite(c[key]).all():raise ValueError('Invalid component normalization')
            dropout=c['config'].get('dropout',.3)
            if not (c['std']>0).all() or c['config']['hidden'] not in [0,256] or not math.isfinite(dropout) or not 0<=dropout<=1:raise ValueError('Invalid component scaling/architecture')
            if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in c['state_dict'].values()):raise ValueError('Nonfinite component state')
            Head(len(classes),c['config']['hidden'],dropout).load_state_dict(c['state_dict'],strict=True)
        threshold=bundle['thresholds']
        if not isinstance(threshold,torch.Tensor) or threshold.shape!=(len(classes),) or not torch.isfinite(threshold).all() or not ((threshold>=0)&(threshold<=1)).all():raise ValueError('Invalid selected thresholds')
        policy=bundle['policy'];branches=policy['branches']
        if not isinstance(branches,list) or not branches:raise ValueError('Missing branches')
        dummy=np.zeros((len(components),1,len(classes)),np.float32)
        for branch in branches:
            convex_mix(dummy,branch['weights'])
            for spec in branch.get('filters',[]):temporal_filter(dummy[0],['x'],[0],spec)
        convex_mix(np.zeros((len(branches),1,len(classes)),np.float32),policy['mix_weights'])
        for spec in policy.get('filters',[]):temporal_filter(dummy[0],['x'],[0],spec)
    except (KeyError,TypeError,RuntimeError,IndexError,AttributeError) as error:
        raise ValueError('Malformed ensemble checkpoint contract') from error
    return bundle


def prune_bundle(bundle):
    """Preserve the policy exactly while removing unreachable model work."""
    validate_bundle(bundle)
    policy=bundle['policy'];pairs=[(mix,branch) for mix,branch in zip(policy['mix_weights'],policy['branches']) if mix>0]
    indices=sorted({k for mix,branch in pairs for k,w in enumerate(branch['weights']) if w>0})
    selected=dict(bundle,components=[bundle['components'][k] for k in indices],retained_component_indices=indices)
    selected['policy']=dict(policy,branches=[dict(branch,weights=[branch['weights'][k] for k in indices]) for mix,branch in pairs],mix_weights=[mix for mix,branch in pairs])
    validate_bundle(selected)
    return selected


def cached_selected(probabilities,bundle,ids,seconds):
    return apply_policy(probabilities[bundle['retained_component_indices']],bundle['policy'],ids,seconds)


def component_predictions(component,features,device='cpu'):
    x=torch.as_tensor(features,dtype=torch.float32)
    count=len(component['classes']);mean=component['mean'];std=component['std']
    if x.ndim!=2 or x.shape[1]!=2048 or mean.shape!=(2048,) or std.shape!=(2048,) or not torch.isfinite(x).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('Invalid component features or normalization')
    model=Head(count,component['config']['hidden'],component['config'].get('dropout',.3)).to(device);model.load_state_dict(component['state_dict'])
    return predict_head(model,((x-mean)/std).to(device)).numpy()


def aligned_cache(root,split,metadata):
    x,y,m=load(root,split,metadata)
    with np.load(Path(root)/(split+'.npz'),allow_pickle=False) as data:ids=data['record_ids']
    durations={r['id']:r['seconds'] for r in metadata['sources']}
    seconds=np.empty(len(ids),np.int64);seen=set();lo=0
    while lo<len(ids):
        identifier=str(ids[lo]);hi=lo+1
        while hi<len(ids) and ids[hi]==identifier:hi+=1
        if identifier in seen or hi-lo!=durations[identifier]:raise ValueError('Noncontiguous/misaligned recording rows')
        seen.add(identifier);seconds[lo:hi]=np.arange(hi-lo);lo=hi
    return x.numpy(),y.numpy(),m.numpy(),ids,seconds


def load_components(paths,metadata,context=1):
    result=[]
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=True)
        if c.get('sample_rate')!=RATE or c.get('context_seconds')!=context or c['classes']!=metadata['classes'] or c['encoder_weight_sha256']!=metadata['encoder_weight_sha256']:
            raise ValueError('Component context/classes/encoder contract mismatch')
        result.append(c)
    return result


def macro_ap(y,p,m,classes):
    value=scores(torch.from_numpy(y),torch.from_numpy(p),torch.from_numpy(m),torch.full((len(classes),),.5),classes)['macro_ap_supported']
    if value is None:raise ValueError('No supported validation classes')
    return value


def cpu_study(cache,heads,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh postprocessing study directory')
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((Path(cache)/'metadata.json').read_text(encoding='utf-8'))
    if len(heads)!=6:raise ValueError('Study requires exactly six prespecified central heads')
    components=load_components(heads,metadata)
    x,y,m,ids,seconds=aligned_cache(cache,'validation',metadata)
    p=np.stack([component_predictions(c,x) for c in components])
    choices=[('original-mlp',[0,0,1,0,0,0]),('original-three',[1/3]*3+[0]*3),('lower-rate-three',[0]*3+[1/3]*3),('all-six',[1/6]*6),('mlp-pair',[0,0,.5,0,0,.5])]
    stages=[];best=-1;winner=None
    for name,weights in choices:
        value=macro_ap(y,convex_mix(p,weights),m,metadata['classes']);stages.append({'name':name,'weights':weights,'validation_ap':value})
        if value>best:best=value;winner=stages[-1]
    filtered=[];best=-1;filter_winner=None
    specifications=[{'kind':'none'}]+[{'kind':'box','width':width} for width in [3,5,9]]+[{'kind':'ewma','alpha':a} for a in [.25,.5]]
    base=convex_mix(p,winner['weights'])
    for spec in specifications:
        value=macro_ap(y,temporal_filter(base,ids,seconds,spec),m,metadata['classes']);filtered.append({'filter':spec,'validation_ap':value})
        if value>best:best=value;filter_winner=filtered[-1]
    result={'stage1':stages,'stage1_winner':winner,'stage2':filtered,'stage2_winner':filter_winner,
            'classes':metadata['classes'],'validation_rows':len(y),'seed':42,'test_labels_loaded':False,
            'input_sha256':{'validation_cache':file_hash(Path(cache)/'validation.npz'),'metadata':file_hash(Path(cache)/'metadata.json'),'heads':[file_hash(path) for path in heads]}}
    write_json(output/'validation-study.json',result)
    # Private probability cache and states permit exact cloud replay without fitting.
    np.savez_compressed(output/'validation-probabilities.npz',probabilities=p,targets=y,masks=m,record_ids=ids,seconds=seconds)
    torch.save({'components':components,'study':result},output/'central-components.pt')
    return result


def select_combination(central_probabilities,five_probability,y,m,ids,seconds,study,components,output):
    """Lock state and thresholds before caller is permitted to load test labels."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():raise ValueError('Selection already frozen')
    classes=components[-1]['classes'];baseline=macro_ap(y,five_probability,m,classes)
    if abs(baseline-.47008825693046674)>1e-4:raise ValueError('Five-second baseline replay discrepancy')
    probabilities=np.concatenate([central_probabilities,five_probability[None]],axis=0)
    central={'weights':study['stage1_winner']['weights']+[0], 'filters':[study['stage2_winner']['filter']]}
    five={'weights':[0]*6+[1], 'filters':[]}
    policies=[{'name':'five-second-baseline','branches':[central,five],'mix_weights':[0,1],'filters':[]}]
    for name,filters in [('central-stage1',[]),('central-stage2',[study['stage2_winner']['filter']])]:
        policies.append({'name':name,'branches':[dict(central,filters=filters),five],'mix_weights':[1,0],'filters':[]})
    for alpha in [0,.25,.5]:
        for spec in [{'kind':'none'},{'kind':'box','width':3}]:
            policies.append({'name':f'central-{alpha}-final-{spec}','branches':[central,five],'mix_weights':[alpha,1-alpha],'filters':[spec]})
    records=[];best=baseline;selected=policies[0]
    for policy in policies:
        value=macro_ap(y,apply_policy(probabilities,policy,ids,seconds),m,classes)
        records.append({'policy':policy,'validation_ap':value})
        if value>best:best=value;selected=policy
    if best<=baseline+1e-4:selected=policies[0];best=baseline
    selected_probability=apply_policy(probabilities,selected,ids,seconds)
    thresholds=select_thresholds(torch.from_numpy(y),torch.from_numpy(selected_probability),torch.from_numpy(m))
    bundle={'kind':'panns_probability_ensemble_v1','classes':classes,'sample_rate':RATE,'components':components,
            'encoder_weight_sha256':components[-1]['encoder_weight_sha256'],'policy':selected,'thresholds':thresholds,
            'lookahead_seconds':lookahead(selected,components)}
    bundle=prune_bundle(bundle)
    torch.save(bundle,output/'selected.pt')
    report={'baseline_validation_ap':baseline,'selected_validation_ap':best,'improved':best>baseline+1e-4,
            'selected_policy':bundle['policy'],'retained_component_indices':bundle['retained_component_indices'],
            'candidates':records,'thresholds':thresholds.tolist(),'lookahead_seconds':bundle['lookahead_seconds'],
            'test_labels_loaded_at_selection':False,'checkpoint_sha256':file_hash(output/'selected.pt')}
    write_json(output/'selection.json',report)
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    replay=cached_selected(probabilities,restored,ids,seconds)
    if not np.array_equal(replay,selected_probability):raise RuntimeError('Policy checkpoint replay mismatch')
    return bundle,report


def infer(checkpoint,weights,audio,output,device='cpu'):
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
    validate_bundle(bundle)
    if file_hash(weights)!=bundle['encoder_weight_sha256']:raise ValueError('Encoder identity mismatch')
    classes=bundle['classes'];thresholds=bundle['thresholds']
    if thresholds.shape!=(len(classes),) or not torch.isfinite(thresholds).all() or not ((thresholds>=0)&(thresholds<=1)).all():raise ValueError('Invalid selected thresholds')
    active={k for mix,branch in zip(bundle['policy']['mix_weights'],bundle['policy']['branches']) if mix>0 for k,w in enumerate(branch['weights']) if w>0}
    contexts={bundle['components'][k]['context_seconds'] for k in active}
    if not contexts or not contexts<={1,5}:raise ValueError('Invalid selected contexts')
    wave,rate=sf.read(audio,dtype='float32',always_2d=True);wave=wave.mean(1)
    if not len(wave) or not np.isfinite(wave).all():raise ValueError('Expected finite nonempty audio')
    if rate!=RATE:
        gcd=math.gcd(rate,RATE);wave=resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
    total=math.ceil(len(wave)/RATE);features={};model=encoder(weights,device)
    for context in sorted(contexts):
        batches=[]
        with torch.inference_mode():
            for lo in range(0,total,16):batches.append(model(windows(wave,range(lo,min(total,lo+16)),context).to(device))['embedding'].cpu().numpy())
        features[context]=np.concatenate(batches)
    p=np.zeros((len(bundle['components']),total,len(classes)),np.float32)
    for k in active:
        component=bundle['components'][k]
        if component['classes']!=classes or component['encoder_weight_sha256']!=bundle['encoder_weight_sha256']:raise ValueError('Component contract mismatch')
        p[k]=component_predictions(component,features[component['context_seconds']],device)
    probability=apply_policy(p,bundle['policy'],np.full(total,'input'),np.arange(total))
    result={'model':bundle['kind'],'lookahead_seconds':lookahead(bundle['policy'],bundle['components']),
            'offline_future_context':lookahead(bundle['policy'],bundle['components'])>0,
            'observations':[{'onset':second,'offset':min(second+1,len(wave)/RATE),
                             'probabilities':dict(zip(classes,row.tolist())),
                             'present':[c for c,v,t in zip(classes,row,thresholds) if v>=t]} for second,row in enumerate(probability)]}
    write_json(output,result);return result


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('study');p.add_argument('--cache',required=True);p.add_argument('--heads',nargs=6,required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--weights',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=vars(parser.parse_args());command=args.pop('command')
    torch.set_num_threads(2)
    if command=='study':cpu_study(**args)
    else:infer(**args)


if __name__=='__main__':main()
