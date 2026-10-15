"""Train on naturally overlapping MAESTRO recordings with time-aligned soft labels.

Recording-disjoint development/validation/test split, masked unannotated classes,
validation-only configuration/checkpoint/threshold selection, single final release.
This is a portfolio holdout study, not the official five-fold DCASE benchmark.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from .experiment import StrongCNN
from .metrics import average_precision, f1_per_class

RATE=16000
OFFICIAL_CLASSES={'birds_singing','car','people talking','footsteps','children voices',
                  'wind_blowing','brakes_squeaking','large_vehicle','cutlery and dishes',
                  'metro approaching','metro leaving'}


def annotations(root):
    records=[]
    for path in sorted((Path(root)/'development_annotation').rglob('*.txt')):
        scene=path.parent.name.removeprefix('soft_labels_')
        rows=[]
        roundoff_clamps=0
        with path.open(encoding='utf-8') as handle:
            for row in csv.reader(handle,delimiter='\t'):
                if len(row)!=4:
                    raise ValueError('Annotation requires onset, offset, class and soft label')
                onset,offset,label,soft=float(row[0]),float(row[1]),row[2],float(row[3])
                if not label.strip() or not all(math.isfinite(value) for value in [onset,offset,soft]) or onset<0 or offset<=onset or not -1e-6<=soft<=1+1e-6:
                    raise ValueError('Invalid annotation interval/probability')
                if not 0<=soft<=1:
                    roundoff_clamps+=1
                    soft=min(1.,max(0.,soft))
                rows.append((onset,offset,label,soft))
        if not rows:
            raise ValueError('Annotation files must contain rows')
        records.append({'id':path.stem,'scene':scene,'rows':rows,'roundoff_clamps':roundoff_clamps,
                        'annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if not records:
        raise ValueError('No natural annotation files found')
    return records


def frozen_split(records):
    if len({r['id'] for r in records})!=len(records):
        raise ValueError('Recording IDs must be globally unique')
    scenes=sorted({r['scene'] for r in records})
    assignment={}
    for scene in scenes:
        members=sorted(r['id'] for r in records if r['scene']==scene)
        if len(members)<5:
            raise ValueError('Need >=5 independent recordings per scene')
        random.Random('maestro-portfolio-v1:'+scene).shuffle(members)
        test_count=max(1,round(len(members)*.2))
        validation_count=max(1,round(len(members)*.2))
        for i,record in enumerate(members):
            assignment[record]='test' if i<test_count else 'validation' if i<test_count+validation_count else 'train'
    return assignment


def mel_filter():
    hz_to_mel=lambda hz:2595*np.log10(1+hz/700)
    mel_to_hz=lambda mel:700*(10**(mel/2595)-1)
    boundaries=mel_to_hz(np.linspace(hz_to_mel(50),hz_to_mel(8000),66))
    frequencies=np.linspace(0,RATE/2,513)
    bank=np.zeros((64,513),dtype=np.float32)
    for k in range(64):
        lo,mid,hi=boundaries[k:k+3]
        bank[k]=np.maximum(0,np.minimum((frequencies-lo)/(mid-lo),(hi-frequencies)/(hi-mid)))
        bank[k]/=max(bank[k].sum(),1e-8)
    return torch.from_numpy(bank)


def logmel(waves):
    power=torch.stft(waves,n_fft=1024,hop_length=320,window=torch.hann_window(1024),
                     return_complex=True).abs().square()
    return torch.log1p(torch.einsum('mf,bft->bmt',mel_filter(),power))[:,None]


def prepare(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records=annotations(root)
    classes=sorted({row[2] for r in records for row in r['rows']})
    assignment=frozen_split(records)
    scene_labels={scene:{row[2] for r in records if r['scene']==scene for row in r['rows']}
                  for scene in {r['scene'] for r in records}}
    index={label:k for k,label in enumerate(classes)}
    arrays={split:{'features':[],'targets':[],'masks':[],'record_ids':[]} for split in ['train','validation','test']}
    source_info=[]
    for record in records:
        candidates=list((root/'development_audio').rglob(record['id']+'.wav'))
        if len(candidates)!=1:
            raise ValueError('Missing/ambiguous WAV for '+record['id'])
        path=candidates[0]
        wave,rate=sf.read(path,dtype='float32',always_2d=True)
        wave=wave.mean(axis=1)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Audio must contain finite samples: '+record['id'])
        if rate!=RATE:
            gcd=math.gcd(rate,RATE)
            wave=resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
        duration=min(len(wave)//RATE,math.floor(max(row[1] for row in record['rows'])))
        target=np.zeros((duration,len(classes)),dtype=np.float32)
        mask=np.zeros_like(target)
        # Zero rows omitted from annotations mean no activity among scene-annotated classes.
        for label in scene_labels[record['scene']]:
            mask[:,index[label]]=1
        for onset,offset,label,soft in record['rows']:
            for second in range(max(0,math.floor(onset)),min(duration,math.ceil(offset))):
                target[second,index[label]]=max(target[second,index[label]],soft)
        features=[]
        for lo in range(0,duration,32):
            windows=[]
            for second in range(lo,min(duration,lo+32)):
                # Five seconds of offline context around the one-second labeled interval.
                start=(second-2)*RATE
                stop=start+5*RATE
                padded=np.pad(wave[max(0,start):min(len(wave),stop)],
                              (max(0,-start),max(0,stop-len(wave))))
                windows.append(padded)
            features.append(logmel(torch.from_numpy(np.stack(windows))).numpy().astype(np.float16))
        split=assignment[record['id']]
        arrays[split]['features'].append(np.concatenate(features))
        arrays[split]['targets'].append(target)
        arrays[split]['masks'].append(mask)
        arrays[split]['record_ids'].extend([record['id']]*duration)
        source_info.append({k:record[k] for k in ['id','scene','annotation_sha256','roundoff_clamps']})
        source_info[-1]['audio_sha256']=file_hash(path)
        print(json.dumps({'prepared':record['id'],'split':split,'seconds':duration}),flush=True)
    contracts={}
    for split,parts in arrays.items():
        features=np.concatenate(parts['features']);targets=np.concatenate(parts['targets']);masks=np.concatenate(parts['masks'])
        path=output/(split+'.npz')
        np.savez_compressed(path,features=features,targets=targets,masks=masks,
                            record_ids=np.asarray(parts['record_ids']))
        contracts[split]={'sha256':file_hash(path),'rows':len(targets),
                          'features_shape':list(features.shape),'target_shape':list(targets.shape),
                          'record_ids':sorted(set(parts['record_ids']))}
    metadata={'dataset':'MAESTRO Real v1','classes':classes,'assignment':assignment,'sources':source_info,
              'sample_rate':RATE,'context_seconds':5,'label_seconds':1,'split_seed':'maestro-portfolio-v1',
              'license':'Custom experimental/noncommercial, source Machine Listening Group at Tampere University',
              'evaluation':'Recording-disjoint custom holdout; NOT official DCASE five-fold challenge score',
              'mask':'Classes outside scene annotation vocabulary are unobserved, not negative'}
    metadata['soft_label_roundoff_tolerance']=1e-6
    metadata['soft_label_roundoff_clamps']=sum(record['roundoff_clamps'] for record in records)
    metadata['cache_contracts']=contracts
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def scores(target,probability,mask,thresholds,classes):
    if target.ndim!=2 or probability.shape!=target.shape or mask.shape!=target.shape or thresholds.ndim!=1 or len(thresholds)!=target.shape[1] or len(classes)!=target.shape[1]:
        raise ValueError('Metric input shapes or class order are inconsistent')
    if not all(torch.isfinite(value).all() for value in [target,probability,mask,thresholds]) or not ((target>=0)&(target<=1)).all() or not ((probability>=0)&(probability<=1)).all() or not ((thresholds>=0)&(thresholds<=1)).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Metrics require finite probabilities and binary observation masks')
    hard=(target>.5).float()
    prediction=(probability>=thresholds).float()
    per_class=[]
    for k,label in enumerate(classes):
        observed=mask[:,k].bool()
        y,p=hard[observed,k],probability[observed,k]
        f1=float(f1_per_class(y[:,None],(p>=thresholds[k]).float()[:,None])[0]) if len(y) else None
        per_class.append({'class':label,'positive':int(y.sum()),'observed':len(y),'f1':f1,
                          'ap':average_precision(y,p) if len(y) else None})
    supported=[r for r in per_class if r['positive']>0]
    official=[r for r in supported if r['class'] in OFFICIAL_CLASSES]
    observed_classes=[r for r in per_class if r['observed']>0]
    official_observed=[r for r in observed_classes if r['class'] in OFFICIAL_CLASSES]
    tp=float((hard*prediction*mask).sum());fp=float(((1-hard)*prediction*mask).sum());fn=float((hard*(1-prediction)*mask).sum())
    average=lambda rows,key:float(np.mean([r[key] for r in rows])) if rows else None
    return {'macro_ap_supported':average(supported,'ap'),
            'macro_f1_supported':average(supported,'f1'),
            'macro_f1_all_observed_classes':average(observed_classes,'f1'),
            'official_class_macro_f1_supported':average(official,'f1'),
            'official_class_macro_f1_all_observed_classes':average(official_observed,'f1'),
            'micro_f1':2*tp/max(1.,2*tp+fp+fn) if mask.sum() else None,'per_class':per_class,
            'unsupported_classes':[r['class'] for r in per_class if not r['positive']]}


def probabilities(model,features,device):
    model.eval();result=[]
    with torch.no_grad():
        for batch in features.split(64):
            result.append(torch.sigmoid(model(batch.to(device))).cpu())
    return torch.cat(result)


def select_thresholds(target,probability,mask):
    thresholds=[]
    hard=(target>.5).float()
    for k in range(target.shape[1]):
        observed=mask[:,k].bool()
        y,p=hard[observed,k],probability[observed,k]
        grid=torch.linspace(.05,.95,19)
        if not y.sum():
            thresholds.append(.5);continue
        f1=[float(f1_per_class(y[:,None],(p>=t).float()[:,None])[0]) for t in grid]
        thresholds.append(float(grid[max(range(len(grid)),key=lambda i:f1[i])]))
    return torch.tensor(thresholds)


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_cache_bundle(root,metadata):
    """Inspect hashes and IDs only, including held-out IDs; never score test data."""
    contracts=metadata.get('cache_contracts')
    if not contracts or set(contracts)!={'train','validation','test'}:
        raise ValueError('Caches lack integrity contracts; regenerate with natural prepare')
    source_sets={}
    for split,contract in contracts.items():
        path=Path(root)/(split+'.npz')
        if file_hash(path)!=contract['sha256']:
            raise ValueError('Cache checksum mismatch: '+split)
        with np.load(path,allow_pickle=False) as cache:
            if set(cache.files)!={'features','targets','masks','record_ids'}:
                raise ValueError('Cache has an invalid schema: '+split)
            ids=cache['record_ids']
        if ids.ndim!=1 or len(ids)!=contract['rows']:
            raise ValueError('Recording IDs and cache rows are misaligned')
        actual=set(ids.tolist())
        expected={record for record,part in metadata['assignment'].items() if part==split}
        if actual!=expected or actual!=set(contract['record_ids']):
            raise ValueError('Cache recording assignment mismatch: '+split)
        source_sets[split]=actual
    if source_sets['train']&source_sets['validation'] or source_sets['train']&source_sets['test'] or source_sets['validation']&source_sets['test']:
        raise ValueError('Recording source crosses cache splits')


def load_cache(path,contract=None,classes=None):
    with np.load(path,allow_pickle=False) as cache:
        arrays={key:cache[key] for key in ['features','targets','masks','record_ids']}
    feature,target,mask,ids=[arrays[key] for key in ['features','targets','masks','record_ids']]
    if feature.ndim!=4 or feature.shape[1:]!=(1,64,251) or target.ndim!=2 or mask.shape!=target.shape or ids.ndim!=1:
        raise ValueError('Invalid natural cache dimensions')
    if not len(feature) or len(feature)!=len(target) or len(ids)!=len(target):
        raise ValueError('Natural feature/target/source rows are misaligned')
    if classes is not None and target.shape[1]!=classes:
        raise ValueError('Natural cache class count differs from metadata')
    if not all(np.isfinite(value).all() for value in [feature,target,mask]) or not ((target>=0)&(target<=1)).all() or not np.isin(mask,[0,1]).all():
        raise ValueError('Natural cache contains invalid features, labels or masks')
    if contract and (list(feature.shape)!=contract['features_shape'] or list(target.shape)!=contract['target_shape']):
        raise ValueError('Natural cache shape differs from integrity contract')
    return {key:torch.from_numpy(arrays[key].astype(np.float32)) for key in ['features','targets','masks']}


def train(cache_root,output,epochs=30,device='auto'):
    cache_root,output=Path(cache_root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final-test.json').exists():
        raise ValueError('Final holdout already released; never tune against repeated final evaluations')
    metadata=json.loads((cache_root/'metadata.json').read_text(encoding='utf-8'))
    classes=metadata['classes']
    validate_cache_bundle(cache_root,metadata)
    contracts=metadata['cache_contracts']
    development=load_cache(cache_root/'train.npz',contracts['train'],len(classes))
    validation=load_cache(cache_root/'validation.npz',contracts['validation'],len(classes))
    # test.npz is deliberately not loaded until configuration and epoch are frozen.
    device='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    torch.set_num_threads(2)
    hard=(development['targets']>.5).float()
    prevalence=(hard*development['masks']).sum(0)/development['masks'].sum(0).clamp_min(1)
    pos_weight=((1-prevalence)/prevalence.clamp_min(.01)).clamp(1,30).to(device)
    configs=[{'name':'soft-bce','width':24,'loss':'bce','lr':.001},
             {'name':'soft-focal','width':24,'loss':'focal','lr':.001},
             {'name':'soft-bce-wide','width':32,'loss':'bce','lr':.0005}]
    protocol={'metadata':metadata,'configs':configs,'epochs':epochs,'seed':42,'device':device,
              'selection':'validation supported-class macro AP; best validation epoch',
              'final_test':'single selected model; no test-based changes',
              'budget':'three configurations, one seed, maximum 30 epochs each; private noncommercial research'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    candidates=[];best_global=-float('inf')
    for config in configs:
        torch.manual_seed(42);np.random.seed(42)
        model=StrongCNN(len(classes),config['width']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        weight=1+(hard/(prevalence+.01)).sum(1)/hard.sum(1).clamp_min(1)
        loader=DataLoader(TensorDataset(development['features'],development['targets'],development['masks']),
                          batch_size=32,sampler=WeightedRandomSampler(weight,len(weight),replacement=True,
                          generator=torch.Generator().manual_seed(42)))
        history=[];best=-float('inf');best_epoch=0
        folder=output/config['name'];folder.mkdir(exist_ok=True)
        for epoch in range(epochs):
            model.train();loss_sum=0.
            for feature,target,mask in loader:
                feature,target,mask=feature.to(device),target.to(device),mask.to(device)
                # Time/frequency masking has no access to holdout features.
                if model.training:
                    f=int(torch.randint(0,58,()).item());t=int(torch.randint(0,240,()).item())
                    feature=feature.clone();feature[:,:,f:f+6,:]=0;feature[:,:,:,t:t+10]=0
                logits=model(feature)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=pos_weight,reduction='none')
                if config['loss']=='focal':
                    p=torch.sigmoid(logits);pt=p*target+(1-p)*(1-target);loss=loss*(1-pt).square()
                loss=(loss*mask).sum()/mask.sum().clamp_min(1)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite natural training loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                loss_sum+=float(loss.detach())*len(target)
            scheduler.step()
            probability=probabilities(model,validation['features'],device)
            validation_scores=scores(validation['targets'],probability,validation['masks'],torch.full((len(classes),),.5),classes)
            value=validation_scores['macro_ap_supported']
            if value is None:
                raise ValueError('Validation has no observed positive classes; AP cannot select a model')
            record={'config':config['name'],'epoch':epoch+1,'train_loss':loss_sum/len(weight),'validation_macro_ap':value}
            history.append(record);print(json.dumps(record),flush=True)
            if value>best:
                best,best_epoch=value,epoch+1;torch.save(model.state_dict(),folder/'best.pt')
            if epoch+1>=15 and epoch+1-best_epoch>=8:break
        result={'config':config,'best_validation_macro_ap':best,'best_epoch':best_epoch,'history':history}
        (folder/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8');candidates.append(result)
        if best>best_global:
            best_global=best
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            torch.save({'state_dict':state,'width':config['width'],'classes':classes,'config':config,
                        'sample_rate':RATE,'context_seconds':5},output/'selected.pt')
    checkpoint=torch.load(output/'selected.pt',map_location=device,weights_only=True)
    model=StrongCNN(len(classes),checkpoint['width']).to(device);model.load_state_dict(checkpoint['state_dict'])
    validation_probability=probabilities(model,validation['features'],device)
    thresholds=select_thresholds(validation['targets'],validation_probability,validation['masks'])
    checkpoint['thresholds']=thresholds;torch.save(checkpoint,output/'selected.pt')
    test=load_cache(cache_root/'test.npz',contracts['test'],len(classes))
    test_probability=probabilities(model,test['features'],device)
    final={'protocol':protocol,'validation_candidates':candidates,'selected_config':checkpoint['config'],
           'thresholds_from_validation':thresholds.tolist(),
           'test_fixed_point_five':scores(test['targets'],test_probability,test['masks'],torch.full_like(thresholds,.5),classes),
           'test_validation_thresholds':scores(test['targets'],test_probability,test['masks'],thresholds,classes),
           'test_always_present':scores(test['targets'],torch.ones_like(test['targets']),test['masks'],torch.full_like(thresholds,.5),classes),
           'limitations':['Not the official five-fold DCASE score; cannot compare directly to challenge leaderboard',
                          'Recording-disjoint; location/session provenance beyond filenames is unavailable',
                          'Five-second offline context includes future context within each recording',
                          'Human soft labels carry annotation uncertainty; binary evaluation uses strict >0.5',
                          'Unsupported classes are listed explicitly, never silently assigned a favorable score',
                          'Single random seed, limited recording count and unknown domain transfer']}
    (output/'final-test.json').write_text(json.dumps(final,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(final,indent=2),flush=True)
    return final


def infer(checkpoint_path,audio_path,output_path):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    classes=checkpoint['classes']
    if checkpoint.get('sample_rate')!=RATE or checkpoint.get('context_seconds')!=5:
        raise ValueError('Unsupported model preprocessing contract')
    thresholds=checkpoint.get('thresholds')
    if thresholds is None or len(thresholds)!=len(classes):
        raise ValueError('Checkpoint requires validated per-class thresholds')
    thresholds=torch.as_tensor(thresholds,dtype=torch.float32)
    if thresholds.ndim!=1 or not torch.isfinite(thresholds).all() or not ((thresholds>=0)&(thresholds<=1)).all():
        raise ValueError('Checkpoint thresholds must be finite probabilities')
    model=StrongCNN(len(classes),checkpoint['width'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    wave,rate=sf.read(audio_path,dtype='float32',always_2d=True)
    wave=wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Audio must contain finite samples')
    if rate!=RATE:
        gcd=math.gcd(rate,RATE)
        wave=resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
    seconds=math.ceil(len(wave)/RATE)
    observations=[]
    torch.set_num_threads(2)
    for lo in range(0,seconds,32):
        windows=[]
        for second in range(lo,min(seconds,lo+32)):
            start=(second-2)*RATE;stop=start+5*RATE
            windows.append(np.pad(wave[max(0,start):min(len(wave),stop)],
                                  (max(0,-start),max(0,stop-len(wave)))))
        with torch.no_grad():
            probability=torch.sigmoid(model(logmel(torch.from_numpy(np.stack(windows)))))
        for index,p in enumerate(probability):
            second=lo+index
            observations.append({'onset':second,'offset':min(second+1,len(wave)/RATE),
                'classes':{label:{'probability':float(value),'present':bool(value>=threshold)}
                           for label,value,threshold in zip(classes,p,thresholds)}})
    result={'model':'MAESTRO real-data multi-label model','sample_rate':RATE,'context_seconds':5,
            'offline_future_context':True,'observations':observations,
            'license':'Dataset-dependent research weights retain MAESTRO experimental/noncommercial terms'}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',default='data/external/maestro');p.add_argument('--output',default='data/processed/maestro')
    p=sub.add_parser('train');p.add_argument('--cache',default='data/processed/maestro');p.add_argument('--output',default='models/maestro-v1');p.add_argument('--epochs',type=int,default=30);p.add_argument('--device',default='auto')
    p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.output)
    elif args.command=='train':train(args.cache,args.output,args.epochs,args.device)
    else:infer(args.checkpoint,args.audio,args.output)
