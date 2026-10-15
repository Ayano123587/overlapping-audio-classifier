"""Frozen PANNs transfer learning on MAESTRO; observed holdout replication."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn

from .natural import annotations, frozen_split, file_hash, scores, select_thresholds

RATE = 32000
WEIGHT_URL = 'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
WEIGHT_MD5 = '541141fa2ee191a88f24a3219fff024e'
CONFIGS = [
    {'name': 'linear-soft', 'hidden': 0, 'weight': False, 'lr': .001},
    {'name': 'linear-balanced', 'hidden': 0, 'weight': True, 'lr': .001},
    {'name': 'mlp-balanced', 'hidden': 256, 'weight': True, 'lr': .0003},
]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def encoder(weights, device):
    """Load the externally pinned original PANNs architecture and state strictly."""
    # Import pinned upstream modules without panns_inference.__init__, whose
    # config performs an unverified shell download of irrelevant AudioSet labels.
    from ._panns.models import Cnn14
    weights = Path(weights)
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix('.part')
        with urlopen(WEIGHT_URL, timeout=90) as response, temporary.open('wb') as handle:
            for block in iter(lambda: response.read(1024*1024), b''):
                handle.write(block)
        temporary.replace(weights)
    digest = hashlib.md5()
    with weights.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_MD5:
        raise ValueError('Pretrained weight checksum mismatch')
    model = Cnn14(RATE, 1024, 320, 64, 50, 14000, 527)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def windows(wave, seconds, context_seconds=5):
    """Five-second context or central-second ablation at encoder's native rate."""
    if context_seconds not in [1,5]:
        raise ValueError('Supported context lengths are one or five seconds')
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected nonempty finite mono waveform')
    result = []
    for second in seconds:
        start = (second-context_seconds//2)*RATE
        stop = start+context_seconds*RATE
        result.append(np.pad(wave[max(0,start):min(len(wave),stop)],
                             (max(0,-start), max(0,stop-len(wave)))))
    return torch.from_numpy(np.stack(result))


def timeline(record, scene_labels, classes, duration):
    target = np.zeros((duration, len(classes)), np.float32)
    mask = np.zeros_like(target)
    index = {label: k for k,label in enumerate(classes)}
    for label in scene_labels:
        mask[:,index[label]] = 1
    for onset,offset,label,soft in record['rows']:
        for second in range(max(0,math.floor(onset)),min(duration,math.ceil(offset))):
            target[second,index[label]] = max(target[second,index[label]],soft)
    return target, mask


def prepare(root, output, weights, device='cuda', context_seconds=5):
    root, output = Path(root), Path(output)
    if (output/'metadata.json').exists():
        raise ValueError('Use a fresh embedding cache directory')
    output.mkdir(parents=True, exist_ok=True)
    if context_seconds not in [1,5]:
        raise ValueError('Supported context lengths are one or five seconds')
    records = annotations(root)
    classes = sorted({row[2] for r in records for row in r['rows']})
    assignment = frozen_split(records)
    scene_labels = {scene:{row[2] for r in records if r['scene']==scene for row in r['rows']}
                    for scene in {r['scene'] for r in records}}
    model = encoder(weights, device)
    parts = {split: [] for split in ['train','validation','test']}
    sources = []
    for record in records:
        paths = list((root/'development_audio').rglob(record['id']+'.wav'))
        if len(paths)!=1:
            raise ValueError('Missing/ambiguous recording '+record['id'])
        wave, rate = sf.read(paths[0], dtype='float32', always_2d=True)
        wave = wave.mean(1)
        if rate != RATE:
            gcd = math.gcd(rate,RATE)
            wave = resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
        duration = min(len(wave)//RATE,math.floor(max(row[1] for row in record['rows'])))
        if not duration:
            raise ValueError('Recording has no complete annotated seconds')
        features = []
        with torch.inference_mode():
            for lo in range(0,duration,16):
                batch = windows(wave,range(lo,min(duration,lo+16)),context_seconds).to(device)
                feature = model(batch)['embedding'].float().cpu().numpy()
                if feature.shape != (len(batch),2048) or not np.isfinite(feature).all():
                    raise ValueError('Invalid encoder embeddings')
                features.append(feature)
        target,mask = timeline(record,scene_labels[record['scene']],classes,duration)
        parts[assignment[record['id']]].append((np.concatenate(features),target,mask,
                                               np.full(duration,record['id'])))
        sources.append({'id':record['id'],'audio_sha256':file_hash(paths[0]),
                        'annotation_sha256':record['annotation_sha256'],'seconds':duration})
        print(json.dumps({'embedded':record['id'],'seconds':duration}),flush=True)
    contracts = {}
    for split, rows in parts.items():
        arrays = [np.concatenate([row[k] for row in rows]) for k in range(4)]
        path = output/(split+'.npz')
        np.savez_compressed(path,features=arrays[0],targets=arrays[1],masks=arrays[2],record_ids=arrays[3])
        contracts[split] = {'sha256':file_hash(path),'rows':len(arrays[0]),
                            'record_ids':sorted(set(arrays[3].tolist()))}
    metadata = {'classes':classes,'assignment':assignment,'sources':sources,'contracts':contracts,
                'encoder':'PANNs Cnn14 AudioSet 0.431','encoder_weight_md5':WEIGHT_MD5,
                'encoder_weight_sha256':file_hash(weights),'sample_rate':RATE,'context_seconds':context_seconds,
                'split':'maestro-portfolio-v1','holdout_status':'previously observed; replication, not untouched'}
    write_json(output/'metadata.json', metadata)
    return metadata


def load(cache, split, metadata):
    path = Path(cache)/(split+'.npz')
    contract = metadata['contracts'][split]
    if file_hash(path)!=contract['sha256']:
        raise ValueError('Embedding cache checksum mismatch')
    with np.load(path,allow_pickle=False) as source:
        if set(source.files)!={'features','targets','masks','record_ids'}:
            raise ValueError('Invalid embedding schema')
        x,y,m,ids = [source[k] for k in ['features','targets','masks','record_ids']]
    expected = {r for r,s in metadata['assignment'].items() if s==split}
    if ids.ndim!=1 or set(ids.tolist())!=expected or expected!=set(contract['record_ids']):
        raise ValueError('Embedding recording split mismatch')
    if x.shape!=(contract['rows'],2048) or y.shape!=(len(x),len(metadata['classes'])) or m.shape!=y.shape or len(ids)!=len(x):
        raise ValueError('Misaligned embedding rows')
    if not all(np.isfinite(a).all() for a in [x,y,m]) or not ((y>=0)&(y<=1)).all() or not np.isin(m,[0,1]).all():
        raise ValueError('Invalid embedding values')
    return tuple(torch.from_numpy(a.astype(np.float32)) for a in [x,y,m])


class Head(nn.Module):
    def __init__(self, classes, hidden=0):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2048,hidden),nn.ReLU(),nn.Dropout(.3),nn.Linear(hidden,classes)) if hidden else nn.Linear(2048,classes)

    def forward(self, x):
        return self.layers(x)


def masked_loss(logits, target, mask, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=pos_weight)
    return (loss*mask).sum()/mask.sum().clamp_min(1)


def predict_head(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([torch.sigmoid(model(batch)) for batch in x.split(256)]).cpu()


def train(cache, output, epochs=80, device='cuda', lr_scale=1.):
    cache, output = Path(cache), Path(output)
    if not math.isfinite(lr_scale) or lr_scale<=0 or epochs<1:
        raise ValueError('Learning rate scale and epochs must be positive and finite')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh run directory; selection/final reports cannot be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    # Validate all source ID sets/hashes before training, without opening heldout targets.
    cohorts = [set(c['record_ids']) for c in metadata['contracts'].values()]
    if any(cohorts[i]&cohorts[j] for i in range(3) for j in range(i)):
        raise ValueError('Sources cross split boundaries')
    for split, contract in metadata['contracts'].items():
        if file_hash(cache/(split+'.npz'))!=contract['sha256']:
            raise ValueError('Embedding cache checksum mismatch')
    x,y,m = load(cache,'train',metadata)
    vx,vy,vm = load(cache,'validation',metadata)
    # Train-only normalization, persisted exactly for inference.
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.01)
    x=((x-mean)/std).to(device);vx=((vx-mean)/std).to(device)
    y,m=y.to(device),m.to(device)
    positives=((y>.5)*m).sum(0)
    weight=((m.sum(0)-positives)/positives.clamp_min(1)).clamp(1,20)
    classes=metadata['classes']; results=[];best=-float('inf')
    configs=[dict(config,lr=config['lr']*lr_scale) for config in CONFIGS]
    protocol={'configs':configs,'epochs':epochs,'seed':42,'lr_scale':lr_scale,
              'selection':'validation supported macro AP; then validation F1 thresholds',
              'holdout_status':metadata['holdout_status'],'metadata':metadata,
              'rare_handling':'masked soft BCE; train-only hard prevalence weight clipped [1,20]; unsupported training classes retained'}
    write_json(output/'protocol.json',protocol)
    for config in configs:
        torch.manual_seed(42)
        model=Head(len(classes),config['hidden']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
        history=[];local_best=-float('inf');best_epoch=0
        generator=torch.Generator().manual_seed(42)
        for epoch in range(epochs):
            model.train();total=0.
            for ids in torch.randperm(len(x),generator=generator).split(128):
                loss=masked_loss(model(x[ids]),y[ids],m[ids],weight if config['weight'] else torch.ones_like(weight))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                total+=float(loss.detach())*len(ids)
            p=predict_head(model,vx)
            value=scores(vy,p,vm,torch.full((len(classes),),.5),classes)['macro_ap_supported']
            if value is None: raise ValueError('No supported validation classes')
            history.append({'epoch':epoch+1,'loss':total/len(x),'validation_ap':value})
            if value>local_best:
                local_best=value;best_epoch=epoch+1
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch+1>=20 and epoch+1-best_epoch>=12: break
        result={'config':config,'best_epoch':best_epoch,'validation_macro_ap':local_best,'history':history}
        results.append(result);write_json(output/(config['name']+'.json'),result)
        # Private diagnostic snapshots support future validation comparisons;
        # these are not threshold-calibrated deployable selected checkpoints.
        torch.save({'state_dict':state,'config':config,'classes':classes,'mean':mean,'std':std,
                    'encoder_weight_sha256':metadata['encoder_weight_sha256'],
                    'sample_rate':RATE,'context_seconds':metadata.get('context_seconds',5)},
                   output/(config['name']+'.pt'))
        print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
        if local_best>best:
            best=local_best
            checkpoint={'state_dict':state,'config':config,'classes':classes,'mean':mean,'std':std,
                        'encoder_weight_sha256':metadata['encoder_weight_sha256'],'sample_rate':RATE,
                        'context_seconds':metadata.get('context_seconds',5)}
    model=Head(len(classes),checkpoint['config']['hidden']).to(device);model.load_state_dict(checkpoint['state_dict'])
    thresholds=select_thresholds(vy,predict_head(model,vx),vm)
    checkpoint['thresholds']=thresholds
    torch.save(checkpoint,output/'selected.pt')
    write_json(output/'selection.json',{'candidates':results,'selected':checkpoint['config'],'thresholds':thresholds.tolist()})
    # Only the frozen winner is now evaluated on the already-observed holdout.
    tx,ty,tm=load(cache,'test',metadata)
    p=predict_head(model,((tx-mean)/std).to(device))
    final={'selected':checkpoint['config'],'validation_macro_ap':best,
           'holdout_status':metadata['holdout_status'],'test_validation_thresholds':scores(ty,p,tm,thresholds,classes),
           'test_fixed_point_five':scores(ty,p,tm,torch.full_like(thresholds,.5),classes),
           'training_positive_counts':positives.cpu().tolist(),'checkpoint_sha256':file_hash(output/'selected.pt')}
    # Actual save/load equivalence, no tolerance hidden by aggregate metrics.
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    replay=Head(len(classes),restored['config']['hidden']).to(device);replay.load_state_dict(restored['state_dict'])
    again=predict_head(replay,((tx-restored['mean'])/restored['std']).to(device))
    final['checkpoint_probability_max_difference']=float((again-p).abs().max())
    if not torch.equal(again,p):raise RuntimeError('Checkpoint prediction mismatch')
    write_json(output/'observed-holdout.json',final)
    return final


def infer(checkpoint_path, weights, audio, output, device='cpu'):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint.get('sample_rate')!=RATE or checkpoint.get('context_seconds') not in [1,5]:
        raise ValueError('Unsupported transfer preprocessing contract')
    count=len(checkpoint['classes'])
    if count<1 or len(set(checkpoint['classes']))!=count:
        raise ValueError('Checkpoint class order must be nonempty and unique')
    for name in ['mean','std','thresholds']:
        value=checkpoint[name]
        if value.shape != ((count,) if name=='thresholds' else (2048,)) or not torch.isfinite(value).all():
            raise ValueError('Invalid checkpoint '+name)
    if not (checkpoint['std']>0).all() or not ((checkpoint['thresholds']>=0)&(checkpoint['thresholds']<=1)).all():
        raise ValueError('Invalid checkpoint scaling or thresholds')
    if file_hash(weights)!=checkpoint['encoder_weight_sha256']:
        raise ValueError('Encoder weights differ from trained checkpoint')
    model=encoder(weights,device)
    head=Head(len(checkpoint['classes']),checkpoint['config']['hidden']).to(device)
    head.load_state_dict(checkpoint['state_dict']);head.eval()
    wave,rate=sf.read(audio,dtype='float32',always_2d=True);wave=wave.mean(1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Audio must contain finite samples')
    if rate!=RATE:
        gcd=math.gcd(rate,RATE);wave=resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
    rows=[]
    for lo in range(0,math.ceil(len(wave)/RATE),16):
        seconds=range(lo,min(math.ceil(len(wave)/RATE),lo+16))
        with torch.inference_mode():
            features=model(windows(wave,seconds,checkpoint['context_seconds']).to(device))['embedding'].cpu()
            p=predict_head(head,((features-checkpoint['mean'])/checkpoint['std']).to(device))
        if not torch.isfinite(p).all():raise RuntimeError('Nonfinite inference')
        for second,probability in zip(seconds,p):
            rows.append({'onset':second,'offset':min(second+1,len(wave)/RATE),
                         'probabilities':dict(zip(checkpoint['classes'],probability.tolist())),
                         'present':[c for c,v,t in zip(checkpoint['classes'],probability,checkpoint['thresholds']) if v>=t]})
    result={'offline_future_context':checkpoint['context_seconds']>1,'context_seconds':checkpoint['context_seconds'],
            'sample_rate':RATE,'observations':rows}
    write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--device',default='cuda');p.add_argument('--context-seconds',type=int,choices=[1,5],default=5)
    p=sub.add_parser('train');p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=80);p.add_argument('--device',default='cuda');p.add_argument('--lr-scale',type=float,default=1.)
    p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--weights',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='prepare':prepare(**args)
    elif command=='train':train(**args)
    else:
        args['checkpoint_path']=args.pop('checkpoint');infer(**args)


if __name__=='__main__':main()
