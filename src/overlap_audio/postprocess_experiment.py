"""Private, no-refit GPU regeneration and validation-only postprocessing study."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time
from urllib.request import Request,urlopen
import zipfile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .fetch_maestro import RECORD,FILES,digest
from .transfer import RATE,encoder,windows,write_json,file_hash
from .natural import scores
from .postprocess import cpu_study,aligned_cache,component_predictions,load_components,select_combination,cached_selected,infer


def unique_input(root,parent,name):
    paths=[p for p in Path(root).rglob(name) if p.parent.name==parent]
    if len(paths)!=1:raise ValueError('Missing/ambiguous private input: '+parent+'/'+name)
    return paths[0]


def audio_only(root):
    """Fetch original complete audio/license; never extract/open annotation files."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);receipt=[]
    for name in ['development_audio.zip','LICENSE.txt']:
        path=root/name
        with urlopen(Request(RECORD+name+'?download=1',headers={'User-Agent':'private-noncommercial-audio-study'}),timeout=60) as response,path.open('wb') as handle:
            for block in iter(lambda:response.read(1024*1024),b''):handle.write(block)
        if digest(path)!=FILES[name]:raise ValueError('Official source checksum mismatch')
        receipt.append({'file':name,'url':RECORD+name+'?download=1','md5':FILES[name],'sha256':file_hash(path),'bytes':path.stat().st_size})
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not (root/item.filename).resolve().is_relative_to(root.resolve()):raise ValueError('Archive path escapes directory')
                archive.extractall(root)
    return receipt


def waveform_probabilities(raw,ids,seconds,metadata,component,model,device='cuda',retain_features=False):
    """Exact original five-second transform; no label/mask inputs."""
    output=np.zeros((len(ids),len(metadata['classes'])),np.float32)
    retained=np.zeros((len(ids),2048),np.float32) if retain_features else None
    sources={r['id']:r for r in metadata['sources']}
    for identifier in dict.fromkeys(ids.tolist()):
        positions=np.flatnonzero(ids==identifier)
        if not np.array_equal(seconds[positions],np.arange(sources[identifier]['seconds'])):raise ValueError('Recording time contract mismatch')
        paths=list((Path(raw)/'development_audio').rglob(identifier+'.wav'))
        if len(paths)!=1 or file_hash(paths[0])!=sources[identifier]['audio_sha256']:raise ValueError('Recording source hash mismatch')
        wave,rate=sf.read(paths[0],dtype='float32',always_2d=True);wave=wave.mean(1)
        if rate!=RATE:
            gcd=math.gcd(rate,RATE);wave=resample_poly(wave,RATE//gcd,rate//gcd).astype(np.float32)
        features=[]
        with torch.inference_mode():
            for lo in range(0,len(positions),16):
                batch=windows(wave,range(lo,min(len(positions),lo+16)),5).to(device)
                features.append(model(batch)['embedding'].float().cpu().numpy())
        features=np.concatenate(features)
        output[positions]=component_predictions(component,features,'cpu')
        if retain_features:retained[positions]=features
        print(json.dumps({'five_second_prediction_record':identifier,'rows':len(positions)}),flush=True)
    return (output,retained) if retain_features else output


def run(input_root,output,temporary):
    start=time.time();output=Path(output);temporary=Path(temporary)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; stop without CPU fallback')
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh private run directory')
    output.mkdir(parents=True,exist_ok=True);temporary.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cache=unique_input(input_root,'private-embeddings','metadata.json').parent
    metadata=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    heads=[unique_input(input_root,parent,name+'.pt') for parent in ['transfer-central-v3','transfer-lr-v4'] for name in ['linear-soft','linear-balanced','mlp-balanced']]
    five_path=unique_input(input_root,'transfer-v2','selected.pt')
    five=load_components([five_path],metadata,context=5)[0]
    weights=[p for p in Path(input_root).rglob('Cnn14.pth') if file_hash(p)==metadata['encoder_weight_sha256']]
    if not weights:raise ValueError('Exact original encoder missing from private inputs')
    environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),
                 'cpu_stage_replay':True,'five_second_checkpoint_sha256':file_hash(five_path),'encoder_weight_sha256':file_hash(weights[0])}
    write_json(output/'environment.json',environment)
    study=cpu_study(cache,heads,output/'cpu-validation')
    with np.load(output/'cpu-validation'/'validation-probabilities.npz',allow_pickle=False) as data:central=data['probabilities']
    states=torch.load(output/'cpu-validation'/'central-components.pt',map_location='cpu',weights_only=True)['components']
    # No test cache or annotation file has been opened so far.
    provenance=audio_only(temporary/'raw');write_json(output/'source-provenance.json',provenance)
    shutil.copy2(temporary/'raw'/'LICENSE.txt',output/'MAESTRO-LICENSE.txt')
    model=encoder(weights[0],'cuda')
    x,y,m,ids,seconds=aligned_cache(cache,'validation',metadata)
    five_p,val_features=waveform_probabilities(temporary/'raw',ids,seconds,metadata,five,model,retain_features=True)
    bundle,selection=select_combination(central,five_p,y,m,ids,seconds,study,states+[five],output/'selected')
    selection_hash=file_hash(output/'selected'/'selection.json')
    # This is the first test-label access. The frozen selection is immutable.
    tx,ty,tm,tids,tseconds=aligned_cache(cache,'test',metadata)
    central_test=np.stack([component_predictions(c,tx) for c in states])
    five_test,test_features=waveform_probabilities(temporary/'raw',tids,tseconds,metadata,five,model,retain_features=True)
    all_test=np.concatenate([central_test,five_test[None]],axis=0)
    restored=torch.load(output/'selected'/'selected.pt',map_location='cpu',weights_only=True)
    p=cached_selected(all_test,restored,tids,tseconds)
    if not np.array_equal(p,cached_selected(all_test,bundle,tids,tseconds)):raise RuntimeError('Actual checkpoint replay mismatch')
    final={'selection':selection,'observed_holdout_status':'Previously observed; no untouched claim',
           'selected':scores(torch.from_numpy(ty),torch.from_numpy(p),torch.from_numpy(tm),bundle['thresholds'],metadata['classes']),
           'five_second_baseline':scores(torch.from_numpy(ty),torch.from_numpy(five_test),torch.from_numpy(tm),five['thresholds'],metadata['classes']),
           'selection_sha256_before_test_loading':selection_hash,'selection_unchanged_after_test':file_hash(output/'selected'/'selection.json')==selection_hash,
           'checkpoint_replay_max_difference':0.,'test_rows':len(ty),'test_recordings':len(set(tids.tolist()))}
    write_json(output/'observed-holdout.json',final)
    # Actual production waveform inference on an authentic validation crop.
    identifier=ids[0];source=next((temporary/'raw'/'development_audio').rglob(str(identifier)+'.wav'))
    wave,rate=sf.read(source,dtype='float32',always_2d=True)
    probe=temporary/'probe.wav';sf.write(probe,wave[:rate*5],rate,subtype='FLOAT')
    proof=infer(output/'selected'/'selected.pt',weights[0],probe,temporary/'probe-prediction.json','cuda')
    probabilities=[value for row in proof['observations'] for value in row['probabilities'].values()]
    write_json(output/'actual-inference-proof.json',{'rows':len(proof['observations']),'probabilities':len(probabilities),'finite_in_unit_interval':bool(np.isfinite(probabilities).all() and min(probabilities)>=0 and max(probabilities)<=1),
              'probe_source_record':str(identifier),'accuracy_claim':False,'lookahead_seconds':proof['lookahead_seconds'],'checkpoint_replay_max_difference':0.})
    # Prospective user-authorized learned-head studies reuse exact authentic
    # five-second features. Build train now; retain already computed val/test.
    train_x,train_y,train_m,train_ids,train_seconds=aligned_cache(cache,'train',metadata)
    unused,train_features=waveform_probabilities(temporary/'raw',train_ids,train_seconds,metadata,five,model,retain_features=True)
    five_cache=output/'five-second-cache';five_cache.mkdir()
    new_metadata=json.loads(json.dumps(metadata));new_metadata['context_seconds']=5
    for split,features,targets,masks,record_ids in [
        ('train',train_features,train_y,train_m,train_ids),('validation',val_features,y,m,ids),
        ('test',test_features,ty,tm,tids)]:
        path=five_cache/(split+'.npz')
        np.savez_compressed(path,features=features,targets=targets,masks=masks,record_ids=record_ids)
        new_metadata['contracts'][split]={'sha256':file_hash(path),'rows':len(features),'record_ids':sorted(set(record_ids.tolist()))}
    write_json(five_cache/'metadata.json',new_metadata)
    write_json(output/'completion.json',{'exit':0,'elapsed_seconds':time.time()-start,'selection_sha256':selection_hash})
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',default='/kaggle/input');parser.add_argument('--output',required=True);parser.add_argument('--temporary',required=True)
    run(**vars(parser.parse_args()))
