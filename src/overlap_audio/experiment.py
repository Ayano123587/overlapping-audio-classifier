"""Validation-only model selection; full-clip lazy mixtures and a new source holdout.

The initial published test source IDs are excluded from the v2 final test. Candidate
selection never constructs final-test audio. A final report is emitted once for the
selected checkpoint, with a separate extreme-prevalence stress evaluation.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .audio import load_wave, spectrogram, augment_wave, augment_spec
from .data import validate_sources
from .metrics import evaluate, thresholds_from_validation
from .model import FocalLoss


class LazyMixtures(Dataset):
    def __init__(self, manifest, split, count, seed, seconds=5., rare_probability=.03, excluded=()):
        validate_sources(manifest['records'])
        if count < 1 or seconds <= 0 or not 0 < rare_probability <= 1:
            raise ValueError('Invalid count/duration/prevalence')
        self.count, self.seed = count, seed
        self.samples = round(8000*seconds)
        self.classes = len(manifest['classes'])
        self.records = {k: [] for k in range(self.classes)}
        for row in manifest['records']:
            if row['split'] == split and row['source'] not in set(excluded):
                self.records[row['label']].append(row)
        if not all(self.records.values()):
            raise ValueError('Need independent source recordings for every class')
        self.cache = {row['path']: load_wave(row['path'], seconds=seconds)
                      for rows in self.records.values() for row in rows}
        self.probabilities = [.65,.35,.15,rare_probability]
        self.targets = torch.stack([self.plan(i)[0] for i in range(count)])

    def plan(self, index):
        rng = random.Random(self.seed + index*1000003)
        labels = [k for k,p in enumerate(self.probabilities) if rng.random() < p]
        if not labels:
            labels = [rng.choices(range(self.classes), weights=self.probabilities, k=1)[0]]
        target = torch.zeros(self.classes)
        components = []
        for label in labels:
            target[label] = 1
            row = rng.choice(self.records[label])
            components.append((row['path'], rng.randrange(self.samples), rng.uniform(.5,1.)))
        return target, components

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        target, components = self.plan(index)
        wave = torch.zeros(self.samples)
        for path, shift, gain in components:
            signal = self.cache[path]
            signal = signal / (signal.square().mean().sqrt()+1e-6)
            wave += torch.roll(signal,shift)*gain
        wave /= max(1.,float(wave.abs().max())/.9)
        return wave, target


class StrongCNN(nn.Module):
    def __init__(self, classes=4, width=24):
        super().__init__()
        blocks=[]
        channels=1
        for output in [width,width*2,width*4]:
            blocks += [nn.Conv2d(channels,output,3,padding=1),nn.BatchNorm2d(output),nn.GELU(),
                       nn.Conv2d(output,output,3,padding=1),nn.BatchNorm2d(output),nn.GELU(),nn.MaxPool2d(2)]
            channels=output
        self.features=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(channels*8,classes))

    def forward(self, spec):
        # Per-example log-power normalization avoids raw gain shortcuts.
        spec=(spec-spec.mean(dim=(-2,-1),keepdim=True))/(spec.std(dim=(-2,-1),keepdim=True)+1e-5)
        features=self.features(spec)
        pooled=nn.functional.adaptive_avg_pool2d(features,(4,1)).flatten(1)
        maximum=nn.functional.adaptive_max_pool2d(features,(4,1)).flatten(1)
        return self.head(torch.cat([pooled,maximum],dim=1))


def predict(model,dataset,device,batch_size=32):
    model.eval()
    output=[]
    with torch.no_grad():
        for wave,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            output.append(torch.sigmoid(model(spectrogram(wave.to(device)))).cpu())
    return torch.cat(output)


def fit_candidate(train,validation,config,seed,device,output):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=StrongCNN(width=config['width']).to(device)
    prevalence=train.targets.mean(0)
    if bool((prevalence == 0).any()):
        raise ValueError('Missing training class; increase mixture count')
    weight=(train.targets/(prevalence+.001)).sum(1)/train.targets.sum(1).clamp_min(1)
    sampler=WeightedRandomSampler(weight,len(train),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(train,batch_size=32,sampler=sampler)
    loss_fn=FocalLoss((1-prevalence).clamp(.1,.98),gamma=config['gamma']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    generator=torch.Generator().manual_seed(seed+100)
    best=-float('inf')
    history=[]
    for epoch in range(config['epochs']):
        model.train()
        total=0.
        for wave,target in loader:
            # Augment on CPU with an explicit generator, then transfer to GPU.
            spec=augment_spec(spectrogram(augment_wave(wave,generator)),generator).to(device)
            target=target.to(device)
            optimizer.zero_grad()
            loss=loss_fn(model(spec),target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total += float(loss.detach())*len(target)
        scheduler.step()
        probability=predict(model,validation,device)
        scores=evaluate(validation.targets,probability,torch.full((4,),.5))
        selection=scores['macro_average_precision']
        record={'epoch':epoch+1,'loss':total/len(train),'validation_macro_ap':selection}
        history.append(record)
        print(json.dumps(record),flush=True)
        if selection > best:
            best=selection
            torch.save(model.state_dict(),output/'best.pt')
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
    probability=predict(model,validation,device)
    thresholds=thresholds_from_validation(validation.targets,probability)
    result={'config':config,'seed':seed,'best_validation_macro_ap':best,'history':history,
            'validation':evaluate(validation.targets,probability,thresholds),
            'thresholds':thresholds.tolist(),'training_support':train.targets.sum(0).int().tolist(),
            'validation_support':validation.targets.sum(0).int().tolist()}
    (output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return model,thresholds,result


def run(manifest_path,previous_manifest_path,output,epochs=30,count=2048,device='auto'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final-test.json').exists():
        raise ValueError('Final test already evaluated; do not repeatedly tune this holdout')
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    previous=json.loads(Path(previous_manifest_path).read_text(encoding='utf-8'))
    excluded={r['source'] for r in previous['records'] if r['split']=='test'}
    final_records=[r for r in manifest['records'] if r['split']=='test' and r['source'] not in excluded]
    final_sources=sorted({r['source'] for r in final_records})
    if not final_sources:
        raise ValueError('Expanded manifest must supply unseen final-test sources')
    device='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    torch.set_num_threads(2)
    configs=[{'name':'focal-fullclip','width':24,'gamma':2.,'lr':.001,'epochs':epochs},
             {'name':'weighted-bce-fullclip','width':24,'gamma':0.,'lr':.001,'epochs':epochs},
             {'name':'focal-wider-fullclip','width':32,'gamma':2.,'lr':.0005,'epochs':epochs}]
    protocol={'selection_metric':'validation macro average precision','final_sources':final_sources,
              'excluded_previously_observed_test_sources':sorted(excluded),'seconds':5.,'sample_rate':8000,
              'train_count':count,'validation_count':1024,'final_count':4096,'seed':42,
              'configs':configs,'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              'device':device,'test_policy':'constructed only after configuration selection; no final-test-based changes'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    training=LazyMixtures(manifest,'train',count,42)
    validation=LazyMixtures(manifest,'validation',1024,43)
    candidates=[]
    best_score=-float('inf')
    for config in configs:
        folder=output/config['name'];folder.mkdir(exist_ok=True)
        model,thresholds,result=fit_candidate(training,validation,config,42,device,folder)
        candidates.append(result)
        if result['best_validation_macro_ap'] > best_score:
            best_score=result['best_validation_macro_ap']
            torch.save({'state_dict':model.cpu().state_dict(),'width':config['width'],
                        'thresholds':thresholds,'classes':manifest['classes'],'sample_rate':8000,'seconds':5.,
                        'config':config},output/'selected.pt')
    checkpoint=torch.load(output/'selected.pt',map_location=device,weights_only=True)
    selected=StrongCNN(width=checkpoint['width']).to(device)
    selected.load_state_dict(checkpoint['state_dict'])
    # Final test remains absent from candidate fitting and validation selection.
    test=LazyMixtures(manifest,'test',4096,44,excluded=excluded)
    probability=predict(selected,test,device)
    stress=LazyMixtures(manifest,'test',8192,45,rare_probability=.001,excluded=excluded)
    stress_probability=predict(selected,stress,device)
    final={'selected_config':checkpoint['config'],'protocol':protocol,'candidate_validation':candidates,
           'test':evaluate(test.targets,probability,checkpoint['thresholds']),
           'test_support':test.targets.sum(0).int().tolist(),
           'test_always_present':evaluate(test.targets,torch.ones_like(test.targets),torch.full((4,),.5)),
           'extreme_prevalence_stress':evaluate(stress.targets,stress_probability,checkpoint['thresholds']),
           'extreme_support':stress.targets.sum(0).int().tolist(),
           'limitations':['Real isolated clips are synthetically mixed, not naturally annotated overlapping scenes',
                          'Within-split mixtures reuse recordings and are not independent real-world samples',
                          'Stress prevalence is configured; inspect realized rare positive/negative supports',
                          'Full five-second clip labels avoid the v1 one-second crop mismatch',
                          'Single seed; source diversity and external-domain evaluation remain required']}
    (output/'final-test.json').write_text(json.dumps(final,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(final,indent=2),flush=True)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--previous-manifest',required=True)
    parser.add_argument('--output',default='models/validation-v2')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--count',type=int,default=2048)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    run(args.manifest,args.previous_manifest,args.output,args.epochs,args.count,args.device)
