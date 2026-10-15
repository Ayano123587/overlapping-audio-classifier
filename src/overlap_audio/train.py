import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from .audio import spectrogram, augment_wave, augment_spec
from .data import mixtures
from .metrics import evaluate, thresholds_from_validation
from .model import SpectrogramCNN, FocalLoss

def train(manifest_path, checkpoint, report, epochs=12, count=512, seed=42, threads=2, rare_probability=.03):
    if epochs<1 or count<32: raise ValueError("epochs >= 1 and count >= 32 required")
    torch.set_num_threads(threads);torch.manual_seed(seed);random.seed(seed)
    manifest=json.loads(Path(manifest_path).read_text())
    datasets={s:mixtures(manifest,s,count if s=="train" else max(128,count//2),seed+i,rare_probability=rare_probability) for i,s in enumerate(["train","validation","test"])}
    x,y,_=datasets["train"];positive=y.sum(0)
    if bool((positive==0).any()): raise ValueError("Training mixtures omitted a class; increase count")
    prevalence=y.mean(0);alpha=(1-prevalence).clamp(.1,.95)
    weights=(y/(prevalence+.001)).sum(1)/y.sum(1).clamp_min(1)
    sampler=WeightedRandomSampler(weights,len(y),replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(TensorDataset(x,y),batch_size=32,sampler=sampler)
    model=SpectrogramCNN(len(manifest["classes"]));loss_fn=FocalLoss(alpha)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(seed+10)
    history=[]
    for epoch in range(epochs):
        model.train();total=0.
        for waveform,target in loader:
            augmented=augment_wave(waveform,generator)
            logits=model(augment_spec(spectrogram(augmented),generator))
            loss=loss_fn(logits,target);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(loss.detach())*len(target)
        history.append(total/len(y))
    def probabilities(waveforms):
        model.eval()
        with torch.no_grad(): return torch.cat([torch.sigmoid(model(spectrogram(batch))) for batch in waveforms.split(32)])
    vx,vy,_=datasets["validation"];tx,ty,_=datasets["test"]
    thresholds=thresholds_from_validation(vy,probabilities(vx))
    test_probs=probabilities(tx)
    result={"dataset":manifest["kind"],"classes":manifest["classes"],"seed":seed,"epochs":epochs,"training_mixtures":len(y),"rare_class_inclusion_probability":rare_probability,"training_positive_support":positive.int().tolist(),"training_positive_ratio":float(positive.max()/positive.min()),"loss":history,"thresholds_selected_on_validation":thresholds.tolist(),"test":evaluate(ty,test_probs,thresholds),"baseline_always_present":evaluate(ty,torch.ones_like(ty),torch.full_like(thresholds,.5)),"source_counts":{s:len({r["source"] for r in manifest["records"] if r["split"]==s}) for s in datasets},"source_disjoint":True,"limitations":["Mixture labels are synthetic presence labels, not natural recording annotations.","This is clip classification, not waveform source separation or event localization.","Small center-cropped subset metrics cannot establish generalization to field recordings.","Configured rare-class rate and realized supports must be reviewed; a smoke run is not evidence at 1:1000 prevalence."]}
    path=Path(checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"classes":manifest["classes"],"thresholds":thresholds,"sample_rate":8000,"seconds":1.},path)
    destination=Path(report);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2));return result

def load_checkpoint(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    model=SpectrogramCNN(len(checkpoint["classes"]));model.load_state_dict(checkpoint["state_dict"]);model.eval()
    return model,checkpoint
