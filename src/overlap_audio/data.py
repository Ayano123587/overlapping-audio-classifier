import csv
import hashlib
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .audio import load_wave, write_wave, SAMPLES, SAMPLE_RATE

CLASSES=["dog","rain","crying_baby","clock_tick"]

def download(url,path):
    request=urllib.request.Request(url,headers={"User-Agent":"overlap-audio-research/0.1"})
    with urllib.request.urlopen(request,timeout=60) as response: data=response.read()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def fetch_esc(output, per_fold=2, revision="33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"):
    if per_fold < 1: raise ValueError("per_fold must be positive")
    """Download a small ESC-10 subset, with official folds and original source IDs."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    base=f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{revision}"
    download(base+"/meta/esc50.csv",output/"esc50.csv")
    download(base+"/LICENSE",output/"LICENSE-ESC50")
    download(base+"/README.md",output/"README-ESC50.md")
    records=[];counts=defaultdict(int)
    for row in csv.DictReader((output/"esc50.csv").open(encoding="utf-8")):
        key=(row["category"],row["fold"])
        if row["category"] not in CLASSES or row["esc10"]!="True" or counts[key]>=per_fold: continue
        path=output/"audio"/row["filename"]
        digest=download(base+"/audio/"+row["filename"],path)
        fold=int(row["fold"]);split="train" if fold<=3 else "validation" if fold==4 else "test"
        records.append({"path":str(path.resolve()),"label":CLASSES.index(row["category"]),"source":row["src_file"],"split":split,"sha256":digest,"filename":row["filename"]})
        counts[key]+=1
    if len(records)!=len(CLASSES)*5*per_fold: raise ValueError("Requested ESC-10 subset could not be fulfilled")
    manifest={"kind":"ESC-10 real-audio subset; synthetic mixtures","classes":CLASSES,"revision":revision,"records":records,"license":"ESC-10 CC BY; retain upstream individual attributions in downloaded LICENSE"}
    validate_sources(records)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest

def validate_sources(records):
    assignment={}
    for row in records:
        if row["split"] not in {"train","validation","test"}: raise ValueError("Unknown split")
        if row["source"] in assignment and assignment[row["source"]]!=row["split"]: raise ValueError("Original source crosses splits")
        assignment[row["source"]]=row["split"]

def synthetic(output, seed=42):
    output=Path(output);rng=np.random.default_rng(seed);records=[]
    for split,count in [("train",20),("validation",6),("test",6)]:
        for label,freq in enumerate([250,700,1400,2600]):
            for i in range(count):
                source=f"{split}-{label}-{i}";t=np.arange(SAMPLES)/SAMPLE_RATE
                signal=.4*np.sin(2*np.pi*(freq+rng.uniform(-25,25))*t+rng.uniform(0,6.28))
                signal*=.5+.5*np.sin(2*np.pi*rng.uniform(1,4)*t)**2
                signal+=rng.normal(0,.02,SAMPLES)
                path=output/"audio"/(source+".wav");write_wave(path,signal)
                records.append({"path":str(path.resolve()),"label":label,"source":source,"split":split})
    manifest={"kind":"synthetic modulated tones only; not animal-call performance","classes":["low_tone","mid_tone","high_tone","rare_high_tone"],"seed":seed,"records":records}
    output.mkdir(parents=True,exist_ok=True);(output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest

def mixtures(manifest, split, count, seed=42, imbalance=True, rare_probability=.03):
    """Split sources BEFORE mixing. Rare-class probability is 0.03 vs 0.65."""
    if not 0 < rare_probability <= 1: raise ValueError("rare_probability must be in (0,1]")
    validate_sources(manifest["records"])
    rng=random.Random(seed);by_class=defaultdict(list)
    for row in manifest["records"]:
        if row["split"]==split: by_class[row["label"]].append(row)
    classes=len(manifest["classes"])
    if set(by_class)!=set(range(classes)): raise ValueError("Every split needs at least one source per class")
    cache={r["path"]:load_wave(r["path"]) for rows in by_class.values() for r in rows}
    waveforms=[];targets=[];provenance=[]
    probabilities=[.65,.35,.15,rare_probability] if imbalance and classes==4 else [.4]*classes
    for _ in range(count):
        labels=[c for c,p in enumerate(probabilities) if rng.random()<p]
        if not labels: labels=[rng.choices(range(classes),weights=probabilities,k=1)[0]]
        wave=torch.zeros(SAMPLES);target=torch.zeros(classes);sources=[]
        for label in labels:
            row=rng.choice(by_class[label]);signal=cache[row["path"]]
            # RMS normalization plus relative gain makes labels audible without clipping.
            signal=signal/(signal.square().mean().sqrt()+1e-6)
            wave+=torch.roll(signal,rng.randrange(SAMPLES))*rng.uniform(.5,1.)
            target[label]=1;sources.append(row["source"])
        wave/=max(1.,float(wave.abs().max())/.9)
        waveforms.append(wave);targets.append(target);provenance.append(sources)
    return torch.stack(waveforms),torch.stack(targets),provenance
