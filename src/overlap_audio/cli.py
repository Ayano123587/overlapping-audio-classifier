import argparse
import json
from pathlib import Path
import torch
from .audio import load_wave,spectrogram
from .data import fetch_esc,synthetic
from .train import train,load_checkpoint

def main(argv=None):
    parser=argparse.ArgumentParser(description="Train and use overlapping sound presence classifiers")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("synthetic");p.add_argument("--output",default="data/raw/synthetic")
    p=sub.add_parser("fetch-esc");p.add_argument("--output",default="data/external/esc10");p.add_argument("--per-fold",type=int,default=2)
    p=sub.add_parser("train");p.add_argument("manifest");p.add_argument("--checkpoint",default="models/audio.pt");p.add_argument("--report",default="docs/results.json");p.add_argument("--epochs",type=int,default=12);p.add_argument("--count",type=int,default=512);p.add_argument("--rare-probability",type=float,default=.03)
    p=sub.add_parser("predict");p.add_argument("checkpoint");p.add_argument("wave");p.add_argument("--output")
    args=parser.parse_args(argv)
    if args.command=="synthetic": synthetic(args.output)
    elif args.command=="fetch-esc": fetch_esc(args.output,args.per_fold)
    elif args.command=="train": train(args.manifest,args.checkpoint,args.report,args.epochs,args.count,rare_probability=args.rare_probability)
    else:
        model,checkpoint=load_checkpoint(args.checkpoint)
        with torch.no_grad(): probs=torch.sigmoid(model(spectrogram(load_wave(args.wave))))[0]
        result={label:{"probability":float(p),"present":bool(p>=threshold)} for label,p,threshold in zip(checkpoint["classes"],probs,checkpoint["thresholds"])}
        text=json.dumps(result,indent=2)
        if args.output: Path(args.output).write_text(text+"\n")
        print(text)
if __name__=="__main__":main()
