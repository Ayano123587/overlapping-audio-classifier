"""Train a five-second cached-feature ensemble on train/validation only."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from overlap_audio.entrypoints import train_only

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',choices=['control','slower','dropout','decay'],default='control')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,1337,2026])
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--device',default='cpu')
    train_only(**vars(p.parse_args()))
