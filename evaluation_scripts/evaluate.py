"""Explicit saved-policy evaluation. Validation is the default."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from overlap_audio.entrypoints import evaluate

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--cache-five',required=True)
    p.add_argument('--cache-one');p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--acknowledge-observed-test',action='store_true')
    args=vars(p.parse_args());five=args.pop('cache_five');one=args.pop('cache_one')
    args['caches']={5:five}
    if one:args['caches'][1]=one
    evaluate(**args)
