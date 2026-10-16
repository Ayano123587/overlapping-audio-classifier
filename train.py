"""Train the fixed selected recipe, or an explicitly requested experimental control."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from overlap_audio.entrypoints import train_only
from overlap_audio.selected_recipe import train_selected

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--recipe',choices=['selected','experiment'],default='selected')
    p.add_argument('--config',choices=['control','slower','dropout','decay'])
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--epochs',type=int);p.add_argument('--device',default='cpu')
    args=vars(p.parse_args()); recipe=args.pop('recipe')
    if recipe=='selected':
        overrides = [args.pop(k) for k in ('config','seeds','epochs')]
        if any(v is not None for v in overrides):
            p.error('Selected recipe is fixed; use --recipe experiment for overrides')
        train_selected(**args)
    else:
        args={k:v for k,v in args.items() if v is not None}
        train_only(**args)
