"""Cold-start reproduction of the fixed selected MAESTRO recipe, without test access."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import torch
from .learned import fit_head, bundle_probabilities
from .natural import file_hash, scores, select_thresholds
from .postprocess import aligned_cache, apply_policy, validate_bundle, lookahead
from .transfer import RATE, WEIGHT_URL, write_json

ENCODER_SHA256 = '0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31'
COHORT_SHA256 = '7bcca054d341dd1e58dac0042ccb33829333bb6567ad5c1e3d2af314a9660d91'
CONFIG = {'name':'mlp-balanced','hidden':256,'weight':True,'lr':.0003,'dropout':.3,'weight_decay':.01}
POLICY = {'name':'fixed-five-second-box3','branches':[{'weights':[1.], 'filters':[]}],
          'mix_weights':[1.], 'filters':[{'kind':'box','width':3}]}


def acquire_weights(destination):
    """Acquire the pinned published encoder bytes without allocating an encoder."""
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) != ENCODER_SHA256:
            raise ValueError('Existing encoder SHA256 mismatch; refusing overwrite')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial = destination.with_name(destination.name+'.part')
    if partial.exists():
        raise ValueError('Partial encoder file exists; inspect it before retrying')
    urllib.request.urlretrieve(WEIGHT_URL,partial)
    if file_hash(partial) != ENCODER_SHA256:
        raise ValueError('Downloaded encoder SHA256 mismatch; partial preserved')
    partial.replace(destination)
    return destination


def cohort_fingerprint(metadata):
    """Hash source identities and semantic contracts, excluding ZIP container hashes."""
    keys = ('classes','assignment','sources','encoder_weight_sha256','sample_rate','context_seconds','split')
    semantic = {k:metadata[k] for k in keys}
    semantic['contracts'] = {split:{k:v for k,v in contract.items() if k != 'sha256'}
                             for split,contract in metadata['contracts'].items()}
    return hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def checked_metadata(cache):
    """Require the complete audited source/split receipt, not an arbitrary subset."""
    cache = Path(cache)
    metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if metadata['encoder_weight_sha256'] != ENCODER_SHA256 or metadata['context_seconds'] != 5 or metadata['sample_rate'] != RATE:
        raise ValueError('Selected encoder/preprocessing contract mismatch')
    for split, rows, records in [('train',6838,29),('validation',2163,10),('test',2368,10)]:
        contract = metadata['contracts'][split]
        if contract['rows'] != rows or len(contract['record_ids']) != records:
            raise ValueError('Incomplete selected-recipe cohort')
    # Test receipt is checked as metadata only: no test NPZ or labels are opened.
    groups = [set(metadata['contracts'][s]['record_ids']) for s in ('train','validation','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)) or len(metadata['sources']) != 49:
        raise ValueError('Invalid recording separation/source coverage')
    if cohort_fingerprint(metadata) != COHORT_SHA256:
        raise ValueError('Selected recipe requires the pinned full-cohort source receipt')
    return metadata


def train_selected(cache, output, device='cpu'):
    """Fit one seed-42 head, earliest best validation AP, then fixed box3 thresholds.

    The 80-epoch ceiling, minimum 20 epochs and patience 12 match the retained
    head's original recipe. No candidate search or held-out evaluation occurs.
    Floating-point environments may produce different weights and stopping epochs.
    """
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a fresh selected training directory')
    metadata = checked_metadata(cache)
    x,y,m,_,_ = aligned_cache(cache,'train',metadata)
    vx,vy,vm,ids,seconds = aligned_cache(cache,'validation',metadata)
    if len(x) != 6838 or len(vx) != 2163:
        raise ValueError('Loaded cohort row count differs from the selected recipe')
    output.mkdir(parents=True)
    receipt = {'recipe':'fixed-mlp-balanced-seed42-box3','config':CONFIG,'seed':42,
               'epochs':80,'minimum_epochs':20,'patience':12,'batch_size':128,
               'checkpoint_rule':'earliest maximum unsmoothed validation supported macro AP',
               'policy':POLICY,'threshold_rule':'validation F1 after fixed smoothing',
               'metadata_sha256':file_hash(cache/'metadata.json'),'cohort_sha256':cohort_fingerprint(metadata),
               'input_hashes':{s:file_hash(cache/(s+'.npz')) for s in ('train','validation')},
               'test_labels_loaded':False,'device':device,'torch_version':str(torch.__version__),
               'source_hashes':{name:file_hash(Path(__file__).parent/name) for name in
                                ('selected_recipe.py','learned.py','transfer.py','postprocess.py','natural.py')}}
    write_json(output/'protocol.json',receipt)
    tensors = [torch.from_numpy(a) for a in (x,y,m,vx,vy,vm)]
    component, history = fit_head(*tensors,metadata['classes'],dict(CONFIG),42,80,device)
    component['encoder_weight_sha256'] = ENCODER_SHA256
    bundle = {'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'classes':metadata['classes'],
              'encoder_weight_sha256':ENCODER_SHA256,'components':[component],
              'thresholds':torch.full((len(metadata['classes']),),.5),'policy':POLICY}
    probability = apply_policy(bundle_probabilities(bundle,{5:vx}),POLICY,ids,seconds)
    bundle['thresholds'] = select_thresholds(tensors[4],torch.from_numpy(probability),tensors[5])
    bundle['lookahead_seconds'] = lookahead(POLICY,[component])
    validate_bundle(bundle)
    torch.save(bundle,output/'selected.pt')
    restored = torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    replay = apply_policy(bundle_probabilities(restored,{5:vx}),restored['policy'],ids,seconds)
    difference = float(abs(replay-probability).max())
    if difference != 0:
        raise RuntimeError('Saved selected checkpoint prediction mismatch')
    report = dict(receipt,history=history,checkpoint_sha256=file_hash(output/'selected.pt'),
                  checkpoint_probability_max_difference=difference,
                  validation=scores(tensors[4],torch.from_numpy(probability),tensors[5],bundle['thresholds'],metadata['classes']))
    write_json(output/'training-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    weights = sub.add_parser('weights'); weights.add_argument('--output',required=True)
    train = sub.add_parser('train'); train.add_argument('--cache',required=True); train.add_argument('--output',required=True); train.add_argument('--device',default='cpu')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'weights': acquire_weights(args['output'])
    else: train_selected(**args)

if __name__ == '__main__': main()
