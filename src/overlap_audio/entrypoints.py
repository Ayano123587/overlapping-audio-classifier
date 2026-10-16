"""Explicit train/validation fitting and separately acknowledged evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from .learned import CONFIGS, SEEDS, fit_head, bundle_probabilities
from .postprocess import aligned_cache, apply_policy, validate_bundle, lookahead
from .natural import scores, select_thresholds, file_hash
from .transfer import RATE, write_json


def train_only(cache, output, config='control', seeds=SEEDS, epochs=80, device='cpu'):
    """Fit a prespecified configuration; never open a test cache."""
    output, cache = Path(output), Path(cache)
    if output.exists():
        raise ValueError('Output exists; choose a fresh training directory')
    choices = {c['name']: c for c in CONFIGS}
    if config not in choices or epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid configuration, epochs, or seeds')
    metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('context_seconds') != 5:
        raise ValueError('Training requires the pinned five-second feature cache')
    x,y,m,_,_ = aligned_cache(cache,'train',metadata)
    vx,vy,vm,ids,seconds = aligned_cache(cache,'validation',metadata)
    output.mkdir(parents=True)
    components, histories = [], []
    tensors = [torch.from_numpy(a) for a in (x,y,m,vx,vy,vm)]
    for seed in seeds:
        component, history = fit_head(*tensors,metadata['classes'],choices[config],seed,epochs,device)
        component['encoder_weight_sha256'] = metadata['encoder_weight_sha256']
        components.append(component); histories.append(history)
        torch.save(component,output/f'seed-{seed}.pt')
        write_json(output/f'seed-{seed}.json',history)
    bundle = {'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'classes':metadata['classes'],
              'encoder_weight_sha256':metadata['encoder_weight_sha256'],'components':components,
              'thresholds':torch.full((len(metadata['classes']),),.5),
              'policy':{'branches':[{'weights':[1/len(seeds)]*len(seeds),'filters':[]}],
                        'mix_weights':[1.], 'filters':[]}}
    probability = apply_policy(bundle_probabilities(bundle,{5:vx}),bundle['policy'],ids,seconds)
    bundle['thresholds'] = select_thresholds(tensors[4],torch.from_numpy(probability),tensors[5])
    bundle['lookahead_seconds'] = lookahead(bundle['policy'],components)
    validate_bundle(bundle); torch.save(bundle,output/'selected.pt')
    report = {'config':choices[config],'seeds':list(seeds),'test_labels_loaded':False,
              'metadata_sha256':file_hash(cache/'metadata.json'),
              'input_hashes':{s:file_hash(cache/(s+'.npz')) for s in ('train','validation')},
              'validation':scores(tensors[4],torch.from_numpy(probability),tensors[5],bundle['thresholds'],metadata['classes']),
              'checkpoint_sha256':file_hash(output/'selected.pt')}
    write_json(output/'training-report.json',report)
    return report


def evaluate(checkpoint, caches, output, split='validation', acknowledge_observed_test=False):
    """Evaluate a saved policy without training, selection, or overwriting."""
    if split not in ('validation','test'):
        raise ValueError('Evaluation split must be validation or test')
    if split == 'test' and not acknowledge_observed_test:
        raise ValueError('Test is already observed; explicit acknowledgement is required')
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new report path')
    bundle = torch.load(checkpoint,map_location='cpu',weights_only=True)
    validate_bundle(bundle)
    contexts = {c['context_seconds'] for c in bundle['components']}
    features, reference, hashes, source_contract = {}, None, {}, None
    for context in sorted(contexts):
        if context not in caches:
            raise ValueError(f'Missing {context}-second cache')
        root = Path(caches[context])
        metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('context_seconds') != context or metadata['classes'] != bundle['classes'] or metadata['encoder_weight_sha256'] != bundle['encoder_weight_sha256']:
            raise ValueError('Cache/checkpoint feature contract mismatch')
        contract = (metadata.get('assignment'),metadata.get('sources'))
        if source_contract is not None and contract != source_contract:
            raise ValueError('Cross-context source/split contract mismatch')
        source_contract = contract
        x,y,m,ids,seconds = aligned_cache(root,split,metadata)
        current = (y,m,ids,seconds)
        if reference is not None and not all(np.array_equal(a,b) for a,b in zip(reference,current)):
            raise ValueError('Cross-context recording/label alignment mismatch')
        reference = current; features[context] = x
        hashes[str(context)] = {'metadata':file_hash(root/'metadata.json'),'split':file_hash(root/(split+'.npz'))}
    y,m,ids,seconds = reference
    probability = apply_policy(bundle_probabilities(bundle,features),bundle['policy'],ids,seconds)
    report = {'split':split,'status':'Observed test; no untouched claim' if split=='test' else 'Validation; selection data',
              'rows':len(y),'checkpoint_sha256':file_hash(checkpoint),'inputs':hashes,
              'metrics':scores(torch.from_numpy(y),torch.from_numpy(probability),torch.from_numpy(m),bundle['thresholds'],bundle['classes'])}
    output.parent.mkdir(parents=True,exist_ok=True); write_json(output,report)
    return report
