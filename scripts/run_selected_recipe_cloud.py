"""Private CPU reproduction from hash-verified plain Python and train/validation data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def run(input_root=Path('/kaggle/input'),working=Path('/kaggle/working')):
    started=time.time(); manifests=list(input_root.rglob('source-manifest.json'))
    if len(manifests)!=1:raise ValueError('Require one exclusive selected-recipe source manifest')
    source=manifests[0].parent;manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
    if manifest['test_arrays_staged'] is not False:raise ValueError('Test arrays forbidden')
    destination=working/'selected-recipe-source'
    destination.mkdir(exist_ok=False)
    seen=set()
    for row in manifest['files']:
        name=row['path'];relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or name in seen:raise ValueError('Unsafe source manifest')
        seen.add(name)
        if name.startswith('cache/') and name not in {'cache/metadata.json','cache/train.npz','cache/validation.npz'}:
            raise ValueError('Unexpected cache input')
        content=(source/row['flat_name']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Source/input checksum mismatch')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    runner=destination/'scripts/run_selected_recipe_cloud.py'
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=hashlib.sha256(runner.read_bytes()).hexdigest():
        raise ValueError('Dispatched runner differs from committed source snapshot')
    sys.path.insert(0,str(destination/'src'))
    import torch
    from overlap_audio.selected_recipe import train_selected
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    report=train_selected(destination/'cache',working/'selected-recipe-reproduction-v1',device='cpu')
    receipt={'source_commit':manifest['source_commit'],'source_manifest_sha256':hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
             'runtime_seconds':time.time()-started,'threads':torch.get_num_threads(),'test_labels_loaded':False,
             'checkpoint_sha256':report['checkpoint_sha256'],'validation':report['validation']}
    (working/'execution-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':run()
