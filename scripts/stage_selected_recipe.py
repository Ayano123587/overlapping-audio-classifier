"""Stage a private plain-source selected-recipe run; no encoder or test arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from overlap_audio.selected_recipe import checked_metadata


def stage(cache,output,dataset_id):
    root=Path(__file__).resolve().parents[1]; cache=Path(cache); output=Path(output)
    if output.exists():raise ValueError('Choose a fresh staging directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Commit source before staging')
    checked_metadata(cache)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    pairs=[(p,p.relative_to(root).as_posix()) for p in sorted((root/'src/overlap_audio').rglob('*.py'))]
    pairs += [(root/'scripts/run_selected_recipe_cloud.py','scripts/run_selected_recipe_cloud.py')]
    pairs += [(cache/name,'cache/'+name) for name in ('metadata.json','train.npz','validation.npz')]
    output.mkdir(parents=True); entries=[]
    for source,name in pairs:
        flat=name.replace('/','__');destination=output/flat
        shutil.copyfile(source,destination)
        entries.append({'path':name,'flat_name':flat,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    manifest={'source_commit':source_commit,'files':entries,'test_arrays_staged':False}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (output/'dataset-metadata.json').write_text(json.dumps({'title':'Private MAESTRO Selected Recipe Reproduction V1',
        'id':dataset_id,'licenses':[{'name':'other'}]},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'source_commit':source_commit,'files':len(entries),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset-id',required=True)
    stage(**vars(p.parse_args()))
