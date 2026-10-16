import json
import numpy as np
import pytest
import torch
from overlap_audio import selected_recipe as recipe
from overlap_audio.transfer import Head


def metadata():
    groups = {'train':[f't{i}' for i in range(29)],'validation':[f'v{i}' for i in range(10)],'test':[f'z{i}' for i in range(10)]}
    return {'encoder_weight_sha256':recipe.ENCODER_SHA256,'context_seconds':5,'sample_rate':32000,
            'classes':['car','footsteps'],'sources':[{}]*49,'assignment':{},'split':'fixture',
            'contracts':{s:{'rows':n,'record_ids':groups[s]} for s,n in [('train',6838),('validation',2163),('test',2368)]}}


def test_complete_receipt_and_source_separation(tmp_path,monkeypatch):
    m=metadata(); path=tmp_path/'metadata.json'
    monkeypatch.setattr(recipe,'COHORT_SHA256',recipe.cohort_fingerprint(m))
    path.write_text(json.dumps(m),encoding='utf-8'); assert recipe.checked_metadata(tmp_path)==m
    m['contracts']['train']['rows']=10;path.write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(ValueError,match='Incomplete'):recipe.checked_metadata(tmp_path)
    m=metadata();m['contracts']['validation']['record_ids'][0]='t0';path.write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(ValueError,match='separation'):recipe.checked_metadata(tmp_path)
    m=metadata();m['sources'][0]={'audio_sha256':'wrong'};path.write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(ValueError,match='receipt'):recipe.checked_metadata(tmp_path)


def test_fixed_recipe_never_opens_test_and_roundtrips(tmp_path,monkeypatch):
    m=metadata();calls=[]
    monkeypatch.setattr(recipe,'checked_metadata',lambda p:m)
    monkeypatch.setattr(recipe,'file_hash',lambda p:'a'*64)
    def aligned(root,split,meta):
        assert split in ('train','validation');calls.append(split)
        n=6838 if split=='train' else 2163
        y=np.tile(np.array([[0.,1.],[1.,0.]],dtype=np.float32),(n//2+1,1))[:n]
        return np.zeros((n,2048),dtype=np.float32),y,np.ones_like(y),np.array(['r']*n),np.arange(n)
    monkeypatch.setattr(recipe,'aligned_cache',aligned)
    def fit(*args):
        assert args[7]==recipe.CONFIG and args[8:]==(42,80,'cpu')
        return {'state_dict':Head(2,256).state_dict(),'classes':m['classes'],'config':dict(recipe.CONFIG),
                'mean':torch.zeros(2048),'std':torch.ones(2048),'sample_rate':32000,'context_seconds':5,'seed':42}, {'best_epoch':1}
    monkeypatch.setattr(recipe,'fit_head',fit)
    torch.set_num_threads(2)
    report=recipe.train_selected(tmp_path,tmp_path/'run')
    bundle=torch.load(tmp_path/'run'/'selected.pt',weights_only=True)
    assert calls==['train','validation'] and report['test_labels_loaded'] is False
    assert len(bundle['components'])==1 and bundle['policy']['filters']==[{'kind':'box','width':3}]
    assert report['checkpoint_probability_max_difference']==0 and bundle['lookahead_seconds']==3
    with pytest.raises(ValueError,match='Output exists'):recipe.train_selected(tmp_path,tmp_path/'run')


def test_encoder_acquisition_fails_closed(tmp_path,monkeypatch):
    destination=tmp_path/'encoder.pt';destination.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Existing'):recipe.acquire_weights(destination)
    destination.unlink()
    monkeypatch.setattr(recipe.urllib.request,'urlretrieve',lambda url,path:path.write_bytes(b'wrong'))
    with pytest.raises(ValueError,match='Downloaded'):recipe.acquire_weights(destination)
    assert destination.with_name(destination.name+'.part').exists() and not destination.exists()


def test_semantic_receipt_allows_fresh_archive_hashes_but_not_source_changes(tmp_path,monkeypatch):
    m=metadata(); expected=recipe.cohort_fingerprint(m)
    monkeypatch.setattr(recipe,'COHORT_SHA256',expected)
    for split in m['contracts']:m['contracts'][split]['sha256']='fresh-container-hash'
    m['holdout_status']='fresh preparation, no labels opened'
    (tmp_path/'metadata.json').write_text(json.dumps(m),encoding='utf-8')
    assert recipe.checked_metadata(tmp_path)==m
    assert recipe.cohort_fingerprint(m)==expected
    m['classes'].reverse()
    assert recipe.cohort_fingerprint(m)!=expected
