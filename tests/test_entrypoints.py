import json
import numpy as np
import pytest
import torch
from overlap_audio import entrypoints


def test_train_only_never_requests_test_and_does_not_overwrite(tmp_path, monkeypatch):
    cache=tmp_path/'cache';cache.mkdir()
    metadata={'context_seconds':5,'classes':['car','footsteps'],'encoder_weight_sha256':'a'*64}
    (cache/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    for split in ('train','validation'):(cache/(split+'.npz')).write_bytes(b'fixture-hash')
    calls=[]
    def aligned(root,split,meta):
        assert split in ('train','validation');calls.append(split)
        x=np.random.default_rng(42).normal(size=(8,2048)).astype('float32')
        y=np.array([[0.,1.],[1.,0.]]*4,dtype='float32')
        return x,y,np.ones_like(y),np.array(['record']*8),np.arange(8)
    monkeypatch.setattr(entrypoints,'aligned_cache',aligned)
    torch.set_num_threads(2)
    result=entrypoints.train_only(cache,tmp_path/'run',seeds=[42],epochs=1)
    assert calls==['train','validation'] and not result['test_labels_loaded']
    with pytest.raises(ValueError,match='Output exists'):
        entrypoints.train_only(cache,tmp_path/'run',seeds=[42],epochs=1)
    with pytest.raises(ValueError,match='acknowledgement'):
        entrypoints.evaluate('missing.pt',{},tmp_path/'out.json',split='test')


def test_cached_evaluation_contract_and_no_overwrite(tmp_path,monkeypatch):
    from overlap_audio.transfer import Head,RATE
    c={'state_dict':Head(2).state_dict(),'config':{'hidden':0},'classes':['car','footsteps'],
       'sample_rate':RATE,'context_seconds':5,'mean':torch.zeros(2048),'std':torch.ones(2048),'encoder_weight_sha256':'a'*64}
    bundle={'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'classes':c['classes'],
            'encoder_weight_sha256':'a'*64,'components':[c],'thresholds':torch.full((2,),.5),
            'policy':{'branches':[{'weights':[1.],'filters':[]}],'mix_weights':[1.],'filters':[]}}
    checkpoint=tmp_path/'selected.pt';torch.save(bundle,checkpoint)
    metadata={'context_seconds':5,'classes':c['classes'],'encoder_weight_sha256':'b'*64}
    (tmp_path/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(ValueError,match='contract mismatch'):
        entrypoints.evaluate(checkpoint,{5:tmp_path},tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()
    metadata['encoder_weight_sha256']='a'*64
    (tmp_path/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    (tmp_path/'validation.npz').write_bytes(b'hash-fixture')
    y=np.array([[0.,1.],[1.,0.]],dtype='float32')
    monkeypatch.setattr(entrypoints,'aligned_cache',lambda *args:(np.zeros((2,2048),dtype='float32'),y,np.ones_like(y),np.array(['record']*2),np.arange(2)))
    report=entrypoints.evaluate(checkpoint,{5:tmp_path},tmp_path/'good.json')
    assert report['split']=='validation' and report['rows']==2
    assert report['metrics']['macro_ap_supported'] is not None
    (tmp_path/'result.json').write_text('historical',encoding='utf-8')
    with pytest.raises(ValueError,match='Output exists'):
        entrypoints.evaluate(checkpoint,{5:tmp_path},tmp_path/'result.json')
