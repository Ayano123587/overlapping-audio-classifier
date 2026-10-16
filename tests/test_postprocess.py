import numpy as np
import pytest
import torch
import soundfile as sf
from overlap_audio.postprocess import convex_mix,temporal_filter,apply_policy,lookahead,infer,validate_bundle,prune_bundle
from overlap_audio.transfer import Head,RATE
from overlap_audio.natural import file_hash


def test_box_record_boundaries_gaps_and_edge_normalization():
    p=np.array([[0.],[0.],[1.],[1.]],np.float32)
    assert np.array_equal(temporal_filter(p,['a','a','b','b'],[0,1,0,1],{'kind':'box','width':9}),p)
    assert np.array_equal(temporal_filter(p,['a']*4,[0,1,4,5],{'kind':'box','width':9}),p)
    edge=temporal_filter(np.array([[0.],[1.],[0.]],np.float32),['a']*3,[0,1,2],{'kind':'box','width':3})
    assert edge[:,0]==pytest.approx([.5,1/3,.5])
    with pytest.raises(ValueError,match='strictly'):temporal_filter(p,['a']*4,[0,1,1,2],{'kind':'none'})


def test_ewma_is_prefix_causal_resets_and_never_mutates_input():
    p=np.array([[0.],[1.],[0.],[1.]],np.float32);original=p.copy()
    spec={'kind':'ewma','alpha':.5}
    full=temporal_filter(p,['a','a','a','b'],[0,1,2,0],spec)
    prefix=temporal_filter(p[:2],['a','a'],[0,1],spec)
    assert np.array_equal(full[:2],prefix)
    assert full[:,0]==pytest.approx([0,.5,.25,1])
    assert np.array_equal(p,original)
    for alpha in [0,float('nan'),2]:
        with pytest.raises(ValueError):temporal_filter(p,['a']*4,np.arange(4),{'kind':'ewma','alpha':alpha})


def test_convex_ensemble_guards_and_policy_latency():
    p=np.array([[[.1],[.3]],[[.9],[.7]]],np.float32)
    assert convex_mix(p,[.25,.75])[:,0]==pytest.approx([.7,.6])
    for weights in [[-1,2],[.2,.2],[float('nan'),1]]:
        with pytest.raises(ValueError):convex_mix(p,weights)
    policy={'branches':[{'weights':[1,0],'filters':[{'kind':'box','width':5}]},{'weights':[0,1],'filters':[]}],
            'mix_weights':[.25,.75],'filters':[{'kind':'box','width':3}]}
    assert lookahead(policy,[{'context_seconds':1},{'context_seconds':5}])==3
    result=apply_policy(p,policy,['a','b'],[0,0])
    assert result[:,0]==pytest.approx([.7,.6])


def test_selected_checkpoint_waveform_replay_with_fixture_encoder(tmp_path,monkeypatch):
    import overlap_audio.postprocess as module
    weights=tmp_path/'weights.pth';weights.write_bytes(b'original fixture')
    components=[]
    for context,value in [(1,.8),(5,.2)]:
        head=Head(2)
        with torch.no_grad():
            head.layers.weight.zero_();head.layers.bias.fill_(float(np.log(value/(1-value))))
        components.append({'state_dict':head.state_dict(),'config':{'hidden':0},'classes':['car','footsteps'],
                           'sample_rate':RATE,'context_seconds':context,'mean':torch.zeros(2048),'std':torch.ones(2048),
                           'encoder_weight_sha256':file_hash(weights)})
    policy={'branches':[{'weights':[1,0],'filters':[]},{'weights':[0,1],'filters':[]}],
            'mix_weights':[.5,.5],'filters':[{'kind':'box','width':3}]}
    bundle={'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'components':components,
            'classes':['car','footsteps'],'encoder_weight_sha256':file_hash(weights),'policy':policy,'thresholds':torch.full((2,),.5)}
    checkpoint=tmp_path/'selected.pt';torch.save(bundle,checkpoint)
    validate_bundle(bundle)
    import copy
    for mutate in [lambda x:x['policy'].update(mix_weights=[1]),lambda x:x['components'][0].update(classes=['bad']),
                   lambda x:x.update(thresholds=torch.tensor([float('nan'),.5])),lambda x:x['components'][0]['std'].__setitem__(0,0),
                   lambda x:x['policy']['branches'][0].update(weights=[1]),lambda x:x['components'][0]['state_dict'].pop('layers.bias')]:
        invalid=copy.deepcopy(bundle);mutate(invalid)
        with pytest.raises(ValueError):validate_bundle(invalid)
    only=copy.deepcopy(bundle);only['policy']['mix_weights']=[0,1]
    pruned=prune_bundle(only)
    assert len(pruned['components'])==1 and pruned['retained_component_indices']==[1]
    assert len(pruned['policy']['branches'])==1 and pruned['policy']['branches'][0]['weights']==[1]
    class FixtureEncoder(torch.nn.Module):
        def forward(self,wave):return {'embedding':torch.zeros((len(wave),2048))}
    monkeypatch.setattr(module,'encoder',lambda weights,device:FixtureEncoder())
    audio=tmp_path/'audio.wav';sf.write(audio,np.zeros(RATE*4,np.float32),RATE)
    result=infer(checkpoint,weights,audio,tmp_path/'result.json')
    assert result['lookahead_seconds']==3 and result['offline_future_context']
    assert len(result['observations'])==4
    for row in result['observations']:assert list(row['probabilities'].values())==pytest.approx([.5,.5],abs=1e-6)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):infer(checkpoint,weights,audio,tmp_path/'bad.json')


def test_inference_rejects_existing_and_input_alias_before_loading(tmp_path,monkeypatch):
    from overlap_audio import postprocess as module
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before checkpoint/encoder allocation')
    monkeypatch.setattr(module.torch,'load',forbidden)
    monkeypatch.setattr(module,'encoder',forbidden)
    paths=[tmp_path/name for name in ('checkpoint.pt','encoder.pt','audio.wav')]
    for path in paths:path.write_bytes(b'original')
    for output in paths:
        with pytest.raises(ValueError,match='aliases'):module.infer(*paths,output)
        assert output.read_bytes()==b'original'
    report=tmp_path/'report.json';report.write_text('preserved',encoding='utf-8')
    with pytest.raises(ValueError,match='exists'):module.infer(*paths,report)
    assert report.read_text()=='preserved'
