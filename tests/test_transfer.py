import json
import numpy as np
import pytest
import torch
import soundfile as sf
from overlap_audio.transfer import Head, masked_loss, timeline, windows, RATE, load, train, infer
from overlap_audio.natural import file_hash


def test_center_window_and_soft_mask_alignment():
    wave=np.arange(3*RATE,dtype=np.float32)
    x=windows(wave,[0,1,2])
    assert x.shape==(3,5*RATE)
    assert torch.equal(x[0,2*RATE:],torch.from_numpy(wave))
    assert torch.count_nonzero(x[0,:2*RATE])==0
    central=windows(wave,[0,1,2],context_seconds=1)
    assert central.shape==(3,RATE)
    assert torch.equal(central.flatten(),torch.from_numpy(wave))
    with pytest.raises(ValueError,match='context'):windows(wave,[0],context_seconds=2)
    record={'rows':[(.5,1.2,'car',.5),(1,2,'car',.8)]}
    y,m=timeline(record,{'car'},['car','unknown'],3)
    assert y[:,0].tolist()==pytest.approx([.5,.8,0])
    assert m[:,1].sum()==0
    assert (y>.5).sum()==1
    with pytest.raises(ValueError):windows(np.array([np.nan],np.float32),[0])


def test_masked_gradient_and_train_only_weights():
    logits=torch.zeros((2,2),requires_grad=True)
    target=torch.tensor([[.8,1.],[.2,0.]])
    mask=torch.tensor([[1.,0.],[1.,0.]])
    loss=masked_loss(logits,target,mask,torch.tensor([2.,20.]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:,1],torch.zeros(2))
    assert logits.grad[0,0]<0


def fixture(cache):
    cache.mkdir();contracts={};assignment={}
    rng=np.random.default_rng(42)
    for split in ['train','validation','test']:
        x=rng.normal(size=(8,2048)).astype(np.float32)
        y=np.array([[0,1],[1,0]]*4,np.float32);m=np.ones_like(y)
        ids=np.full(8,split+'_record')
        np.savez(cache/(split+'.npz'),features=x,targets=y,masks=m,record_ids=ids)
        contracts[split]={'sha256':file_hash(cache/(split+'.npz')),'rows':8,'record_ids':[split+'_record']}
        assignment[split+'_record']=split
    metadata={'classes':['car','footsteps'],'contracts':contracts,'assignment':assignment,
              'encoder_weight_sha256':'fixture','holdout_status':'original fixture only','context_seconds':1}
    (cache/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    return metadata


def test_actual_head_train_checkpoint_and_fresh_guard(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    metadata=fixture(tmp_path/'cache')
    import overlap_audio.transfer as module
    original=module.load
    def guarded(cache,split,metadata):
        if split=='test':
            assert (tmp_path/'run'/'selection.json').exists()
            assert (tmp_path/'run'/'selected.pt').exists()
        return original(cache,split,metadata)
    monkeypatch.setattr(module,'load',guarded)
    report=train(tmp_path/'cache',tmp_path/'run',epochs=1,device='cpu',lr_scale=1/3)
    assert report['checkpoint_probability_max_difference']==0
    assert report['selected']['lr']==pytest.approx(.001/3 if report['selected']['hidden']==0 else .0003/3)
    assert len(report['test_validation_thresholds']['per_class'])==2
    state=torch.load(tmp_path/'run'/'selected.pt',weights_only=True)
    assert state['mean'].shape==(2048,)
    assert state['context_seconds']==1
    assert all((tmp_path/'run'/(name+'.pt')).exists() for name in ['linear-soft','linear-balanced','mlp-balanced'])
    # Test saved-head waveform plumbing with an explicitly fake encoder; real
    # pretrained encoder/checkpoint inference is separately executed in cloud.
    weights=tmp_path/'fake-encoder.pth';weights.write_bytes(b'original fixture')
    state['encoder_weight_sha256']=file_hash(weights)
    checkpoint=tmp_path/'fixture.pt';torch.save(state,checkpoint)
    class FakeEncoder(torch.nn.Module):
        def forward(self,batch):
            assert batch.shape[1]==RATE
            return {'embedding':torch.zeros((len(batch),2048))}
    monkeypatch.setattr(module,'encoder',lambda weights,device:FakeEncoder())
    audio=tmp_path/'fixture.wav';sf.write(audio,np.zeros(RATE*2,np.float32),RATE)
    replay=infer(checkpoint,weights,audio,tmp_path/'inference.json')
    assert len(replay['observations'])==2 and not replay['offline_future_context']
    head=Head(2,state['config']['hidden']);head.load_state_dict(state['state_dict']);head.eval()
    expected=torch.sigmoid(head((-state['mean']/state['std'])[None])).detach()[0]
    assert list(replay['observations'][0]['probabilities'].values())==pytest.approx(expected.tolist())
    state['std'][0]=0;torch.save(state,checkpoint)
    with pytest.raises(ValueError,match='scaling'):infer(checkpoint,weights,audio,tmp_path/'bad.json')
    with pytest.raises(ValueError,match='fresh'):train(tmp_path/'cache',tmp_path/'run',epochs=1,device='cpu')
    with (tmp_path/'cache'/'test.npz').open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load(tmp_path/'cache','test',metadata)
    for invalid in [0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='positive'):train(tmp_path/'cache',tmp_path/'bad',lr_scale=invalid)


def test_unknown_target_values_do_not_change_head_gradient():
    torch.manual_seed(42);head=Head(2)
    x=torch.randn(3,2048);m=torch.tensor([[1.,0.]]*3)
    y=torch.zeros(3,2)
    a=masked_loss(head(x),y,m,torch.ones(2))
    y[:,1]=1
    b=masked_loss(head(x),y,m,torch.ones(2))
    assert torch.equal(a,b)
