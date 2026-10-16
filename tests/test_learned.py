import numpy as np
import torch
from overlap_audio.learned import fit_head,flatten_mixture
from overlap_audio.postprocess import apply_policy,prune_bundle,validate_bundle,cached_selected,component_predictions
from overlap_audio.transfer import Head,RATE


def test_learned_fit_reproducible_train_only_moments_and_saved_state(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(9)
    x=torch.randn(8,2048);y=torch.tensor([[0.,1.],[1.,0.]]*4);mask=torch.ones_like(y)
    vx=x+100
    config={'name':'original-fixture','hidden':256,'lr':.001,'dropout':.5,'weight_decay':.1}
    a,history=fit_head(x,y,mask,vx,y,mask,['car','footsteps'],config,42,epochs=2,device='cpu')
    b,other=fit_head(x,y,mask,vx,y,mask,['car','footsteps'],config,42,epochs=2,device='cpu')
    assert torch.equal(a['mean'],x.mean(0)) and not torch.equal(a['mean'],vx.mean(0))
    assert all(torch.equal(a['state_dict'][key],b['state_dict'][key]) for key in a['state_dict'])
    assert history==other and all(np.isfinite(row['loss']) for row in history['history'])
    assert history['validation_ap']==max(row['validation_ap'] for row in history['history'])
    path=tmp_path/'state.pt';torch.save(a,path)
    restored=torch.load(path,weights_only=True)
    assert np.array_equal(component_predictions(a,x.numpy()),component_predictions(restored,x.numpy()))


def test_flattened_pipeline_is_linear_boundary_safe_and_pruned():
    def component(context):
        model=Head(2)
        return {'state_dict':model.state_dict(),'config':{'hidden':0},'classes':['car','footsteps'],
                'sample_rate':RATE,'context_seconds':context,'mean':torch.zeros(2048),'std':torch.ones(2048),
                'encoder_weight_sha256':'a'*64}
    old={'kind':'panns_probability_ensemble_v1','sample_rate':RATE,'classes':['car','footsteps'],
         'encoder_weight_sha256':'a'*64,'components':[component(1),component(5)],'thresholds':torch.ones(2)*.5,
         'policy':{'branches':[{'weights':[1,0],'filters':[{'kind':'box','width':3}]},
                               {'weights':[0,1],'filters':[{'kind':'ewma','alpha':.5}]}],
                   'mix_weights':[.25,.75],'filters':[{'kind':'box','width':3}]}}
    new=[component(5) for _ in range(3)]
    rng=np.random.default_rng(42);p=rng.uniform(size=(5,8,2)).astype(np.float32)
    ids=np.array(['a']*4+['b']*4);seconds=np.array([0,1,4,5]*2)
    old_p=apply_policy(p[:2],old['policy'],ids,seconds)
    flat=flatten_mixture(old,new,{'kind':'box','width':5},0)
    validate_bundle(flat)
    assert np.max(np.abs(apply_policy(p,flat['policy'],ids,seconds)-old_p))<1e-6
    chosen=prune_bundle(flat)
    assert len(chosen['components'])==2
    assert np.array_equal(cached_selected(p,chosen,ids,seconds),apply_policy(p,flat['policy'],ids,seconds))
