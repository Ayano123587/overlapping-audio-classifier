import json
import torch
import pytest
from sklearn.metrics import average_precision_score
from overlap_audio.data import synthetic
from overlap_audio.experiment import LazyMixtures, StrongCNN, fit_candidate
from overlap_audio.audio import spectrogram
from overlap_audio.metrics import average_precision


def test_ap_ties_match_independent_scorer():
    targets=torch.tensor([1.,0.,1.,0.,1.,0.])
    scores=torch.tensor([.7,.7,.2,.1,.2,.1])
    assert average_precision(targets,scores)==pytest.approx(average_precision_score(targets.numpy(),scores.numpy()))


def test_lazy_mixtures_are_deterministic_and_keep_full_clip(tmp_path):
    manifest=synthetic(tmp_path)
    dataset=LazyMixtures(manifest,'train',8,42)
    first,target=dataset[3]
    again,second_target=dataset[3]
    assert first.shape==(40000,)
    torch.testing.assert_close(first,again)
    torch.testing.assert_close(target,second_target)
    assert target.sum()>=1
    with pytest.raises(ValueError):
        LazyMixtures(manifest,'test',8,42,excluded={r['source'] for r in manifest['records'] if r['split']=='test'})


def test_strong_model_finite_gradients_and_checkpoint_selection(tmp_path):
    torch.set_num_threads(2)
    manifest=synthetic(tmp_path/'sources')
    train=LazyMixtures(manifest,'train',32,42,rare_probability=.3)
    validation=LazyMixtures(manifest,'validation',32,43,rare_probability=.3)
    folder=tmp_path/'run';folder.mkdir()
    config={'width':4,'gamma':2.,'lr':.001,'epochs':1}
    model,thresholds,result=fit_candidate(train,validation,config,42,'cpu',folder)
    assert len(result['history'])==1
    assert torch.isfinite(thresholds).all()
    assert (folder/'best.pt').exists()
    assert json.loads((folder/'validation.json').read_text())['seed']==42
    model.eval()
    output=model(spectrogram(train[0][0]))
    assert output.shape==(1,4)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
