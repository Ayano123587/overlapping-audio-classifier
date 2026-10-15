import json
import numpy as np
import pytest
import torch
torch.set_num_threads(2)
from overlap_audio.audio import load_wave,write_wave,spectrogram,augment_spec,augment_wave
from overlap_audio.model import SpectrogramCNN,FocalLoss
from overlap_audio.data import validate_sources,synthetic,mixtures
from overlap_audio.metrics import average_precision,evaluate,thresholds_from_validation
from overlap_audio.train import train,load_checkpoint

def test_focal_gradient_and_extreme_logits():
    logits=torch.tensor([[100.,-100.,0.,-2.]],requires_grad=True)
    target=torch.tensor([[0.,1.,1.,0.]])
    loss=FocalLoss([.5]*4)(logits,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0,0]>0 and logits.grad[0,1]<0

def test_focal_gamma_zero_matches_weighted_bce():
    logits=torch.tensor([[1.,-1.]])
    target=torch.tensor([[1.,0.]])
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)*.5
    assert FocalLoss([.5,.5],gamma=0)(logits,target)==pytest.approx(expected)

def test_source_leakage_rejected():
    with pytest.raises(ValueError,match="crosses"):
        validate_sources([{"source":"original","split":"train"},{"source":"original","split":"test"}])

def test_wave_resampling_and_padding(tmp_path):
    path=tmp_path/"sample.wav"
    write_wave(path,np.sin(2*np.pi*500*np.arange(4000)/16000)*.2,16000)
    waveform=load_wave(path)
    assert waveform.shape==(8000,) and torch.isfinite(waveform).all()
    assert torch.count_nonzero(waveform[2000:])==0
    assert spectrogram(waveform).shape==(1,1,129,63)

def test_augmentations_deterministic():
    wave=torch.ones(8000)
    a=augment_wave(wave,torch.Generator().manual_seed(1));b=augment_wave(wave,torch.Generator().manual_seed(1))
    assert torch.equal(a,b) and not torch.equal(a,wave)
    spec=torch.ones(1,1,129,63);original=spec.clone()
    augmented=augment_spec(spec,torch.Generator().manual_seed(2))
    assert torch.equal(spec,original) and augmented.shape==spec.shape

def test_model_shape():
    assert SpectrogramCNN(4)(torch.zeros(2,1,129,63)).shape==(2,4)

def test_metrics_ties_and_missing_class():
    assert average_precision(torch.tensor([1.,0.]),torch.tensor([.5,.5]))==.5
    assert average_precision(torch.zeros(2),torch.ones(2)) is None
    target=torch.tensor([[1.,0.],[0.,1.]])
    result=evaluate(target,target,torch.tensor([.5,.5]))
    assert result["macro_f1"]==1 and result["macro_average_precision"]==1
    assert thresholds_from_validation(torch.zeros(2,1),torch.ones(2,1)).tolist()==[.5]

def test_mixtures_source_disjoint_and_deterministic(tmp_path):
    manifest=synthetic(tmp_path)
    a,y,provenance=mixtures(manifest,"train",32)
    b,z,_=mixtures(manifest,"train",32)
    assert torch.equal(a,b) and torch.equal(y,z)
    assert a.shape==(32,8000) and y.shape==(32,4)
    assert all(source.startswith("train-") for row in provenance for source in row)
    assert a.abs().max()<=.901 and bool((y.sum(1)>=1).all())

def test_training_checkpoint_and_report_roundtrip(tmp_path):
    synthetic(tmp_path/"data")
    report=train(tmp_path/"data/manifest.json",tmp_path/"model.pt",tmp_path/"results.json",epochs=1,count=128)
    model,metadata=load_checkpoint(tmp_path/"model.pt")
    assert metadata["classes"]==report["classes"]
    assert model(torch.zeros(1,1,129,63)).shape==(1,4)
    assert json.loads((tmp_path/"results.json").read_text())["source_disjoint"]


def test_rare_probability_validation(tmp_path):
    manifest=synthetic(tmp_path)
    with pytest.raises(ValueError,match="rare_probability"):
        mixtures(manifest,"train",32,rare_probability=0)
    _,target,_=mixtures(manifest,"train",1024,rare_probability=.001)
    assert target[:,3].sum()<target[:,0].sum()/100
