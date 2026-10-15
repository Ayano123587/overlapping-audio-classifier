import json
import numpy as np
import pytest
import soundfile as sf
import torch
from overlap_audio.natural import annotations,frozen_split,logmel,prepare,scores,select_thresholds,train,infer,validate_cache_bundle,file_hash,load_cache


def original_fixture(root):
    annotation=root/'development_annotation'/'soft_labels_original_scene'
    audio=root/'development_audio'/'original_scene'
    annotation.mkdir(parents=True);audio.mkdir(parents=True)
    for index in range(5):
        # Original short tones validate plumbing only, never real-data performance.
        wave=np.sin(np.arange(32000)*2*np.pi*(200+index*20)/16000).astype(np.float32)*.1
        sf.write(audio/f'original_scene_{index:02d}.wav',wave,16000)
        (annotation/f'original_scene_{index:02d}.txt').write_text(
            '0\t1\tcar\t0.8\n0\t1\tfootsteps\t0.2\n1\t2\tcar\t0.1\n1\t2\tfootsteps\t0.7\n',encoding='utf-8')


def test_natural_recording_group_split_and_alignment(tmp_path):
    torch.set_num_threads(2)
    original_fixture(tmp_path/'raw')
    records=annotations(tmp_path/'raw')
    assignment=frozen_split(records)
    assert assignment==frozen_split(records)
    assert len(assignment)==5
    assert set(assignment.values())=={'train','validation','test'}
    metadata=prepare(tmp_path/'raw',tmp_path/'cache')
    assert metadata['classes']==['car','footsteps']
    ids={}
    for split in ['train','validation','test']:
        with np.load(tmp_path/'cache'/f'{split}.npz',allow_pickle=False) as cache:
            assert cache['features'].shape[1:]==(1,64,251)
            assert np.isfinite(cache['features']).all()
            assert cache['masks'].min()==1
            assert cache['targets'][0].tolist()==pytest.approx([.8,.2])
            ids[split]=set(cache['record_ids'])
    assert not(ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test'])


def test_masked_classes_are_not_false_negatives():
    target=torch.tensor([[1.,0.],[0.,1.]])
    probability=torch.tensor([[.9,.9],[.9,.9]])
    mask=torch.tensor([[1.,0.],[0.,1.]])
    result=scores(target,probability,mask,torch.tensor([.5,.5]),['car','footsteps'])
    assert result['micro_f1']==1
    assert result['macro_f1_supported']==1
    thresholds=select_thresholds(target,probability,mask)
    assert thresholds.shape==(2,)


def test_natural_feature_and_annotation_invalid_input(tmp_path):
    assert logmel(torch.zeros(2,80000)).shape==(2,1,64,251)
    original_fixture(tmp_path)
    path=next((tmp_path/'development_annotation').rglob('*.txt'))
    path.write_text('0\t1\tcar\t1.1\n')
    with pytest.raises(ValueError):annotations(tmp_path)
    path.write_text('0\t1\tcar\t1.000000031316843\n')
    records=annotations(tmp_path)
    assert sum(r['roundoff_clamps'] for r in records)==1
    corrected=next(r for r in records if r['id']==path.stem)
    assert corrected['rows'][0][3]==1.
    path.write_text('nan\t1\tcar\t0.5\n')
    with pytest.raises(ValueError):annotations(tmp_path)


def test_natural_training_selection_and_single_release(tmp_path):
    torch.set_num_threads(2)
    original_fixture(tmp_path/'raw')
    prepare(tmp_path/'raw',tmp_path/'cache')
    result=train(tmp_path/'cache',tmp_path/'model',epochs=1,device='cpu')
    assert len(result['validation_candidates'])==3
    assert result['test_fixed_point_five']['per_class'][0]['observed']==2
    checkpoint=torch.load(tmp_path/'model'/'selected.pt',weights_only=True)
    assert checkpoint['classes']==['car','footsteps']
    assert torch.isfinite(checkpoint['thresholds']).all()
    audio=next((tmp_path/'raw'/'development_audio').rglob('*.wav'))
    prediction=infer(tmp_path/'model'/'selected.pt',audio,tmp_path/'prediction.json')
    assert len(prediction['observations'])==2
    assert set(prediction['observations'][0]['classes'])=={'car','footsteps'}
    with pytest.raises(ValueError,match='already released'):
        train(tmp_path/'cache',tmp_path/'model',epochs=1)


def test_cache_checksums_and_recording_membership(tmp_path):
    original_fixture(tmp_path/'raw')
    metadata=prepare(tmp_path/'raw',tmp_path/'cache')
    validate_cache_bundle(tmp_path/'cache',metadata)
    path=tmp_path/'cache'/'train.npz'
    with np.load(path,allow_pickle=False) as cache:
        arrays={key:cache[key] for key in cache.files}
    arrays['record_ids']=np.full(len(arrays['record_ids']),'wrong-source')
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='checksum'):
        validate_cache_bundle(tmp_path/'cache',metadata)
    metadata['cache_contracts']['train']['sha256']=file_hash(path)
    with pytest.raises(ValueError,match='assignment'):
        validate_cache_bundle(tmp_path/'cache',metadata)


def test_cache_rejects_misaligned_rows(tmp_path):
    path=tmp_path/'misaligned.npz'
    np.savez_compressed(path,features=np.zeros((3,1,64,251)),targets=np.zeros((2,2)),
                        masks=np.ones((2,2)),record_ids=np.asarray(['a','a']))
    with pytest.raises(ValueError,match='misaligned'):
        load_cache(path)


def test_unsupported_scores_are_json_null_not_nan():
    zero=torch.zeros(3,2)
    result=scores(zero,zero,torch.ones_like(zero),torch.tensor([.5,.5]),['car','footsteps'])
    assert result['macro_ap_supported'] is None
    assert result['macro_f1_all_observed_classes']==0
    assert 'NaN' not in json.dumps(result,allow_nan=False)
    unobserved=scores(zero,zero,zero,torch.tensor([.5,.5]),['car','footsteps'])
    assert unobserved['macro_f1_all_observed_classes'] is None
    assert unobserved['micro_f1'] is None
    json.dumps(unobserved,allow_nan=False)
    with pytest.raises(ValueError,match='finite'):
        scores(zero,torch.full_like(zero,float('nan')),zero,torch.tensor([.5,.5]),['car','footsteps'])
