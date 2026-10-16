# Overlapping environmental sound classification

A PyTorch multilabel classifier for naturally overlapping MAESTRO recordings. A frozen PANNs Cnn14 encoder feeds a trained 17-class MLP. Training keeps unknown scene annotations masked; inference applies recording-safe temporal smoothing.

| Model | Validation AP | Observed test AP | Test micro F1 |
|---|---:|---:|---:|
| Five-second transfer head | 0.47009 | 0.39538 | 0.46061 |
| Selected head + box3 | 0.47670 | 0.40298 | 0.47153 |

The improvement is small. Children voices, coffee machine and shopping cart lost AP. Several classes have absent or extremely thin supervision. Inference needs three seconds of future context.

## Notebooks

- [Data and label audit](notebooks/01_results_eda.ipynb): recording counts, source checks, soft labels and class coverage.
- [Model and experiments](notebooks/02_complete_natural_audio_workflow.ipynb): visible preprocessing, models, loss, scorer, training loop, actual full-data replay and learning curves.
- [Results and inference](notebooks/03_results_and_inference.ipynb): class errors, support, thresholds and actual saved-waveform proof.

The notebooks retain executed tables and plots. They never rescore the observed test. Data and weights stay private.

## Run

~~~bash
python -m pip install -e ".[dev,transfer,notebook]"
python -m overlap_audio.fetch_maestro --audio
python -m overlap_audio.transfer prepare --root data/external/maestro --output data/features/five-second-cache --weights data/external/panns/Cnn14.pth --device cpu --context-seconds 5
python train.py --cache data/features/five-second-cache --output models/my-fresh-run --config control --device cpu
python evaluation_scripts/evaluate.py --checkpoint models/my-fresh-run/selected.pt --cache-five data/features/five-second-cache --output data/reports/my-validation.json
python -m pytest -q
~~~

Feature extraction needs substantial RAM; use cloud compute on a limited-memory host. Root training fits a fresh three-seed configuration using train/validation only and rejects existing outputs. It does not reproduce the historical winner merely by using its name.

For the privately retained production artifact:

~~~bash
python -m overlap_audio.postprocess predict --checkpoint models/postprocess-v5/selected.pt --weights data/external/panns/Cnn14.pth --audio data/my-recording.wav --output data/my-new-prediction.json
~~~

## Scope and sources

This is a custom 29/10/10 recording split, not the official DCASE five-fold benchmark. Test results are already observed; repeated validation selection, scene bias and unknown location/session overlap limit generalization.

[Report](Report.md) records the recipe and negative experiments. [Experiment ledger](docs/experiment-ledger.json) retains hashes, failed runs and test-exposure timing. MAESTRO has a [custom experimental/noncommercial license](docs/MAESTRO-LICENSE.txt). PANNs code is MIT; pretrained weights are CC-BY 4.0. The optional ESC-10 adapter retains [clip attribution](docs/ESC10-ATTRIBUTION.md). No audio, caches or model binaries are published.
