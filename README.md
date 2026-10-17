# Overlapping environmental sound classification

A PyTorch multilabel classifier for naturally overlapping MAESTRO recordings. The run uses all 49 recordings in the official MAESTRO Real v1 release, with no recording subsample; the [coverage audit](docs/dataset-coverage.json) verifies archive IDs and annotation hashes. A frozen PANNs Cnn14 encoder feeds a trained 17-class MLP. Training keeps unknown scene annotations masked; inference applies recording-safe temporal smoothing.

| Model | Validation AP | Observed test AP | Test micro F1 |
|---|---:|---:|---:|
| Five-second transfer head | 0.47009 | 0.39538 | 0.46061 |
| Selected head + box3 | 0.47670 | 0.40298 | 0.47153 |

All seconds in each split are retained. Each class's AP uses only its known annotations; the mean includes classes with known positive labels: 12 validation classes and 13 test classes. The [full-validation diagnostics](docs/selected-validation.json) include all 17 outputs. Missing supervision and classes without validation positives produce unreliable thresholds and false alarms. Inference needs three seconds of future context.

## Notebooks

- [Data and label audit](notebooks/01_results_eda.ipynb): recording counts, source checks, soft labels and class coverage.
- [Training and validation](notebooks/02_complete_natural_audio_workflow.ipynb): visible preprocessing, models, loss, scorer, training loop, actual full-data replay, learning curves and full-validation class errors.
- [Results and inference](notebooks/03_results_and_inference.ipynb): class errors, support, thresholds and actual saved-waveform proof.
- [Deployment diagnostics](notebooks/04_deployment_and_threshold_diagnostics.ipynb): actual saved-head replay, fixed-threshold mistakes, recording-boundary checks and measured cached inference timing.

The main notebook was executed end to end on Python 3.13.15 with the exact CPU packages in [requirements-notebook.txt](requirements-notebook.txt). It trains a full-data control, verifies fresh checkpoint reload on all 2,163 validation seconds, then replays the separate retained deployment head. Its first cell checks the runtime. The [execution record](docs/notebook-execution.json) identifies each notebook's artifacts and execution scope. The complete private feature cache and retained checkpoint are required. These analyses do not rescore the observed test.

To rerun the main notebook, create a Python 3.13.15 environment and install `python -m pip install -r requirements-notebook.txt`, then `python -m pip install -e . --no-deps`. The requirements use the official PyTorch CPU wheel index; SoundFile needs libsndfile 1.2.2. Open the notebook from the repository root after restoring the pinned private artifacts.

## Run

~~~bash
python -m pip install -e ".[dev,transfer,notebook]"
python -m overlap_audio.fetch_maestro --audio
python -m overlap_audio.selected_recipe weights --output data/external/panns/Cnn14.pth
python -m overlap_audio.transfer prepare --root data/external/maestro --output data/features/five-second-cache --weights data/external/panns/Cnn14.pth --device cpu --context-seconds 5
python train.py --cache data/features/five-second-cache --output models/my-fresh-run --recipe selected --device cpu
python evaluation_scripts/evaluate.py --checkpoint models/my-fresh-run/selected.pt --cache-five data/features/five-second-cache --output data/reports/my-validation.json
python -m pytest -q
~~~

Feature extraction needs substantial RAM; use cloud compute on a limited-memory host. Root training defaults to the fixed selected recipe: one seed-42, 256-unit head, train-only normalization and class weights, validation checkpoint stopping, fixed box3 smoothing, and validation-only thresholds. It requires the complete pinned 6,838/2,163-row training/validation cache and never opens test labels. Fresh fits reproduce the recipe; CPU and CUDA runs can produce different weights and scores. Use `--recipe experiment --config control` for the separate three-seed control. The weight acquisition command verifies the published PANNs SHA256 before feature extraction.

Run the fresh selected checkpoint on a recording:

~~~bash
python -m overlap_audio.postprocess predict --checkpoint models/my-fresh-run/selected.pt --weights data/external/panns/Cnn14.pth --audio data/my-recording.wav --output data/my-new-prediction.json
~~~

## Scope and sources

This is a custom 29/10/10 recording split, not the official DCASE five-fold benchmark. Test results are already observed; repeated validation selection, scene bias and unknown location/session overlap limit generalization.

[Report](Report.md) gives the training recipe and selected results. [Reproducibility record](docs/reproducibility.json) pins the sources, selection protocol and retained checkpoint. MAESTRO has a [custom experimental/noncommercial license](docs/MAESTRO-LICENSE.txt). PANNs code is MIT; pretrained weights are CC-BY 4.0. The optional ESC-10 adapter retains [clip attribution](docs/ESC10-ATTRIBUTION.md). No audio, caches or model binaries are published.
