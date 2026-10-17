# Environmental sound classification

The selected five-second PANNs head with recording-safe box3 smoothing achieves **0.47670 validation AP** and **0.40298 observed-test AP**. The unsmoothed head scores 0.47009 and 0.39538. Test micro F1 is **0.47153**, all-17-class macro F1 **0.30153**, and supported-class macro F1 **0.39431**.

## Data and model

The run uses every recording in [MAESTRO Real v1](https://zenodo.org/records/7244360): 49 recordings and 17 classes. The recording split gives 6,838 training, 2,163 validation and 2,368 test whole seconds. The [coverage audit](docs/dataset-coverage.json) checks all annotation IDs and hashes. Unknown scene classes are masked; soft annotation scores train the head and strict >0.5 labels evaluate it.

Frozen [PANNs Cnn14 weights](https://zenodo.org/records/3987831) produce native 32 kHz, five-second features. A trained 256-unit MLP uses dropout 0.3, train-only scaling and prevalence weights, masked soft BCE, AdamW LR 0.0003 and validation checkpoint selection. Validation F1 sets class thresholds after fixed box3 filtering, which resets at recording boundaries and missing seconds. `train.py --recipe selected` reproduces the one-head seed-42 recipe with an 80-epoch ceiling, minimum 20 epochs and patience 12; the earliest maximum unsmoothed validation AP determines the checkpoint. It checks the complete pinned cohort, never loads test labels, and saves a reload-verified deployment bundle. A full-cohort CPU fit and exact saved-checkpoint replay verify this command; CPU/CUDA training need not reproduce identical weights or scores. The retained validation-best CUDA model supplies the reported results. The [selected-recipe training record](docs/selected-training.json) contains the actual full-cohort CPU fit history and saved-checkpoint proof.

| Model | Validation AP | Observed test AP | Test micro F1 |
|---|---:|---:|---:|
| Scratch spectrogram CNN | 0.41552 | 0.37617 | not recorded |
| Five-second transfer head | 0.47009 | 0.39538 | 0.46061 |
| Selected head + box3 | **0.47670** | **0.40298** | **0.47153** |

## Working inference

The retained checkpoint is `models/postprocess-v5/selected.pt`, SHA256 `d082cb659e8ab94b9661fca741630cf12e09b951aa7ec74a544deceb7792c689`. Real-waveform inference produced 85 finite probabilities for five seconds, with exact cached replay. The [reproducibility record](docs/reproducibility.json) pins encoder, feature metadata and selection sources.

The [full-validation audit](docs/selected-validation.json) binds all 2,163 validation rows to the retained checkpoint and independently checked metrics, with a 2e-7 float32 arithmetic tolerance. Classwise AP, frozen thresholds and false-positive/false-negative counts are plotted in the main notebook. No validation or test rows are omitted.

The [training notebook](notebooks/02_complete_natural_audio_workflow.ipynb) was executed in one CPU session with all 6,838 training and 2,163 validation seconds. It trains a fixed control, saves and reloads its checkpoint with zero probability difference, then replays the retained head at 0.4767020191 validation AP. The fresh control is distinct from the retained CUDA deployment model. The [deployment notebook](notebooks/04_deployment_and_threshold_diagnostics.ipynb) loads the selected artifact on all 2,163 validation seconds, shows masked classwise mistakes and verifies reload and recording-boundary behavior. Cached-head timing excludes audio loading and encoder computation.

## Limits

This is a custom 29/10/10 recording split, not the official DCASE five-fold benchmark. The test was previously observed and development selection was repeated. Inference requires three seconds of future context. Unknown location/session overlap and upstream pretrained exposure limit transfer claims.

Children voices, coffee machine and shopping cart lose AP relative to unsmoothed inference. Coffee/cart each have only one positive test second; four test classes have no positives and several classes lack training supervision. Five validation classes have no positives; their frozen thresholds produce substantial false alarms. Those thresholds have no evidence of useful calibration. The result does not establish reliable rare-class detection or statistical significance. Audio, embeddings and model weights are not redistributed; MAESTRO's custom experimental/noncommercial terms apply.
