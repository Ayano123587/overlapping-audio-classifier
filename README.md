# Overlapping Audio Classifier



A PyTorch pipeline for multilabel sound presence classification in noisy, overlapping audio. It includes a spectrogram CNN, class-aware focal loss, rare-positive sampling, waveform augmentation, spectrogram masking, validation-selected thresholds and original-recording-disjoint evaluation.



This is **clip classification**, not waveform source separation or event localization. The executed demos include synthetic tones and mixtures of a real ESC-10 subset; neither establishes field-recording or extreme 1:1000 performance.



## Install and reproduce



```bash

python -m pip install -e ".[dev]"

overlap-audio synthetic

overlap-audio train data/raw/synthetic/manifest.json --epochs 12 --count 512 --checkpoint models/synthetic.pt --report docs/synthetic-results.json

overlap-audio fetch-esc --per-fold 2

overlap-audio train data/external/esc10/manifest.json --epochs 15 --count 512 --checkpoint models/esc10.pt --report docs/esc10-results.json

overlap-audio predict models/esc10.pt data/external/esc10/audio/1-100032-A-0.wav

python -m pytest -q

```



The fetch command downloads **40 actual public WAV files**, official metadata and attribution from a pinned ESC-50 revision. It selects only ESC-10 licensed rows from dog, rain, crying baby and clock tick, two clips per class per official fold. Original Freesound recording IDs are checked before any waveform mixing. Folds 1–3 train, fold 4 validation, fold 5 test. The test source recordings never appear in training mixtures. See [provenance and clip hashes](docs/dataset-provenance.json), [selected contributor attribution](docs/ESC10-ATTRIBUTION.md), and [methodology](docs/methodology.md).



Audio files and checkpoints remain local and ignored; reproduce them with the commands above. The manifest stores absolute local paths. Move data by regenerating or editing the local manifest. Only aggregate metrics and public provenance are committed.



## Measured results



The real-audio mixture run achieved **macro F1 0.490**, **micro F1 0.651** and **macro AP 0.644**, compared with always-present macro F1 0.446. The rare clock-tick class scored **F1 0.0** with six test positives; this baseline does not solve rare-event detection. Training supports were 388/196/91/9, approximately 43:1 imbalance. There were 23/7/7 distinct original train/validation/test recordings among 24/8/8 clips.



## Pipeline



1. Decode 16-bit PCM WAV, average channels, anti-alias resample to 8 kHz and center crop/pad to one second.

2. Validate original source IDs across splits, then mix one or more source classes with random shifts and relative gain. Class inclusion probabilities are .65/.35/.15/.03; empty draws receive one weighted class.

3. Compute log-power STFT (256-point Hann window, 128-sample hop). Preserve coarse frequency positions through CNN pooling.

4. Train with per-class focal alpha derived from training prevalence and inverse-prevalence weighted sampling. Augment with gain/noise/time shifts and time/frequency masks.

5. Select each class threshold on validation F1 only. Report held-out macro/micro F1, per-class AP/support and an always-present baseline. Checkpoints contain weights, classes and thresholds and load with `weights_only=True`.



Use `--rare-probability 0.001 --count 8192` for an unmeasured extreme-imbalance experiment. Larger mixture counts are needed to retain rare positives; training rejects any omitted class. This option controls class inclusion before the nonempty-mixture fallback, so always inspect realized positive supports. The committed runs use .03.



Training runs on CPU by default with two Torch threads. `--count` controls training mixtures; each evaluation split uses max(128, count/2) mixtures. Seed 42 is fixed in the CLI for reproducibility. The Python `train` API exposes seed and thread arguments.



## Executed evidence



[Modulated-tone smoke](docs/synthetic-results.json) trains on 512 mixtures with training positive supports 379/199/77/15, approximately 25:1 imbalance, and evaluates 256 held-out mixtures. Its perfect score reflects an intentionally easy frequency-recognition task. [Real ESC-10 mixture run](docs/esc10-results.json) evaluates source-disjoint mixtures from a tiny four-class subset and reports actual measured results, including weak classes. These metrics must not be described as full ESC-50 results or natural overlapping-call benchmarks.



## Layout



Production code is in `src/overlap_audio/`. Requested pipeline entry points live in `src/data/`, `src/features/`, `src/models/` and `src/visualization/`; shared repository paths are in `src/config.py`. The notebook only explores saved metrics. `data/{raw,external,interim,processed}` and model binaries are ignored. CI runs tests and one actual training epoch.



## Limits



Center crops may omit an event while retaining its source-file label. Synthetic mixing creates approximate presence labels, not verified per-window annotations. One-second 8 kHz input discards long context and high-frequency calls. Noise is synthetic; no denoising or source extraction is implemented. Two clips per class/fold provide very weak evidence of recording diversity, and reused sources make mixtures statistically dependent within each split. No confidence intervals or deployment claims are made. Checkpoints should be treated as experimental artifacts. Increase source diversity and evaluate annotated real mixtures before considering practical use.



Code MIT licensed. ESC-10 data retain CC BY 3.0 and individual Freesound attribution; full ESC-50 licensing is different. No audio files are redistributed in this repository.

