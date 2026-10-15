# Executed MAESTRO natural-audio evaluation

The private Kaggle version 2 run completed successfully in **1,075.26 seconds** on a Tesla T4. It trained on real, naturally overlapping MAESTRO Real v1 recordings rather than synthetic waveform mixtures. The fixed recording split contains **29 training, 10 validation and 10 test recordings**, spanning five acoustic scenes. Source recordings, annotations, caches and learned weights remain local or private.

The model consumes five seconds of offline audio at 16 kHz to predict the central one-second label. Context includes future audio within the recording, so this is not streaming recognition. Scene-specific unannotated classes are masked rather than treated as negatives. Human soft labels are retained during training; binary scoring uses the strict soft-label `> 0.5` rule.

## Selection and finite-training audit

Three configurations were prespecified. Each used the same recording split and seed, with at most 30 epochs and validation-based early stopping. Configuration and checkpoint selection used validation supported-class macro average precision (AP); class thresholds were then selected on validation F1. The executed source loads the test cache only after those selections. No candidate was chosen from test performance.

| Candidate | Width | Loss | Learning rate | Epochs completed | Selected epoch | Best validation AP |
|---|---:|---|---:|---:|---:|---:|
| soft-bce | 24 | bce | 0.001 | 18 | 10 | 0.40695 |
| soft-focal | 24 | focal | 0.001 | 23 | 15 | 0.41040 |
| soft-bce-wide | 32 | bce | 0.0005 | 15 | 6 | 0.41552 |

The selected model is the **width-32 soft BCE CNN**, learning rate 0.0005, checkpoint epoch 6. All 56 recorded training losses and validation AP values are finite. Separate candidate selection files agree with the final report, and each recorded best epoch is the maximum validation AP in its history. These checks audit the recorded run; they do not independently re-decode every waveform or establish that the model learned every rare class.

## Once-evaluated custom test split

| Model / thresholds | F1 across all 17 observed classes | F1 across 13 supported classes | AP across 13 supported classes | Micro F1 |
|---|---:|---:|---:|---:|
| Selected model, fixed 0.5 | 0.23978 | 0.31355 | 0.37617 | 0.35440 |
| Selected model, validation thresholds | 0.26878 | 0.35149 | 0.37617 | 0.46714 |
| Always present baseline | 0.20848 | 0.27262 | 0.18639 | 0.31619 |

All 17 classes have observed test labels, but four have **zero positive test labels**: announcement, cash register beeping, door opens/closes and furniture dragging. Their F1 is retained as zero in the all-observed average; AP is undefined and recorded as null. Supported-class F1/AP average only the 13 classes with positive test support. Micro F1 pools observed class/window decisions and can be dominated by common classes. These definitions must accompany the headline scores.

The model exceeds the always-present baseline on these aggregate measures, but rare-event performance remains weak. Coffee machine and shopping cart each have only one positive test second and F1 near zero. The independent annotation audit also found no hard training positives for announcement, cash register or door, and no validation positives for several additional classes. Such cohorts cannot support a claim that all classes or extreme imbalance have been solved.

| Class | Positive test seconds | Observed test seconds | F1 at validation thresholds | AP |
|---|---:|---:|---:|---:|
| announcement | 0 | 483 | 0.00000 | undefined |
| birds_singing | 56 | 412 | 0.25641 | 0.25725 |
| brakes_squeaking | 55 | 481 | 0.16667 | 0.30844 |
| car | 689 | 893 | 0.93587 | 0.97955 |
| cash register beeping | 0 | 483 | 0.00000 | undefined |
| children voices | 113 | 2368 | 0.10646 | 0.16816 |
| coffee machine | 1 | 391 | 0.00510 | 0.00286 |
| cutlery and dishes | 25 | 391 | 0.19048 | 0.11953 |
| door opens/closes | 0 | 601 | 0.00000 | undefined |
| footsteps | 213 | 2368 | 0.32665 | 0.22125 |
| furniture dragging | 0 | 391 | 0.00000 | undefined |
| large_vehicle | 112 | 481 | 0.37902 | 0.48993 |
| metro approaching | 93 | 601 | 0.61091 | 0.73525 |
| metro leaving | 61 | 601 | 0.41948 | 0.43514 |
| people talking | 1160 | 2368 | 0.83312 | 0.89951 |
| shopping cart | 1 | 483 | 0.00413 | 0.01818 |
| wind_blowing | 89 | 412 | 0.33503 | 0.25516 |

This is a **custom recording-disjoint holdout**, not the official five-fold DCASE challenge protocol. Even fields named `official_class_macro_*` in the archived raw report are class-subset diagnostics on this custom holdout, not official challenge results. Location/session independence beyond recording filenames is unavailable. There are no confidence intervals or multiple-seed robustness results. MAESTRO natural soundscapes and the earlier four-class ESC-10 synthetic mixtures have different labels, classes and evaluation units; their scores are not directly comparable.

## Exact execution evidence

The public aggregate record [maestro-results.json](maestro-results.json) preserves the original cloud protocol and per-class report, all candidate histories, environment, completion status, source/checkpoint hashes and local audit conclusions. No weights, waveforms, per-window predictions or source annotation tables are copied into docs.

- Executed source revision: `5bc408e669f00281fb0c1c7c5183c21142099def`.
- Embedded source archive SHA256: `0b315456a6e3d5e9817c8efcb444943cbb23b0550da307a95746a246d2b3685b`; independently recomputed from the private runner.
- Private runner SHA256: `868a76a038a75731c4f4d4207fe9274742d7c9abf126f266fe915274ccaf68fb`.
- Selected checkpoint SHA256: `990e7de2784aa569da73ff362aa3b6d7753419b1b7d7553b400d644f7dbb66c8`.
- Environment: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1, Tesla T4.

The production predictor loaded that actual checkpoint on an existing five-second ESC WAV and emitted five one-second outputs with 85 finite probabilities in [0,1]. This confirms inference mechanics only; it is not additional MAESTRO accuracy evidence.

## Failed run and later source hardening

Version 1 stopped before training because one official annotation value was `1.000000031316843`. A full annotation audit identified one tiny floating-point excursion. Version 2 permitted at most 1e-6 roundoff, clamped that excursion and preserved the split and strict binary labels. The version-1 environment/log hashes are retained alongside the successful run; it was a real preprocessing failure, not a discarded unfavorable test score.

The cache identity/hash checks, additional finite-input guards and empty-supported-cohort handling in [independent-audit.md](independent-audit.md) were added **after version 2 launched**. Current integration tests pass 20 cases, but those later guards did not run retroactively in the cloud experiment. Version 2 prepared fresh caches and trained in the same private process; the review found no evidence of mixed or corrupted caches. Its archived source revision remains authoritative for executed behavior. Future reproductions use the stricter current contracts and may need cache regeneration.

MAESTRO Real v1 is licensed for experimental, noncommercial use by the Machine Listening Group at Tampere University and the dataset authors. Retain the full source license and attribution. The repository software license does not relicense audio or learned dataset-dependent weights.
