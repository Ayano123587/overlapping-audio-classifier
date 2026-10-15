# Executed pretrained transfer evaluation

The selected five-second PANNs transfer model delivers a modest, mixed improvement on the **previously observed** MAESTRO recording holdout. Supported-class AP rises from **0.37617 to 0.39538**, and F1 across all 17 observed classes rises from **0.26878 to 0.29482**. Micro F1 decreases from **0.46714 to 0.46061**. This is one-seed descriptive evidence, with no claim of statistical superiority or solved rare-event recognition.

Frozen AudioSet-pretrained Cnn14 embeddings feed a learned 256-unit ReLU/dropout head. Training retains human soft labels and masks unknown scene classes. The selected weighted-soft-BCE head uses learning rate 0.0003, training-only normalization and positive weights, and its validation-selected checkpoint is epoch 1. Class thresholds come only from validation F1.

## Controlled iterations and stopping

Three heads were compared in each of three prespecified studies. Five-second embeddings were selected first; a central-second temporal-resolution hypothesis then reduced validation AP. A final head-only study repeated the one-second heads at one-third learning rates, prompted by very early validation peaks. The smaller steps gave only a slight recovery within that context and did not improve the global winner. The study stops at this validation plateau.

| Study | Best validation AP | Observed AP, 13 supported | Observed F1, all 17 | Observed micro F1 |
|---|---:|---:|---:|---:|
| Scratch CNN baseline | 0.41552 | 0.37617 | 0.26878 | 0.46714 |
| Five-second transfer, selected | 0.47009 | 0.39538 | 0.29482 | 0.46061 |
| One-second transfer | 0.41289 | 0.33693 | 0.25610 | 0.47657 |
| One-second, one-third LR | 0.41432 | 0.33684 | 0.26472 | 0.48288 |

The final policy was committed as **bc17133 before opening any transfer holdout report or log**. All nine candidate checkpoints and three study winners were selected using validation AP; exact ties retain the earlier winner. Each cloud script computed its selected-head observed-holdout report after its own validation selection. Those reports remained closed during cross-study selection. The original scratch holdout had already been observed in the preceding experiment, so this follow-up cannot establish untouched-test generalization.

The central-second studies have higher observed micro F1 than the selected transfer model. Model selection remains governed by the previously frozen validation AP objective. Validation AP covers 12 classes with positive validation support; observed-holdout AP covers 13. These cohorts differ. Every study report and candidate history is retained in [transfer-results.json](transfer-results.json); negative outcomes were not discarded.

## Rare classes and regressions

There are 13 classes with positive test support and four without positive test labels. AP is undefined for the latter and recorded as null; their F1 remains zero in the all-observed average. Coffee machine and shopping cart each have only one positive test second, and their F1 remains near zero. Announcement, cash register and door have no hard training positives; furniture dragging and shopping cart each have only one. Positive weighting does not create missing supervision. Coffee machine and shopping cart also have no positive validation examples, so their threshold fallback is 0.5 rather than an informative calibration estimate.

| Class | Positive test seconds | Scratch AP | Transfer AP | Scratch F1 | Transfer F1 |
|---|---:|---:|---:|---:|---:|
| announcement | 0 | undefined | undefined | 0.0000 | 0.0000 |
| birds_singing | 56 | 0.2573 | 0.3007 | 0.2564 | 0.4773 |
| brakes_squeaking | 55 | 0.3084 | 0.1980 | 0.1667 | 0.2046 |
| car | 689 | 0.9795 | 0.9584 | 0.9359 | 0.8219 |
| cash register beeping | 0 | undefined | undefined | 0.0000 | 0.0000 |
| children voices | 113 | 0.1682 | 0.1619 | 0.1065 | 0.2604 |
| coffee machine | 1 | 0.0029 | 0.0041 | 0.0051 | 0.0051 |
| cutlery and dishes | 25 | 0.1195 | 0.0795 | 0.1905 | 0.0000 |
| door opens/closes | 0 | undefined | undefined | 0.0000 | 0.0000 |
| footsteps | 213 | 0.2212 | 0.4419 | 0.3266 | 0.3536 |
| furniture dragging | 0 | undefined | undefined | 0.0000 | 0.0000 |
| large_vehicle | 112 | 0.4899 | 0.5221 | 0.3790 | 0.4757 |
| metro approaching | 93 | 0.7352 | 0.6985 | 0.6109 | 0.6520 |
| metro leaving | 61 | 0.4351 | 0.4694 | 0.4195 | 0.4912 |
| people talking | 1160 | 0.8995 | 0.8764 | 0.8331 | 0.7880 |
| shopping cart | 1 | 0.0182 | 0.0054 | 0.0041 | 0.0041 |
| wind_blowing | 89 | 0.2552 | 0.4235 | 0.3350 | 0.4781 |

Footsteps AP improves substantially, while brakes, cutlery, car and several other class measures regress. These are descriptive class outcomes; no class-specific choice or correction was made from them. All metrics use the same strict soft-label >0.5 conversion and scene masks.

## Execution and reproducibility

All 49 annotation hashes and the recording assignment match the scratch baseline. The two embedding preparations have identical audio/annotation hashes, encoder identity, row counts and class order. There are 29/10/10 recordings and 6,838/2,163/2,368 labeled seconds in train/validation/test. The final lower-rate study mounted the original private one-second caches and verified their contracts; it did not repeat audio extraction.

Measured script processing totaled **1,773.12 seconds (29.55 minutes)**, including a retained 133.39-second failed smoke. Version 1 loaded the real pretrained model but passed an inference-mode tensor to an autograd head. The runner repaired this by cloning the tensor outside inference mode. No MAESTRO training or validation occurred in that failed attempt. Subsequent runs completed on Tesla T4 hardware with hard stops constrained to the cumulative 45-minute allowance. Queue and platform setup time are not included in script timings.

Actual saved checkpoints reproduce cached head probabilities with maximum difference **0.0** in all three studies. Production waveform CLI checks also loaded each actual checkpoint and original encoder and emitted five seconds / 85 finite probabilities. The first two use a real MAESTRO recording crop; the lower-rate check uses one checksum-pinned, attributed ESC-10 WAV. These checks establish inference mechanics, not additional accuracy. Final local source tests: **24 passed in 23.42 seconds**. The package wheel was checked to contain the transfer modules and upstream MIT notice.

The selected local artifact is `models/transfer-v2/selected.pt`, SHA256 `12ae9a2f64126cdd0eeb563d8ee70301323e3c35b0bb8dd820b5bfa70083923d`. It requires the original encoder SHA256 `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`. The predictor verifies encoder identity, training feature scaling and validated thresholds. Audio, caches and all learned weights remain ignored and private.

## Provenance and limits

[MAESTRO Real v1](https://zenodo.org/records/7244360) is supplied by the Machine Listening Group at Tampere University under the [full experimental/noncommercial notice](MAESTRO-LICENSE.txt). [PANNs weights](https://zenodo.org/records/3987831) are attributed to Kong et al. under the source record’s CC-BY 4.0 terms; the [PANNs paper](https://arxiv.org/abs/1912.10211) and pinned MIT inference code are identified in [the frozen protocol](transfer-protocol.md). The source data terms continue to govern dataset-dependent heads.

This uses a custom recording split. The [official DCASE protocol](https://dcase.community/challenge2023/task-sound-event-detection-with-soft-labels) requires five fold rounds, so these values are not official challenge scores. Official-fold evaluation was outside this bounded compatible-baseline/context study. Source location/session independence and pretraining clip overlap are unverified. Native 32 kHz transfer versus 16 kHz scratch introduces an encoder/preprocessing confounder. One-second inputs differ from PANNs pretraining duration. Correlated windows, one seed, uncertain crowd labels and thin rare-class support limit generalization claims. The selected five-second model uses future offline context and does not perform streaming recognition, waveform separation or precise event-boundary localization.
