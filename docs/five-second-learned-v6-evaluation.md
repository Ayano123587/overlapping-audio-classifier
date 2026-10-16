# Controlled learned-head follow-up

The prespecified study trained twelve actual 2048→256→17 MLP heads on authentic MAESTRO five-second PANNs embeddings. Kaggle rejected the GPU submission before launch because its weekly GPU quota was exhausted. The prospective CPU amendment preserved the complete candidate grid and selection rules: two CPU threads, one fit at a time, three fixed seeds per configuration, train-only normalization/prevalence weights, masked soft BCE, AdamW, and validation plateau stopping. No encoder was allocated or fine-tuned locally. The complete study took 630.39 seconds.

| Configuration | Three-seed ensemble validation AP | Individual seed mean AP | Seed sample SD |
|---|---:|---:|---:|
| Control, LR 0.0003/dropout 0.3/decay 0.01 | 0.45998 | 0.45586 | 0.01021 |
| LR 0.0001/dropout 0.3/decay 0.01 | 0.45959 | 0.45837 | 0.01009 |
| LR 0.0001/dropout 0.5/decay 0.01 | 0.45895 | 0.45757 | 0.00670 |
| LR 0.0001/dropout 0.5/decay 0.1 | 0.45929 | 0.45748 | 0.00674 |

The fixed equal-seed rule chose control, not a favorable individual seed. Five-second smoothing raised that ensemble's validation AP to 0.46659, below the retained V5 policy's 0.47670. Fixed learned mixture weights 0.25/0.5/0.75/1 all failed to beat V5; weight zero retained it. Every fit's best epoch was 1–5, then training continued to the prespecified minimum 20 epochs with plateau stopping. Stronger regularization and slower optimization failed this validation hypothesis; more epochs on the same objective are not supported by these curves. All seed histories and candidates remain in `five-second-learned-v6-selection.json`.

The retained policy is the actual previously trained five-second head plus recording-safe centered box3. Three seconds of lookahead restrict it to offline or delayed use. Its unchanged observed test macro AP is 0.40298, supported-class macro F1 0.39431, all-class macro F1 0.30153, and micro F1 0.47153. The saved V6 checkpoint's cached replay difference is exactly zero; its metrics equal the prior policy. The original V5 checkpoint, whose real-waveform GPU inference already passed, is retained as the production artifact rather than presenting the negative retraining round as a new successful model.

V5's observed aggregate was opened after V6 protocol/code freeze but before V6 selection completed. The V6 runner loaded its test labels only after its policy/checkpoint freeze. This broader process was not fully blinded; no untouched or independent test claim is appropriate. No grid, seed, stop, promotion threshold, or objective changed in response to the V5 report. Repeated selection on ten validation recordings creates optimism, and CPU versus previous GPU training is a hardware confounder. Thin and absent class supervision remains unresolved.

An encoder fine-tuning/data-augmentation hypothesis would require a new prospective protocol and available compute. It was not run; GPU quota exhaustion must not be bypassed. This round stops after the controlled learned comparisons plateau, retains the measured smoothing gain and negative results, and makes no claim of optimality or statistically established improvement.
