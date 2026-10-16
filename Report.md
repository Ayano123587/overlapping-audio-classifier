# Audio experiment report

The selected five-second PANNs head plus box3 reached observed-test AP **0.40298**, compared with **0.39538** before smoothing. Micro F1 increased from **0.46061** to **0.47153**. All-17-class macro F1 is **0.30153**; supported-class macro F1 is **0.39431**.

## Recipe

MAESTRO Real v1: [Zenodo 7244360](https://zenodo.org/records/7244360), 49 recordings, 17 classes, 6,838/2,163/2,368 train/validation/test seconds. Recordings are split before feature extraction. Unknown scene classes are masked; soft scores train the head and strict >0.5 labels evaluate it.

Frozen [PANNs Cnn14 weights](https://zenodo.org/records/3987831) use native 32 kHz and five-second windows. Encoder SHA256: 0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31. The selected MLP has 256 hidden units, dropout 0.3, train-only scaling/prevalence weights, masked soft BCE, AdamW LR 0.0003 and validation checkpoint selection. Thresholds use validation F1. Box3 never crosses recordings or missing seconds.

| Comparison | Validation AP | Observed test AP |
|---|---:|---:|
| Scratch CNN | 0.41552 | 0.37617 |
| Five-second transfer | 0.47009 | 0.39538 |
| Central one-second | 0.41289 | 0.33693 |
| Central lower LR | 0.41432 | 0.33684 |
| Selected five-second + box3 | 0.47670 | 0.40298 |
| Best filtered three-seed V6 candidate, rejected | 0.46659 | No separate release |

V6 trained twelve heads: four LR/dropout/decay configurations, each with seeds 42/1337/2026. No ensemble or fixed mixture beat the retained validation policy. GPU submission was rejected by the weekly service quota; the prospectively amended CPU study completed in 630.39 seconds. The initial transfer smoke failed on inference tensors entering autograd and produced no valid model score.

## Reproduction checks

Retained checkpoint: models/postprocess-v5/selected.pt. SHA256: d082cb659e8ab94b9661fca741630cf12e09b951aa7ec74a544deceb7792c689. Real-waveform inference produced 85 finite probabilities over five seconds. Saved-head replay matched exactly. The main notebook repeats a fixed full-data control fit and checks its recorded state, then replays selected validation inference.

Source checksums, input hashes, framework versions, original protocol snapshots and run identities are in the [ledger](docs/experiment-ledger.json). Detailed historical records remain in Git history.

## Limits

This test was already observed. V5 selected its policy before its own test load; its aggregate was opened after V6 protocol/code freeze but before V6 selection finished. No V6 objective, grid, seed, stopping rule or promotion threshold changed. The overall process was not fully blinded.

AP fell for children voices, coffee machine and shopping cart; F1 fell for children voices and large vehicles. Coffee/cart each have one positive test second, four classes have no positives, and several classes lack training supervision. CPU/GPU and ensemble-size changes complicate comparisons. The study establishes neither reliable rare-class detection, statistical significance, new-location transfer nor an optimum.
