# Completed audio transfer handoff

Branch: codex/improve-v2. Latest executed source: 7fe283a; original five-second
executed source: 96d4f72. Final validation policy was locked in bc17133 before any
transfer holdout report/log was opened. Documentation revision is the commit that
contains this handoff. Root handles publication; no new repository was created.

Final choice: five-second PANNs Cnn14 frozen encoder + learned 256-unit balanced
soft-BCE MLP, validation AP 0.470088. One-second context AP 0.412886 and one-third-LR
one-second AP 0.414323 failed to improve the winner. All nine candidate histories
are retained. Stop at this validation plateau; no further tuning is planned.

Observed selected-model AP 0.395376 vs scratch 0.376170; all-17-class F1 0.294822
vs 0.268783; micro F1 0.460607 vs 0.467140. Improvement is modest and mixed. Coffee
machine/shopping cart have one positive test second each and near-zero F1. Cutlery,
brakes, car and several other measures regress. No claim of untouched test,
official five-fold DCASE score, statistical superiority or solved rare events.

Final checks: python -m pytest -q -> 24 passed in 23.42 seconds. Package wheel
contains transfer modules and upstream MIT notice. Three actual checkpoint
waveform CLI proofs emitted five seconds / 85 finite probabilities each; cached
head serialization replay max difference 0.0. Report/selection/source integrity
and per-class aggregate arithmetic independently checked.

GPU jobs are all terminal. Initial failed smoke 133.39 s, five-second 974.86 s,
central-second 610.75 s, lower-rate head-only 54.12 s; cumulative measured script
processing 1773.12 s (29.55 min), within the authorized 45-minute allowance. Version1
smoke failed before MAESTRO training because an inference tensor entered autograd;
runner cloned it outside inference mode, with the original failure retained.
No live local encoder/training process remains. RAG may use the released GPU slot.

Private cloud IDs:
- aikyatansinha/portfolio-maestro-panns-transfer-v2, version2 COMPLETE
- aikyatansinha/portfolio-maestro-panns-central-second-v3, version1 COMPLETE
- aikyatansinha/portfolio-maestro-panns-head-lr-v4, version1 COMPLETE

Private runners, exact hashes and outputs remain under
D:/CodexData/portfolio/kaggle/audio-transfer-v2,
audio-transfer-central-v3 and audio-transfer-lr-v4. The central job retained
private embedding caches/candidate heads, which the LR job mounted and verified.
No source audio, derived embeddings or model weights are published.

Local usable ignored selected head: models/transfer-v2/selected.pt,
SHA256 12ae9a2f64126cdd0eeb563d8ee70301323e3c35b0bb8dd820b5bfa70083923d.
Required ignored encoder: data/external/panns/Cnn14.pth,
SHA256 0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31.
Private model folders retain the full MAESTRO license. Encoder attribution is
under data/external/panns/ATTRIBUTION.txt. Follow-up checkpoints are also retained
under models/transfer-central-v3 and models/transfer-lr-v4.

Reproduction commands are in README.md and docs/transfer-evaluation.md. Production
CLI: python -m overlap_audio.transfer predict --checkpoint models/transfer-v2/selected.pt
--weights data/external/panns/Cnn14.pth --audio YOUR_WAV --output data/processed/prediction.json.
Use the complete command on one line. Cache/training commands use fresh directories;
checkpoint inference verifies encoder identity/scaling/thresholds.

Remaining limits: observed holdout, one seed, few recordings, unknown location/
session independence, uncertain labels, unsupported/rare classes, native-rate and
pretraining-duration confounders, unaudited pretraining source overlap, and selected
model's offline future context. Official-fold evaluation, streaming recognition and precise event-boundary
localization are not implemented or claimed. No unfinished training or additional project work.
