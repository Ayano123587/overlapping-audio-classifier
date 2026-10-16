# Active audio improvement handoff

Scope: existing overlapping-audio-classifier only. Branch codex/improve-v2.
User removed the earlier artificial GPU cap; use free private Kaggle resources,
fail fast on malformed data/checkpoints, preserve service quota failures, never
purchase paid compute. Audio has first GPU priority coordinated through root.

Postprocessing objective/candidates were frozen before scoring in b4da9eb; exact
private input hashes and guarded implementation in e12eb4d. Prospective compute
amendment and five-second-cache retention in b16abf1 preserve original protocol
history. CPU stages completed: MLP pair AP 0.413789 versus original central 0.412886;
box-5 smoothing improves pair to 0.439523, still below five-second baseline 0.470088.
Broader ensembles, box-9 and causal EWMA alternatives were retained, not discarded.

Private GPU aikyatansinha/portfolio-maestro-audio-postprocess-v5 version1 is RUNNING.
Its frozen source b16abf1, bundle SHA256
28f466a5bc307d524b3085778063de3190aabfc96cacfb1c2b1a9f96c288deba;
runner SHA256 eb11211185a929de3318d3498bb8854fdcdc9c0df5b50e1a3111dfeee03f00ed.
Runner/staging directory D:/CodexData/portfolio/kaggle/audio-postprocess-v5/kernel.
It downloads authentic audio/license only, validates every used source SHA256,
requires exact historical five-second validation AP replay, freezes combined policy
before opening test labels, performs actual waveform inference, and persists complete
private five-second caches after selection for the next learned study.

User explicitly wants actual trained improvement beyond cached postprocessing.
Next learned protocol frozen in 9d7d69f BEFORE any learned candidate scores or this
round's test inspection: control, slower learning rate, dropout 0.5, decay 0.1;
three fixed seeds 42/1337/2026. Choose configuration by equal-seed-ensemble validation
AP, report every seed/history and mean/std, then fixed smoothing/mixing choices.
Tests for new learned code pass 6 cases (learned + postprocess) in 75.46 s;
prior full suite 28 passed in 31.25 s before learned module/optional Head dropout.
No local encoder allocations. Further full tests needed after learned source settles.

Current next actions: finish learned implementation/testing, wait for V5 terminal,
retrieve selection/completion before any observed-test report, then launch actual
multi-seed learned-head GPU study on mounted five-second cache, with no full audio
refetch. Production selected model uses strict versioned checkpoint schema,
source identity, train-only moments, class order, pruned zero-weight paths and
validation thresholds; tests cover masks, boundaries, gaps, causal EWMA, model
state/normalization failures and exact replay. Preserve all failures and negative
hypotheses. Root publishes clean commits and manages eventual goal pause.

## Current checkpoint: 16 October 2026 learned round

V5 private GPU kernel completed, elapsed 1018.85 s. Historical five-second validation AP replay was exact; selected box3 AP 0.4767020 versus 0.4700883. Its observed test AP 0.4029755 versus 0.3953760; supported F1 0.3943097; micro F1 0.4715285. Actual real-waveform proof passed (5 rows, 85 finite probabilities), cached checkpoint replay difference 0. Full aggregates and source/environment receipts are under docs/postprocess-v5-*.json. Private checkpoint: D:/CodexData/portfolio/kaggle/audio-postprocess-v5/output/postprocess-v5/selected/selected.pt; SHA256 d082cb659e8ab94b9661fca741630cf12e09b951aa7ec74a544deceb7792c689.

Kaggle rejected V6 launch before execution: maximum weekly GPU quota 30 hours reached. No paid compute or bypass. Source runner retained D:/CodexData/portfolio/kaggle/audio-learned-v6/kernel; public rejection receipt in docs. Prospective CPU amendment commit 77c4617 keeps identical configs/seeds/rules, explicit CPU only, two threads and one head at a time. No local PANNs allocation. Actual study runs exec session 32798, command: PYTHONPATH=src python -u -m overlap_audio.learned --input-root D:/CodexData/portfolio/kaggle/audio-learned-v6/local-input --output models/learned-v6-cpu --device cpu. Train/validation caches authentic and hash-checked. Individual seed checkpoints/histories saved atomically between fits; study protocol records exact private inputs.

V5 observed aggregate was opened after V6 protocol/code freeze but BEFORE V6 selection completed; see docs/postprocess-v5-timing.md. Do not claim fully blinded global selection or untouched test. No candidate/rule changes based on that report. V6 own labels open only after final policy/checkpoint.

Root train.py is train/validation only; evaluation_scripts/evaluate.py defaults validation, requires explicit observed-test acknowledgement and fresh outputs. Report.md carries concise history. Full suite: 32 passed, 86.52 s after final source changes. Root publishes; source/docs only. Next: finish V6 immutable selection, observed report and actual checkpoint proof, document negative hypotheses and stop on evidence/quota rather than fabricate gains.

## FINAL authoritative state: completed audio round

No active audio jobs or training processes remain. V5 GPU COMPLETE; V6 GPU submission REJECTED before start due weekly 30-hour quota; no retry/payment/bypass. V6 explicit CPU study completed all twelve fits in 630.39 s. Every new three-seed configuration, smoothing alternative and fixed mixture failed to beat prior validation AP 0.4767020191. Retain original V5 production checkpoint at models/postprocess-v5/selected.pt, SHA256 d082cb659e8ab94b9661fca741630cf12e09b951aa7ec74a544deceb7792c689. Exact encoder remains data/external/panns/Cnn14.pth, SHA256 0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31. Root may copy these ignored files into original repository; never publish binaries/data.

Actual V6 saved state equals retained production recursively including every tensor, normalization, threshold, class and policy. Actual evaluation_scripts CLI on V6 saved checkpoint matched frozen validation AP exactly; test replay difference zero and immutable selection unchanged. Original retained production passed GPU real-waveform proof (5 rows/85 finite probabilities). Observed test AP 0.4029754993, all-class F1 .3015309296, supported F1 .3943096772, micro F1 .4715284715. Per-class regressions/rare limitations and earlier-report timing caveat explicit. Full 32 tests passed after final source changes; subsequent edits aggregate/docs/notebook only.

Public evidence: docs/postprocess-v5-*.json, docs/five-second-learned-v6-{protocol,selection,observed-holdout,completion,cli-validation}.json, docs/production-checkpoint-receipt.json; readable Report.md and docs/five-second-learned-v6-evaluation.md. Private inputs/runner under D:/CodexData/portfolio/kaggle/audio-learned-v6; historical cache hashes and source receipts pinned. Every individual seed history retained in aggregate selection and ignored checkpoints. Notebook EDA only. No claim of untouched test, solved rare classes, encoder fine-tuning, statistical significance, or optimality. No further identical-grid tuning justified by early validation plateau; new encoder/data hypothesis requires separate protocol and available compute. Root handles publication/CI/user goal management.
