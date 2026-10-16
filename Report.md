# MAESTRO overlapping audio report

Real natural overlapping sound, frozen recording split: 29 train, 10 validation, 10 observed test recordings. Soft training labels, unknown classes masked, strict evaluation labels > 0.5. Macro AP excludes classes without positive support. This is a custom split, not the official five-fold benchmark. Test is already observed and cannot establish untouched generalization.

| Actual learned solution | Validation macro AP | Observed test macro AP |
|---|---:|---:|
| Scratch spectrogram CNN | 0.41552 | 0.37617 |
| Frozen PANNs, five-second MLP | 0.47009 | 0.39538 |
| Central one-second MLP | 0.41289 | 0.33693 |
| Central one-second lower learning rate | 0.41432 | 0.33684 |
| Five-second MLP + recording-safe box3 (V5) | 0.47670 | 0.40298 |
| V6 best three-seed head ensemble + box5 (not promoted) | 0.46659 | Not used for promotion |

The five-second transfer model improved macro AP but reduced micro F1 from 0.46714 to 0.46061. Central context and its lower-rate follow-up failed to beat five-second features. V5 selected smoothing using validation only and removed all central branches. Its three-second lookahead means offline inference, not immediate streaming.

V5 observed micro F1 is 0.47153 and all-17-class macro F1 is 0.30153. Improvements are not universal: AP regressed for children voices, coffee machine, and shopping cart; F1 regressed for children voices and large vehicles. Rare-class performance remains poor. Its actual saved wrapper replay matched exactly and real-waveform inference produced 85 finite probabilities across five seconds. The V5 aggregate was opened after V6 protocol freeze but before V6 selection completed; this is retrospective observed-test evidence, not fully blinded global evaluation.

V6 prospectively compares four five-second learning-rate/dropout/decay configurations, each with three seeds, selected by ensemble validation AP. Its protocol predates V5 test inspection. No further test-guided tuning is permitted. Rare classes remain poorly identified: several have no training positives, and some have only one positive in held-out recordings. See `docs/transfer-evaluation.md` and the immutable JSON protocols for evidence, limitations, provenance, and failed hypotheses.

V6 completed all twelve learned fits on CPU after an actual Kaggle weekly GPU quota rejection. No new ensemble or fixed mixture beat V5 on validation; production retains V5. Best epochs were 1–5, supporting a plateau for this optimization/regularization grid. See `docs/five-second-learned-v6-evaluation.md` for all comparisons and execution limitations. Encoder fine-tuning was not run, and no optimum is claimed.

Run `python train.py --help` for explicit train/validation-only cached-feature training. Run `python evaluation_scripts/evaluate.py --help` for separate saved-policy evaluation; test requires explicit acknowledgement and every output path must be fresh. Data, pretrained weights, learned checkpoints, and caches remain private and ignored under their source licenses.
