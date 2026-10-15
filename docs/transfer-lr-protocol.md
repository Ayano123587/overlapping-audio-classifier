# Conditional final learning-rate iteration

Frozen before any transfer holdout score is opened. The five-second transfer
validation curves peak at epochs 2, 2 and 1 for linear-soft, linear-balanced and
MLP-balanced. Hypothesis: a threefold smaller learning rate provides finer early
optimization and improves validation AP rather than immediately overshooting a
useful transferred representation. This is not evidence that more training will
improve generalization.

If the central-second job completes with at least five minutes left in the same
cumulative 2,700-second allowance, reuse its private cached embeddings. Repeat
exactly the same three head configurations, seed, training-only normalization,
soft-label losses, thresholds, maximum 80 epochs and stopping rule, multiplying
all learning rates by 1/3. No new audio extraction or encoder compute is needed.
Stop at the remaining cumulative runtime, and preserve failures. This is the
last conditional refinement in this bounded study, not an open-ended search.

Select the final checkpoint among five-second original, one-second original and
one-second lower-rate winners by supported-class validation macro AP only.
Exact ties retain the earlier winner. Disconfirmation: no validation improvement
over prior winners. Previously observed holdout metrics cannot choose a learning
rate, context, threshold or whether to run this experiment. Each selected head's
holdout report remains an observed replication and will be disclosed, including
negative results. Declare this protocol unexecuted if time is insufficient.

Reproduction adds `--lr-scale 0.3333333333333333` to the documented transfer train
command, using a fresh output directory. All source/model/data licenses persist.
