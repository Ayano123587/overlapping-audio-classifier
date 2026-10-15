# Prespecified temporal-resolution follow-up

Frozen before viewing any transfer heldout score. The first transfer experiment
compares three heads on five-second pooled embeddings. A distinct second comparison
will test **central one-second embeddings**, holding all other head configurations,
seed, source split, labels, masks, class order, and validation selection rules fixed.
Hypothesis: five-second audio tagging pools neighboring events that are absent from
the central label; removing surrounding context may improve validation AP for brief
events. Disconfirmation: no improvement in supported-class validation macro AP over
the five-second selected model. One-second audio differs from PANNs pretraining clip
length and may instead reduce performance, particularly for sustained scene cues.

Only launch if the first terminal runtime leaves a sufficient cumulative budget.
Include the failed 133.39-second smoke in the total 2,700-second GPU allowance.
The follow-up runner hard stop is remaining budget, rounded down conservatively.
Retain every failure and all validation histories. Compare context winners using
validation supported-class macro AP only; report tie preference for five seconds.
Each context's selected head may be evaluated once on the already-observed holdout;
neither outcome can select context or trigger retuning. This remains a validation
iteration with observed evaluation, not an independent new final test.

One-second inference covers its labeled second and completes after that second's
audio arrives. It is not a streaming latency benchmark or subsecond localization.
The five-second checkpoint remains supported and retains future offline context.
