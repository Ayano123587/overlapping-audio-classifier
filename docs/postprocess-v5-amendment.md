# Prospective compute and training amendment

The user explicitly removed the earlier GPU limit before this round's cloud job
was launched. The original protocol and its commit remain authoritative history;
the one-run / 900-second artificial restriction is withdrawn prospectively.
Use free private Kaggle resources as available, fail without GPU, preserve service
quota failures, and do not purchase paid compute. Remaining service quota is not
exposed by the current CLI and must be recorded as unavailable rather than zero.

The three already-frozen postprocessing stages, metric, recording split, closed
test-label rule and candidate sets are unchanged. After their policy is saved,
retain complete authentic five-second embeddings and original label/mask contracts
privately. Validation/test embeddings are reused; encode only the remaining training
recordings. This avoids repeated full-audio downloads for subsequent learned work.

Next training hypothesis, before inspecting this round's test outcome: the original
five-second MLP's best validation epoch was one, suggesting a slower optimization
step and stronger regularization deserve a controlled comparison. Compare the
original MLP against learning rate 0.0001, then dropout 0.5 at that rate, then
weight decay 0.1 with the same dropout/rate. Use seeds 42, 1337 and 2026, retain every
best-epoch checkpoint and all histories, and avoid selecting a favorable seed.
The exact configuration/ensemble selection rule will be committed before any new
training candidate is scored. Train/validation data alone govern model choices;
the old test remains observed and cannot govern a new study or recover independence.

No guaranteed improvement is promised. Stop individual fits on nonfinite values,
source/cache violations or validation plateau. Preserve negative hypotheses. Any
encoder fine-tuning requires a subsequent validation-motivated protocol with
train-only augmentation and the same recording boundaries, not an unplanned search.
