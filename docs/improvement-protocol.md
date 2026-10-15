# Performance improvement protocol

The first ESC-10 model is a baseline, not a finished extreme-imbalance solution. Its rare
clock-tick class had F1 zero. The one-second crop could remove the event while retaining
a five-second source-file label, a confirmed limitation. Do not tune repeatedly against
that already-observed test set or describe synthetic mixtures as natural soundscapes.

## Full-clip diagnostic (implemented, not yet run at scale)

`overlap_audio.experiment` introduces full five-second lazy mixtures, a larger CNN,
validation-only architecture/loss/checkpoint selection, and a new final holdout that
excludes every original baseline test recording. It also defines a separate 0.001 rare
inclusion stress test. Its local tests train a one-epoch original-fixture smoke only.
No v2 performance improvement is claimed before an executed numerical report exists.

## Natural overlapping sound (priority experiment)

[MAESTRO Real v1](https://doi.org/10.5281/zenodo.7244360) supplies real-life recordings
and human-aggregated temporal soft labels. It has a custom experimental/noncommercial
license; attribution goes to Machine Listening Group, Tampere University and the
dataset authors. Retain the entire downloaded license. This code's MIT license does
not extend to the dataset, learned dataset-dependent weights, or source audio.

The official record contains 49 recordings. Documentation disagrees about total duration;
use decoded sample counts and prepared temporal-label counts, not an inferred duration.
All source data, derived caches and weights stay local or in a private Kaggle research run.

Hypothesis: learned multi-label models using real time-aligned soundscapes and masked
scene-specific annotation vocabularies will yield stronger rare-class evidence than the
initial cropped-source mixture setup. The datasets and tasks differ, so numerical scores
must not be presented as a directly comparable v1 improvement.

Before training, freeze an entire-recording split within each acoustic scene using fixed
seed `maestro-portfolio-v1`. Approximately 60/20/20 percent of recordings form training,
validation and test. Five-second offline context predicts the central one-second label.
Classes outside a scene's annotation vocabulary are masked as unobserved, not negatives.
Binary evaluation uses the task's strict soft-label >0.5 conversion.

Three prespecified candidates compare weighted soft BCE, focal modulation, and a wider
CNN, with the same source split, seed, data features and maximum 30 epochs. Select the
configuration and epoch using validation supported-class macro average precision, then
select thresholds on validation F1. Do not load test caches during model selection.
Evaluate the selected model once at threshold0.5 and validation thresholds. Report
every class's support and unsupported classes explicitly, alongside an always-positive
baseline. This custom holdout is NOT the official five-fold DCASE challenge score.

Initial cloud budget: one private free Kaggle GPU run, maximum two hours, no paid compute.
Stop for nonfinite loss, malformed data, source overlap, checksum mismatch or GPU failure.
Local tests must first verify aligned labels, finite features, masks, scorer agreement,
gradient behavior and checkpoint selection. Capture source revision, checksums, exact
commands, environment, exits and run identifier. Any subsequent hypothesis needs a fresh
validation comparison; an observed final test cannot be relabeled untouched.

Known remaining risks: recording IDs do not establish location/session independence;
crowd labels are uncertain; rare classes may have no holdout positives; a single seed
does not establish robust superiority; this is offline classification with future context,
not streaming localization or waveform source separation.

The first private Kaggle run obtained a T4 and completed source download, then stopped
before training because one official soft label was `1.000000031316843`. Full annotation
validation identified exactly one float-rounding excursion. The parser now permits at
most 1e-6 rounding outside [0,1], clamps only such excursions, records the count, and
still rejects malformed intervals, nonfinite values and substantive invalid probabilities.
This preprocessing correction changes neither the split nor binary labels. The initial
failure and its environment/log are retained in the portfolio experiment record.
