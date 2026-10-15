# Independent evaluation audit

A separate read-only review checked the natural-audio implementation and all 67,159
official annotation rows. It reproduced 49 unique recording IDs, exactly one-second
intervals, no duplicate class/interval rows, and identical six-class annotation vocabularies
within each scene. The frozen recording split is 29 training, 10 validation and 10 test.
No observed float32 conversion changed the strict >0.5 binary label.

The review found no direct use of test labels/features for model selection. Five-second
context aligns with the central one-second label and includes future context, as disclosed.
Out-of-scene classes are masked consistently in training and scoring. AP tie handling
matches an independent scorer. These checks are evidence, not a guarantee of all source
or location independence.

Confirmed robustness defects were fixed:

- Cached arrays previously discarded recording IDs. Production caches now record source
  WAV and cache SHA256s, row/shape contracts and recording IDs. Training verifies hashes,
  assignments and source disjointness before selection; it inspects held-out IDs/hash bytes
  only at this stage, not held-out features or labels. Loaded arrays also reject misaligned
  rows, nonfinite data, invalid masks and inconsistent feature/class dimensions.
- Empty supported-class cohorts previously produced NaN aggregates. Undefined aggregates
  now serialize as null; all-observed F1 still includes observed classes with zero positives.
  A validation set with no supported AP cannot select a model and is rejected explicitly.
- Preparation rejects empty/nonfinite audio, and inference rejects malformed/nonfinite
  probability thresholds. Scoring validates finite probabilities and binary masks.

These guards were added after Kaggle version2 was launched. That run prepares fresh caches
and trains in the same private process, without importing any reused cache. There is no
evidence its caches were mixed or corrupted. Its recorded source revision remains the
authoritative executed code; do not claim that later integrity guards ran retroactively.
New reproductions require the stricter contracts and may need cache regeneration.

The annotation audit also found no hard training positives for announcement, cash-register
or door classes, and no validation positives for several additional rare classes. Supported
class averages cannot validate those cohorts. Reports must retain every class's support,
all-observed aggregates and unsupported lists. Do not change the frozen split or tune using
test performance to hide this limitation.
