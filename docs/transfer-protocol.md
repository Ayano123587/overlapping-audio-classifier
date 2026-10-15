# Frozen pretrained audio study

This bounded follow-up uses the existing MAESTRO Real v1 recording split, now an
**observed holdout**, to test whether frozen AudioSet-pretrained PANNs Cnn14 features
improve validation macro AP over the scratch CNN's 0.41552. No claim of untouched
test evidence or official DCASE five-fold performance will be made. The existing
holdout AP 0.37617 is context, never a selection criterion.

The main change is a frozen pretrained encoder. Five-second centered offline
waveform windows predict the central one-second label, identically aligned to the
earlier experiment. Native encoder rate is 32 kHz; the scratch encoder used 16 kHz.
This is a confounder in an encoder-family comparison, not a controlled architecture
ablation. Existing source hashes, scene annotation masks and strict >0.5 binary
labels remain in force. Zero-positive classes remain in all-observed F1; undefined
AP is null. Unknown labels cannot affect the loss or metric.

Three candidates, fixed before launch: linear unweighted soft BCE (lr .001), linear
weighted soft BCE (lr .001), and 256-hidden ReLU/dropout .3 weighted soft BCE
(lr .0003). AdamW decay .01, batch 128, seed 42, maximum 80 epochs; stop after at least 20
epochs with 12 epochs without improved validation supported-class macro AP. Weighted
heads use train-only strict-positive prevalence, positive weight clipped [1,20].
Normalization uses training embeddings only. Select epoch/head by validation AP,
then per-class thresholds by validation F1. Evaluate only the frozen winner on the
observed holdout once. No test-based changes. One private free GPU run, total 45 min
hard stop, fail if GPU unavailable, stop for checksum/finite/split violations.

Official source: https://zenodo.org/records/7244360, DOI 10.5281/zenodo.7244360.
MAESTRO custom experimental/noncommercial license remains authoritative; retain
downloaded full license, do not publish audio, annotations or derived weights.
Pretrained encoder source: https://github.com/qiuqiangkong/panns_inference (MIT).
Pinned inference wheel 0.1.1 SHA256
97f6b56b6c9467cf00e21f041e1f88933188c65c1b5ca64eeb3c92e37fb27fc2.
Pretrained weights: https://zenodo.org/records/3987831 (CC-BY 4.0),
Cnn14_mAP=0.431.pth, publisher MD5 541141fa2ee191a88f24a3219fff024e;
actual SHA256 recorded at extraction. Dependencies torchlibrosa 0.1.0/librosa 0.11.0.
Only the wheel's models.py and pytorch_utils.py are vendored, with whitespace
normalized and no semantic changes, retaining the
MIT notice. Original wheel module SHA256 values are respectively
91f99e879bd42d93850f819914086dd1167f1067f26550a0561f25c4eefa157f and
8a97dcec4d8dfa1f3ceb93de909c8ffe78828350ab011f18c2518659e687a5eb.
The upstream package initializer is intentionally excluded: it downloads irrelevant
AudioSet label metadata using a shell command without checksum verification.
PANNs paper: https://arxiv.org/abs/1912.10211 (Kong et al.). MAESTRO postdates the
published 2020 weights, but clip-level pretraining/source overlap has not been audited.

Limitations: one seed, few independent recordings, correlated overlapping windows,
no official folds/location-disjoint guarantee, uncertain crowd labels, unsupported
rare classes, and offline future context. Official-fold retraining is outside this
one-run budget; report this explicitly. Cache hashes and recording assignments are
validated afresh. Current guards run in this follow-up rather than being retroactive.
