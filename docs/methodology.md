# Methodology and experiment provenance

## Primary sources

- [ESC-50 official repository](https://github.com/karolpiczak/ESC-50): dataset structure, official folds, `src_file` identifiers, ESC-10 membership and clip license attribution.
- [Pinned upstream license](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE): ESC-10 is Creative Commons Attribution 3.0; the full dataset is Attribution-NonCommercial. Selected contributor notices are preserved separately.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002): modulating cross entropy by `(1-p_t)^gamma`; this project adapts it independently per multilabel class.
- [SpecAugment](https://arxiv.org/abs/1904.08779): time/frequency masking motivation. This implementation uses basic masks without time warping and does not claim full SpecAugment replication.
- [PyTorch STFT documentation](https://docs.pytorch.org/docs/stable/generated/torch.stft.html): complex STFT with an explicit Hann window.

## Source integrity

Split original recordings before constructing mixtures. ESC official folds group recording fragments, and this implementation additionally rejects any `src_file` occurring in multiple selected splits. Synthetic sources have split-specific IDs. Training, validation and test mixture seeds are distinct. Reusing a source across many same-split mixtures is intentional augmentation, but those mixtures are not independent biological or environmental observations.

No fitted preprocessing uses validation/test labels. Focal class weights and sampling weights derive only from training positives. Per-class thresholds are selected from a fixed .1–.9 grid using validation F1. The test set is evaluated after training without tuning on it. The selected threshold is .5 when validation has no positive examples. AP is threshold independent, grouped by tied prediction scores; unsupported test classes report null AP.

## Benchmark interpretation

The synthetic experiment is a four-frequency pipeline smoke test with random amplitude modulation, phase and small frequency jitter. It cannot represent animal-call diversity. Its rarest training label has 15 positives versus 379 for the common class: around 25:1, not an extreme 1:1000 validation.

The real-data experiment downloads 40 ESC-10 clips (24 train, 8 validation, 8 test), then constructs artificial mixtures. Four classes do not establish performance on the full ten- or fifty-class benchmark. Center-crop source labels may be wrong if events occur outside the window. Reported F1/AP measure synthetic mixtures of these specific source crops. Poor rare-class results are retained, not replaced by smoke results.

A natural overlapping audio benchmark with verified time annotations (e.g. DESED), more source recordings, mel/high-rate front ends, and a formal imbalance ablation are useful next experiments. They are not current delivered results.
