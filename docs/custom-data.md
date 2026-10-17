# Optional synthetic-overlap adapter

The optional `overlap-audio train` command mixes single-source PCM clips into synthetic overlaps. It is separate from the real MAESTRO workflow, whose root `train.py` consumes the complete feature cache. This adapter does not supply the reported MAESTRO results. Build a JSON manifest from your own licensed audio catalog:

```json
{
  "kind": "your catalog description and license",
  "classes": ["class_a", "class_b"],
  "records": [
    {"path": "/absolute/path/audio.wav", "label": 0, "source": "original-recording-id", "split": "train"}
  ]
}
```

Run the optional adapter with fresh local output paths:

```bash
overlap-audio train data/my-manifest.json --checkpoint models/my-mixture.pt --report data/reports/my-mixture.json
```

Each source file has one source class; the pipeline creates multilabel overlap mixtures. Supply at least one source for every class in each of train, validation and test. `label` is the zero-based class index. Every excerpt from a shared original recording must use the same `source` ID and split. Do not assign different IDs to cropped segments from the same recording. Include recording-level provenance and license in your private catalog. For non-four-class inputs, class inclusion probabilities default to .4 each.

Files must be 16-bit uncompressed PCM WAV; convert other codecs offline with appropriate rights and retain provenance. The preprocessing center crops one second, so labels should describe that window. Weak whole-recording labels can otherwise create noise. Downloaded/raw audio, private manifests and trained weights belong under ignored data/ and models/ paths.
