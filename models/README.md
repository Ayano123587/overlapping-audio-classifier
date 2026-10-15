# Local model artifacts

Run the documented training commands to create synthetic.pt and esc10.pt. Binary checkpoints are ignored. Aggregate training histories, test metrics and limitations are tracked under docs/. No pretrained model is redistributed.

The executed natural-audio experiment produced `maestro-v1/selected.pt` on a private
Kaggle Tesla T4. Its SHA256 is
`990e7de2784aa569da73ff362aa3b6d7753419b1b7d7553b400d644f7dbb66c8`.
The local copy loads with `weights_only=True` through the documented natural prediction
command. The MAESTRO experimental/noncommercial terms also apply to dataset-dependent
artifacts; checkpoints and derived caches remain ignored and are not redistributed here.
Reproduce training from the official licensed source; use the aggregate report in docs
to inspect selection, scores and unsupported classes.
