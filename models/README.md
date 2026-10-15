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

The five-second pretrained transfer checkpoint is local at `transfer-v2/selected.pt`;
its SHA256 is `12ae9a2f64126cdd0eeb563d8ee70301323e3c35b0bb8dd820b5bfa70083923d`.
It requires the original Cnn14 encoder SHA256
`0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`, at
`data/external/panns/Cnn14.pth`. Encoder weights are attributed to Kong et al. and
Zenodo record 3987831 (CC-BY 4.0); MAESTRO-derived heads retain the separate
experimental/noncommercial source terms. Keep the complete MAESTRO notice with
private copies. Context-specific checkpoints record native 32 kHz preprocessing,
training-only feature scaling and validation thresholds. The production inference
command verifies encoder identity before applying the head. No binary is public.
