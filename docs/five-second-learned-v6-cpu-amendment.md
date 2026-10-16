# Prospective CPU amendment

Before any V6 training or candidate scoring, Kaggle rejected the private GPU submission with `Maximum weekly GPU quota of 30.00 hours reached.` No experiment began. The original frozen GPU runner and source receipt are retained; no paid compute or quota bypass is authorized or attempted.

Run the same cached-feature head study on CPU explicitly, with two Torch threads, one head at a time, identical configurations, seeds, optimizer, loss, maximum epochs, validation plateau stops, and selection rules. Do not load or allocate PANNs locally. Only cached 2048-dimensional authentic features and small learned heads are needed. CPU and GPU floating-point differences are an additional confounder. Historical feature and prior-policy replay guards remain mandatory. Test labels remain closed until the final V6 policy/checkpoint is fixed. CUDA remains the default and fails closed if unavailable; CPU is an explicit amended experiment choice, never a silent fallback.

Actual waveform inference is already proven for the same V5 production wrapper on GPU. A newly selected V6 head still requires saved-checkpoint cached replay; full waveform inference must be reported separately as completed or pending, without claiming an unrun encoder check.
