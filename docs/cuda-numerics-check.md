# CUDA numerical investigation

Environment: WSL2, Ubuntu 22.04, NVIDIA GeForce RTX 3050 Laptop GPU,
PyTorch 2.14.1+cu130.

The failed user run completed 25,000 pretraining updates. Its Critic loss
became infinite during the fourth online training call (TensorBoard cycle 53).
The preceding logged losses were about 111.3 and 71.7, with finite rewards.

An independent CUDA run reproduced a failure using only the bundled offline
data. A captured failure at gradient update 429 had finite observations in
[-1, 7], actions in [-1, 1], rewards in approximately [-0.67, 0.46], and finite
weights with absolute values below 0.18. One Critic Q2 output nevertheless
reached approximately 1.66e33. Its squared loss overflowed, then its gradients
became non-finite.

Re-evaluating the captured inputs and weights on CPU and CUDA yielded Q2
outputs in approximately [-0.62, 1.74]; 1,000 repeated evaluations on each
backend passed. This is evidence of an intermittent computation problem,
not evidence that the saved weights or offline data contain NaNs. It does
not identify the exact driver/library defect.

The project now sets `DISABLE_ADDMM_CUDA_LT=1` before training's first CUDA
linear operation. This option exists in the installed PyTorch CUDA library
and avoids the fused cuBLASLt addmm path while retaining CUDA acceleration.
The setting is applied by both `src/test.env` and `train.py`; an explicit
user-supplied value takes precedence.

A separate scheduling bug was also corrected: SAC actor and target-network
update frequencies now use a continuously increasing gradient-step counter.
Previously, the target was updated throughout one 500-update training call
and not updated at all throughout the next call. The model-save cadence
still uses completed training calls.

Non-finite gradient checks remain enabled. Tests write only temporary
TensorBoard logs and do not overwrite the project's saved models.
