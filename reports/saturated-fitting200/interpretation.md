# Saturated valid-surface fitting comparison

The bounded comparison is complete, and **the predeclared fitting response is
not established**. The additional surface loss yields small mean improvements,
but neither co-primary boundary measure meets the required 10% reduction against
both controls. This result does not justify longer training or server allocation.

## Fixed comparison and measured result

The model is the released Sat3DGen with a shared 4,448-parameter residual scene
adapter. A0 is frozen. Fresh A2 and revised A3 each receive exactly 200 updates,
learning rate 0.00003 and seed 17, on the same eight reviewed VIGOR training views.
Both use identical RGB, perceptual and region losses, sampling, rendering,
illumination and frozen reference models. A3 alone adds saturated valid-surface
supervision with coefficient 0.5. Historical boundary loss is diagnostic only.
Only the final update is scored; no checkpoint selection was performed.

| Final fitting metric | Frozen A0 | Fresh A2 | Revised A3 |
|---|---:|---:|---:|
| Full boundary error, lower is better | 0.063514 | 0.061973 | 0.061477 |
| Nonperimeter boundary error, lower is better | 0.116382 | 0.116521 | 0.112260 |
| Building IoU, higher is better | 0.556596 | 0.572563 | 0.577168 |
| LPIPS, lower is better | 0.537679 | 0.538086 | 0.538007 |
| SSIM, higher is better | 0.175838 | 0.176797 | 0.177038 |

Against A2, A3 reduces full boundary error by **0.800%** and nonperimeter error
by **3.657%**. Against A0, the reductions are **3.207%** and **3.542%**. All four
fail the predeclared 10% requirement. Only **3 of 8** views jointly improve both
distances against A2, below the required four. The quality limits pass: IoU rises
by 0.461 percentage points versus A2 and 2.057 points versus A0; LPIPS changes by
-0.0147% and +0.0610%, respectively.

The paired geographic bootstrap intervals describe these eight training examples
only. For improvement against A2, the full-boundary 95% interval is
[-0.000969, 0.002231] and the nonperimeter interval is [-0.001160, 0.012411]; both
include zero. Against A0, full-boundary improvement is positive within the
descriptive interval [0.000941, 0.003093], while nonperimeter improvement has
interval [-0.002311, 0.014522]. These intervals do not establish held-out effects.

All eight targets contain buildings, so the building-target stratum equals the
overall result. The empty-target stratum is **absent**, with no performance
estimate. No scored prediction is empty and no one-empty boundary penalty
explains the means. Mean false-positive area fraction is 0.032784 for A0,
0.034764 for A2 and 0.034289 for A3: A3 is slightly better than A2 but worse than
A0 on this descriptive measure.

## Verification and local inspection

The frozen reporter verifies all 400 ordered finite update rows, matched sample
schedules, eight periodic and two final adapter/optimizer/RNG checkpoints, 24 RGB
predictions, 24 evaluated masks, 24 exact target copies and 24 probability arrays.
The 357 trial source/input pins and 361 runner identity pins are unchanged.
Both runner stderr files are empty, all eight stages completed, the runner and
children exited, and the GPU lock is absent. Stage logs retain upstream notices.
Peak allocated training memory is 663.095 MiB for each run; this is a PyTorch
allocation measurement, not total device memory. Recorded training time is
863.978 seconds for A2 and 868.536 seconds for A3.

The retained A2 first-update checkpoint and its receipt verify. The actual GPU
resume recorded exact immediate adapter, AdamW, scaler and Python/NumPy/Torch
CPU/CUDA RNG readback before the next training draw. That update is included
within A2's 200-update budget. The final audit checks retained evidence on CPU;
it neither repeats GPU restoration nor performs an optimizer update.

Reproduction under a fresh private path yields byte-identical aggregate JSON
and paired gallery. The frozen A0 images also match all eight images of the
earlier reference exactly. Fresh A2 is not a bitwise historical retraining
reproduction: its mean boundary error differs by -0.000294867, IoU by
-0.000301912 and LPIPS by -0.000003848. The cause is not established, and this
comparison uses the fresh matched A2 control throughout.

Assistant inspection covered all eight target/A0/A2/A3 RGB comparisons and their
validity-aware building-mask overlays at native crop resolution. Major layout,
skyline and facade mismatches remain. Changes between A2 and A3 are small, and
high building IoU can coexist with visibly incorrect facade structure. This
inspection does not support a claim of substantially recovered building geometry.
It is not independent human certification of masks or structural accuracy.
The original annotation uncertainty remains recorded and no mask was changed.

The raw aggregate preserves the reporter's `manual_gallery_review_pending: true`
flag because it is written before inspection. The separate
[completion audit](completion-audit.json) records completed assistant inspection;
it does not alter that raw report. Probability hashes were recorded at completion;
the original evaluator did not independently store NPY hashes during evaluation.

## Interpretation and limits

The saturation correction fixes the measured far-distance loss reversal in the
52-case controlled screen and supplies finite nonzero adapter gradients in seven
of eight zero-update probes. At large fixed translations the loss still plateaus.
Passing these prerequisites does not establish improved generated imagery, and
the current fitting trial fails its declared response criterion.

Full boundary error includes crop-closing edges: 1,676 of 2,263 target edge pixels
lie on the image perimeter. The co-required nonperimeter measure removes only
the outermost image rows and columns from the edge sets; it is not an independent
measurement of complete physical building contours. Both measures use the
unchanged reviewed validity masks and an evaluation segmenter distinct from the
training segmenter, so segmentation errors and incomplete visible support remain
limitations. Neither buildings hidden by occlusion nor metric 3D accuracy are
validated here.

This is one seed on eight training examples with incomplete QA. No Seattle,
audit, calibration or reviewed development images were evaluated in this phase.
The absent empty-target stratum, annotation uncertainty, evaluator dependence and
small sample prevent claims of generalization, reliability, novelty or conference
readiness. Earlier negative experiments, excluded views and original annotations
remain preserved. The failed v1 candidate received no optimizer updates.

No longer budget, new seed, sweep or server experiment is queued. A future
proposal would need a separately justified change and a fixed evaluation design;
this negative result is retained rather than converted into an open-ended search.

The training source is commit `a92e69c33fdfa9cb0d79d6ff2c00245800a7e5e3`, whose
[CI run](https://github.com/akshat131211/satground-structural-research/actions/runs/36514158057)
passed 485 tests with two optional CUDA skips and Node checks. Its Linux
full-dependency fingerprint reproduces all 52 synthetic inputs exactly. The
earlier identity-check failure remains unresolved; no guard was relaxed.
Publication of this completion report uses separate exact-commit CI verification.

The [aggregate](aggregate.json), [completion audit](completion-audit.json) and
[status](../SATURATED_SURFACE_STATUS.json) contain aggregate evidence only.
All images, masks, sample identities, per-view records and checkpoint payloads
remain local.
