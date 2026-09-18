# Completed contour correction: structural improvement is not established

The fixed experiment finished: fresh A2 and corrected A3 each received exactly
500 optimizer steps at learning rate 0.00003, seed 17. The contour correction
restored training targets, but this development comparison does **not** establish
a useful structural improvement or justify longer training or server scaling.

## Primary result and ablations

Boundary error is normalized by image diagonal; lower is better. Aggregates and
paired bootstrap intervals give equal weight to geographic groups. Positive
boundary differences below mean improvement. Intervals concern geographic
resampling of these fixed predictions, not variation across training runs.

| Corrected A3 versus fresh A2 | Relative boundary reduction | Absolute reduction, 95% geographic CI | IoU change | LPIPS relative change |
|---|---:|---:|---:|---:|
| All 24 development views, 10 groups | 1.795% | 0.001603 [-0.000228, 0.004741] | +0.109 percentage points | +0.069% worse |
| 21 reviewed views, 10 groups | 1.688% | 0.001627 [-0.000315, 0.004782] | +0.087 percentage points | +0.064% worse |
| 21 targets containing buildings, 9 groups | 2.324% | 0.001832 [-0.000277, 0.005441] | +0.149 percentage points | +0.015% worse |

Every primary boundary interval includes zero. The original three reviewed views
show only 0.046% boundary reduction, with an interval spanning zero. The reviewed
and building-containing strata have the same count but are different selections.

Against **legacy A3**, corrected A3 reduces all-view boundary error by only
**0.029%**, absolute 0.0000256, CI [-0.000185, 0.000202]. IoU falls 0.0196
percentage points and LPIPS increases 0.0078%. This tiny difference cannot support
an improvement caused by the correction, especially given the control drift below.

Against **frozen A0**, corrected A3 has 2.135% lower boundary error, +0.833
percentage points IoU and 0.361% worse LPIPS. The boundary CI spans zero. Against
ordinary **A1**, it has only 0.363% lower boundary error (CI spans zero), +3.608
percentage points IoU and 2.462% better LPIPS. A2 also recovers most of A1's IoU
and perceptual degradation; the recovered quality is not evidence for the new
contour term. All descriptive comparisons use the same targets and settings.

The three building-free targets retain one false-building prediction and one
maximum empty-mask boundary penalty for **every** model. Scored false-positive
pixels are A0: 92, A1: 55, historical A2: 104, fresh A2: 107, legacy A3: 99,
corrected A3: 99, out of 134,838 scored pixels. The new primary difference is not
an empty-mask penalty transition. The eight-pixel reduction versus fresh A2 is
small and does not remove the false prediction.

See the [unchanged generated table](comparison/report.md),
[all strata and intervals](comparison/summary.json), and
[completion audit](completion-audit.json). The generated report deliberately
retains its `interpretation_pending` field; this document supplies that subsequent
interpretation without changing reproducible generated outputs.

## Historical control and numerical diagnosis

Fresh A2 does not reproduce historical A2 bitwise: 0/24 image hashes match. Mean
absolute RGB difference is 0.001935 on a 0–1 scale (about 0.49 of one 8-bit level);
the largest per-image mean is 0.007866. Final adapter relative L2 difference is
0.001868. Fresh A2's boundary error is 0.370% lower, IoU is 0.101 percentage points
lower and LPIPS is 0.054% worse. Its boundary CI excludes zero for these fixed
predictions; this is control drift, not a method result. The drift is larger
than the corrected-versus-legacy A3 boundary difference.

The checks found identical original arrays in all 100 training label files,
unchanged source-image/style/manifest identities, matching recorded environment
and pretrained revisions, and the same A2 hyperparameters. Only label-cache
location, the zero-weight boundary calculation mode and the explanatory QA text
differ. First-step RGB, LPIPS, region and total losses match exactly; the gradient
norm differs by -0.0000117645. The mathematical A2 objective is unchanged, but
its exact computational graph is not asserted identical.

A post-run probe used the first sampled training view at the initial adapter,
without optimizer updates. Five backward passes through **the same retained
renderer graph with the same upstream gradient** differed: relative L2 differences
from the first pass ranged from 0.0000182 to 0.0000376. PyTorch's diagnostic
determinism warnings identified CUDA grid-sampling, reflection-padding and
antialiased bilinear-resize backward operations, plus the absent deterministic
CuBLAS workspace setting. Legacy and corrected zero-boundary-weight objectives
had exactly equal scalar totals on this view; their image gradients differed by
relative L2 0.0000337. Adapter weights stayed unchanged and reference models
stayed frozen. No historical training settings were modified.

This directly demonstrates numerical backward variation in this execution path;
it does not prove that a single operation explains every historical difference.
Separately, fresh A2's fixed checkpoint regenerated **24/24 byte-identical images**.
Thus repeat inference succeeds, while exact retraining does not. No new training
replicate or seed was introduced. Numerical variation remains a limitation of
small method-effect claims.

## Local inspection and verification

The assistant inspected all six local paired-gallery pages covering the 21
reviewed views, including the original three. Corrected A3 and legacy A3 look
very similar. Large discrepancies in building placement, skyline and facade
extent remain across the models, along with distorted foreground objects and
vegetation. Several scene layouts remain visibly unlike their real targets.
There is no clear visual rescue of the largest errors. This is qualitative
assistant inspection, not blinded human scoring or new label approval. The
restricted imagery stays under `runs/contour500-seed17/gallery` and is not in Git.

Verified all 1,000 sequential finite training rows, ten periodic adapter/optimizer
checkpoints and both final checkpoints, 48 new predictions, the shared 21 reviewed
mask identities and all 177 runner pins. The report was recomputed to a fresh
local directory: aggregate JSON agrees after excluding its timestamp, Markdown
agrees byte-for-byte and all six gallery pages have identical hashes. All 110
A2 artifacts preserved before recovery remain unchanged. Stage logs finish
successfully and runner stderr is empty.

A2 recorded 3,441.5 seconds and A3 3,923.4 seconds, with each peaking at 668.3 MiB
of PyTorch-allocated VRAM; these are not whole-device memory or total-project
costs. The planned Windows Update interruption added 43 discarded A3 steps
(315.8 seconds through its last logged step), separately preserved. The three
preflight updates are also separate. Neither contributes to a selected model.

## Decision and limits

The targeted implementation issue is fixed: positive support increased from 48
to 43,127 pixels across 90 of 92 nonempty training views. **Target support is
not target correctness or model improvement.** The labels remain pseudo-labels;
the teacher may place or omit contours incorrectly. Their positive share of
boundary support is still only about 0.84%; this fact alone does not establish
why the method failed or authorize a new loss-weight sweep.

This is one seed, 100 training views and 24 repeatedly inspected development
views. Three evaluation targets remain pseudo-labels, accepted masks retain
documented taxonomy/occlusion uncertainties, and independent pairing, pose,
footprint and near-duplicate QA is incomplete. Development groups may have been
seen by the pretrained model. Geographic intervals do not supply independent
training replication. Seattle, calibration and audit images remain sealed.

The structural-supervision question is **not supported by this bounded test**;
the result is inconclusive, not proof that structural supervision cannot work.
No novelty, audited generalization, calibrated reliability or server-gate claim
is made. Stop this diagnosed experiment and its monitor. Do not automatically
extend training, tune weights, add seeds or allocate servers. A future proposal
should first address independent pairing/pose and training-contour QA and the
remaining large layout errors; it requires a separately specified experiment.
