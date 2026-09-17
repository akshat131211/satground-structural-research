# Pairing, coverage and objective-gradient diagnosis

This diagnosis precedes any further training. It keeps VIGOR, Sat3DGen, the
current A1/A2/A3 definitions and every accepted mask unchanged. No optimizer
step, new loss weight or model-selection decision is made.

## Camera and coverage

Check all 21 reviewed development views against the pinned source pairing
records and reference camera/crop code. Generate source-only context diagrams
at each view's actual yaw, plus the other three cardinal directions. Keep
images, camera coordinates, identifiers and per-view observations local.
Image-plane distances to a crop edge are not metres or verified visibility
masks. Matching source records/code does not independently validate heading,
capture date, building identity or metric scale.

## Training-only gradient probe

Use all 100 existing training views, without selecting by error, labels or
validation performance. Check the zero-output adapter and saved conservative
A3 steps 300 and 500. Use fixed training illumination, cached scene features,
unchanged renderer settings and a deterministic per-sample rendering seed
shared across states. The seed convention is for a repeatable diagnostic;
it does not reproduce each historical random training draw.

Compute the exact existing weighted RGB, LPIPS, region and boundary components
(weights 1, 0.1, 0.5, 0.5), their gradients with respect to rendered RGB and all
4,448 adapter parameters, their norms and pairwise cosines. Compare boundary
gradients with the A2 objective (RGB + LPIPS + region), and check the combined
image gradient against the unmodified Objective implementation. Retain zero
norms as explicit zeros; undefined ratios/cosines are unavailable, not zero.
Verify adapter tensors, base-model freeze state and checkpoint files remain
unchanged. Do not infer causal generalization from gradient magnitude alone.

Report both per-image median distributions and equal-geographic-group means,
with labelled empty/nonempty training strata. Inspect boundary support: valid
pixels, eroded valid pixels and target-boundary pixels. Record runtime and peak
VRAM. Raw per-sample records and gradient vectors remain local.

This probe measures gradients before AdamW's moment scaling, weight decay,
gradient accumulation and clipping; it does not claim to reconstruct the
historical optimizer's effective update. A small boundary contribution can
motivate a separately controlled experiment, but is not by itself evidence
that increasing its weight would help. No automatic weight sweep or longer run.

The first one-image smoke test exposed a 0.0304% difference between summed
component image gradients and the full objective with default GPU math. The
failed output is preserved. Disabling TF32 for this diagnostic passed the
unchanged strict summation check at all three states. Full results explicitly
record this numerical setting; historical training settings are unchanged.
