# Diagnosis and the next controlled experiment

All eight intermediate evaluations completed. Together with the original final
checkpoints and frozen A0, the [report](report.md) contains eleven verified
model states scored with identical current labels and poses. No training was
performed to obtain this diagnosis; no earlier result or primary checkpoint
was replaced. The [trajectory chart](trajectory.png) plots the aggregate evidence.

## What the trajectory shows

- A1 has LPIPS 0.561335 at step 100 and 0.592269 at step 200: a **5.51% worsening**.
  On the 22 targets with scored building pixels, IoU falls from 0.491255 to
  0.407967 over the same interval. Deterioration begins before the large boundary
  jump at step 400. Fixed-view inspection shows increasing blur/ghosting.
- A1 training-window mean RGB error decreases from 0.205714 (steps 1–100) to
  0.178339 (401–500), while LPIPS rises from 0.571352 to 0.582936. Its output-layer
  weight norm grows from 0.2091 at step 100 to 0.4334 at step 500. This is
  consistent with stronger adaptation trading perceptual quality for pixel
  reconstruction. These sampled-window statistics do not prove overfitting or
  establish learning rate as the cause.
- A3 is more stable in LPIPS and improves the building-target aggregate over
  later steps. Yet its final boundary errors worsen on the same three reviewed
  views compared with A0 and A1. Its sharper images still miss actual buildings.
  Aggregate improvement and faithful building reconstruction are not equivalent.
- There are **22 nonempty and two empty scored target masks**. At step 500,
  empty-target false-positive area is 0.140% for A1 and 1.857% for A3, averaged
  equally across their two geographic groups. Nevertheless, the original
  empty-target boundary error is 1.0 for A1 and 0.5 for A3. A1 has a nonempty
  prediction in both empty targets (139 pixels in total); A3 has one (1,889
  pixels). The discontinuous empty-boundary convention favors eliminating one
  tiny component even while the total false-positive area increases. Both
  diagnostics are retained; no components or difficult views are removed.

These results support investigating update strength and preserving separate
false-positive diagnostics. They do not demonstrate a new structural method.
The masks are predominantly pseudo-labels, the two empty cases are too few for
a population claim, and geographic data QA remains incomplete.

## Actions started from this diagnosis

The [prospective revision specification](../../docs/CONSERVATIVE_ADAPTATION.md)
fixes learning rate **0.00003**, with fresh 500-step A1/A2/A3 adapters, seed 17,
and otherwise matched settings. A2 isolates building-region supervision; A3
adds boundary supervision. This is one diagnosed revision, not a rate sweep.
The serial job is separate from every preserved original run. Its results are
pending and cannot be described as an improvement.

Eighteen additional target-only mask proposals were selected without model
scores, across all ten available development geographic groups, using distinct
tiles/panoramas and excluding accepted scenes. Sixteen have scored building
pixels under the proposal labels and two have empty scored building masks.
Their exact hashes, dimensions and binary mask values were verified. **Zero of
these 18 masks is human-approved.** The separate editor batch can be reviewed
while training uses the unchanged original three-mask index.

Double-click **Open Additional Review.cmd** in the project root and choose a
view from 1 to 18. Paint the visible building mask and separately review the
validity mask. Save a version and report which view is ready. The existing
user preference automatically scores newly added building pixels, while manual
validity edits remain possible. Full pairing/footprint QA is a separate task.

Verification: 86 Python tests passed, two optional CUDA tests skipped. The
actual GPU completed all eight new generation/evaluation stages. Fresh-run
configuration checks enforce equal budgets/settings across A1/A2/A3, and the
new report rejects the previous learning rate instead of mislabelling its result.
