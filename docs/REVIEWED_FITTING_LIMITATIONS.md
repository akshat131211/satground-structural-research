# Boundary support audit during the eight-view fitting diagnosis

This CPU audit was defined after frozen A0 evaluation, while the predeclared
200-step A2/A3 pair was active on 28 September 2026. It is a post-hoc diagnostic,
not a new primary metric, checkpoint-selection rule or success criterion. The
frozen trainer, evaluator, user masks and protocol are unchanged.

The eight masks provide 1,043 positive training contour pixels across 418,975
pixels whose 3-by-3 validity neighborhood is scored: **0.249% positive support**.
The training absolute boundary loss averages over this full validity area.
Therefore most of its scored locations have a zero target gradient. This fact
does not establish that its gradient is ineffective, but limits interpretations
of an aggregate improvement as evidence of learning visible building outlines.

The primary evaluator has 2,263 target inner-edge pixels. **1,676 (74.1%)** lie
on the outermost image rows or columns. Its inner-edge definition erodes masks
against outside-image background, creating closing edges for buildings cut by
the crop. The training morphological boundary does not impose that outside-image
background. Both definitions have valid uses, but they measure different edge
sets in these particular crops.

All eight A0 targets and predictions have nonempty primary boundaries; the
baseline's errors do not arise from the maximum empty-boundary penalty. Support
is uneven. One fitting view has just three positive training contour pixels and
three evaluated target edges away from the image perimeter. Per-view scores and
local galleries are needed alongside the primary aggregate result.

`scripts/diagnose_fitting_boundaries.py` reproduces the exact primary mask scores
before computing a separate distance that excludes the outermost image rows and
columns from both boundary sets. It retains the explicit both-empty/one-empty
conventions, reports their counts, and never changes validity or stored masks.
This alternate distance is descriptive and must not replace the predeclared
criterion or be used to select a favorable checkpoint.

The baseline audit is [aggregate evidence](../reports/reviewed-fitting200/boundary-support-audit.json).
After the final pair completes, run the same diagnostic on all three final model
outputs into a fresh file. Interpret its changes alongside IoU, LPIPS, all eight
view results and the paired galleries. Fitting scores remain training-example
evidence with incomplete independent data QA; no server gate or generalization
claim follows from them.
