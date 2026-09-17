# Anchored contour correction: fixed laptop experiment

The prior boundary mask left only 48 positive pixels across 100 training views.
This experiment tests a correction within the existing single-overhead-image
VIGOR + Sat3DGen structural-supervision question. It does not test longer training.

## Frozen target rule

Keep all cached building, region-validity, class and confidence arrays unchanged.
No teacher inference or human-mask modification is needed. Construct a separate
boundary-support mask using these training pseudo-labels only:

1. Retain every position scored by the original eroded validity mask.
2. Add a raw 3-by-3 morphological building-boundary pixel only when a confident
   building pixel and a confident static nonbuilding pixel both occur within a
   four-pixel Chebyshev radius (9-by-9 window).
3. New positions require their 3-by-3 class neighborhood to contain no vegetation
   or dynamic classes. Both anchors require the existing confidence threshold 0.7.
4. Keep the hard morphological boundary target and mean-over-supported-pixels
   normalization. Do not rebalance positives, change weights or tune radius.

Four pixels is a fixed local tolerance chosen before auditing the new targets,
not a measured geospatial tolerance. This rule supports a transition through
some uncertain pixels; it cannot guarantee the teacher classified it correctly.
Keep ambiguous gaps wider than that local support excluded. Empty targets do
not acquire positive boundaries. Human development masks are never training input.

Before training, require positive support in at least half of the 92 nonempty
training views and at least 1,000 positive positions in total. These are fixed
computational-feasibility checks, not ground-truth or quality thresholds. Inspect
a deterministic training-side packet locally; assistant inspection remains
distinct from independent human verification. Retain failure outputs if this
rule fails instead of silently sweeping its parameters.

## Matched comparison

Fresh A2 and corrected A3 adapters each run 500 optimizer steps, seed 17,
learning rate 0.00003, accumulation 8, size 256, width 16, and the same fixed
training illumination, views, renderer, region weight 0.5 and LPIPS weight 0.1.
Both use the same extended label files. A2 boundary weight is zero; A3 uses 0.5.
Only the extra boundary term distinguishes their training configurations.
Use a separate three-step A3 startup/resume check; never use it as a selected
checkpoint or include its updates in either fresh comparison model.

Save every 100 steps but evaluate the predeclared final step 500. No best-step
selection, extra seed, larger budget, weight/radius sweep or server allocation.
On failure preserve partial outputs and resume only matching identities. Pin
the source, protocol, configurations, labels and accepted index before launch.
Run one GPU subprocess at a time.

## Evaluation and limitations

Evaluate the same 24 development views with the same 21 reviewed mask pairs
and three remaining pseudo-label targets. Use the frozen B1 evaluation segmenter.
Primary comparison: corrected A3 versus fresh A2. Also compare preserved A0,
ordinary A1 and legacy A3 descriptively using the same evaluation-label version.
Report geographic paired bootstrap intervals, reviewed and nonempty-target
boundaries/IoU, empty-target false-positive area and LPIPS. Inspect local paired
galleries. Check that the fresh A2 control reproduces its historical counterpart;
if it does not, diagnose differences before attributing changes to contours.

Target support is not a model improvement. This is one exploratory seed on
previously inspected development views; data QA is incomplete. No server-gate,
novelty or audited-generalization conclusion is eligible. Seattle, audit and
calibration imagery remain sealed. Independent training-contour review is still
needed before claiming that the new targets are verified annotations.

Historical experiments and their definitions are immutable. Their original
source hashes belong to their original Git revisions. The completed gradient
diagnosis was published at 166fb577980be6cc909d189f1c99e14b475068bf; use that
revision when reproducing its exact source-identity checks.
