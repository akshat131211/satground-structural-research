# Bounded fitting diagnosis on reviewed training examples

Protocol prepared on 28 September 2026, before new optimizer updates. The
completed contour experiment did not establish structural improvement. This
diagnosis asks whether the same shared adapter can fit the available reviewed
training outlines, and whether adding the boundary term helps beyond A2.

## Cohort and annotation boundary

All 12 completed submissions remain immutable. View 2 is marked mismatched and
view 8 has uncertain coverage; both are excluded. On 28 September the user
explicitly selected the eight-view option, excluding views 1 and 10 as well.
The fitting cohort is views 3, 4, 5, 6, 7, 9, 11 and 12, with 1,043 positive
target contour pixels after the existing 3-by-3 validity erosion. Record the
actual disposition locally and verify the exact mask versions before launch.
This resolves the subset choice, not independent data QA. No replacements are
selected and the excluded masks remain preserved.

A separate fitting bundle records actual inclusion/exclusion decisions, checks
completed submission hashes and exact target crops, and copies the binary PNG
values into float32 NPZ arrays. It does not paint, smooth, fill, erode or expand
the stored masks. The objective still erodes validity at evaluation of its
boundary term. These are human-reviewed/corrected machine proposals, with
incomplete independent geographic QA; they must never be called pseudo-labels
merely to pass the original training loader.

The 21 reviewed development masks are never fitting inputs. No development,
Seattle, audit or calibration images are generated or scored in this test.
Excluded submissions and raw human decisions remain private and preserved.

## Fixed experiment

`configs/reviewed-fitting200/protocol.json` fixes one fresh A2 and A3 at exactly
200 optimizer steps each, seed 17, learning rate 0.00003. Batch size is one,
with eight microbatches per optimizer step. Render settings, frozen backbone,
4,448-parameter shared adapter, initialization and sampled views are matched.
Periodic checkpoints at 50, 100 and 150 steps are for recovery, not selection;
only the final 200-step checkpoint is scored. A0 is a frozen reference.

The fixed illumination histogram remains the previously computed mean from
the 100-view training universe. Configuration `train_manifest` identifies this
universe for style provenance; `fitting_manifest` identifies the only examples
used for gradients and fitting scores. Keeping these roles explicit avoids
falsely attributing the existing histogram to the smaller reviewed subset.
No target-derived illumination is used at prediction time.

Both models optimize RGB L1 + 0.1 LPIPS + 0.5 building-region BCE. A3 adds 0.5
times the absolute difference between differentiable 3-by-3 morphological
boundaries, averaged over validity whose full 3-by-3 neighborhood is scored.
Both terms use the exact submitted building/validity values. The old anchored
pseudo-label support extension is not applied to these masks. The frozen B0
segmenter supplies differentiable probabilities; its weights remain frozen.

## Predeclared interpretation

Generate A0, A2 and A3 on every included **training** view at fixed render seed
101. Use the existing independent B1 evaluator, exact reviewed targets and
validity, normalized symmetric boundary distance, building IoU and LPIPS.
Report equal geographic-group means, per-view changes, building-containing and
empty-target strata, and false-positive area separately. If a stratum is empty,
report it as absent, not as a zero error. Retain the existing evaluator's
explicit empty-boundary convention and inspect local galleries for its effects.

A limited positive fitting result requires all of the following at step 200:

- A3 boundary error at least 10% below A2 and at least 10% below A0.
- At least half of included views have strictly lower boundary error than A2.
- IoU decreases by no more than 0.01 and LPIPS increases by no more than 5%
  relative to each of A2 and A0.
- All updates, checkpoints, predictions, masks and aggregate identities verify.

A zero baseline makes a relative change undefined and cannot satisfy a positive
relative-gain criterion. Report all absolute measurements regardless of this
decision. A positive result means **fitting response on this selected training
subset only**. Failure does not rule out every possible structural method.
No confidence interval here is evidence of held-out generalization. One seed,
small selected cohort, semantic-model dependence, partial contour support and
incomplete data QA prohibit a server-gate, novelty or publication-readiness claim.

## Execution and recovery

The new scripts operate separately from historical `src/satground` training
and completed experiment files. Bundle preparation is CPU-only and requires an
explicit local disposition record; it cannot infer a human choice from a
generic resume command. Generated configurations and mask indices stay ignored.

The trainer refuses missing or altered bundles, incorrect budgets, sealed
splits, overlapping development identities and changed source/input hashes.
A serial runner holds a project GPU lock, records child commands and logs,
and does not automatically remove a stale lock. Inspect live processes and the
actual failure before compatible checkpoint recovery. Retain partial inference
or evaluation outputs for diagnosis instead of silently overwriting them.
No extra optimizer updates, seeds, tuning or extended budget follow from this
protocol. Stop after the bounded diagnosis and its verified report.
