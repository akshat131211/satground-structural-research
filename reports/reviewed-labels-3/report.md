# Reviewed-label diagnostic: 3 views

These are annotation-review diagnostics on fixed model outputs. Changes from earlier scores reflect changed target labels and validity regions. The small, previously inspected review set does not establish model improvement or complete the main data QA.

GJ adapts density and appearance jointly; GD adapts density only. Lower boundary error and higher building IoU are preferable. Each view uses the same accepted masks for both models.

| Review view | Scored pixels | GJ boundary | GD boundary | GJ IoU | GD IoU |
|---|---:|---:|---:|---:|---:|
| 1 | 41.3% | 0.082974 | 0.082898 | 0.806923 | 0.827901 |
| 2 | 90.3% | 0.061869 | 0.062389 | 0.585436 | 0.588302 |
| 3 | 70.6% | 0.206382 | 0.201834 | 0.074994 | 0.073540 |

Each rerun retains all 24 preselected views: 3 reviewed targets and 21 pseudo-labelled targets. The mixed aggregate is descriptive and is not a human-verified benchmark.

Checks confirm unchanged predictions, view sets, segmenter revisions, all image-quality metrics, and structural metrics for every unreviewed view. Model-generated building masks still come from the evaluation segmenter. Human review corrects target labels, not predicted masks.

Masks are reviewed edits of machine proposals, with provenance retained for each view. Different validity coverage limits comparisons across views. No server gate has passed.

See [aggregate evidence](summary.json), [the initial one-view report](../reviewed-label-probe/report.md), and [mask review instructions](../../docs/MASK_REVIEW_GUIDE.md).
