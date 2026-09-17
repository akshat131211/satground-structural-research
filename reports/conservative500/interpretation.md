# Completed matched revision: structural improvement is not established

All three 500-step runs finished on the RTX 4050 at learning rate 0.00003,
seed 17. The fixed 100 training views, 24 development views, three reviewed
mask pairs and 21 pseudo-labelled targets are unchanged across models. All
1,500 logged steps and final adapter/optimizer tensors are finite. The saved
reports reproduce exactly; all 96 prediction files across A0/A1/A2/A3 pass
hash checks. The frozen A0 predictions also match the original baseline.

## Primary comparison and ablations

Lower boundary error and LPIPS are better; higher IoU is better. Means give
each geographic group equal weight.

| Model | Supervision | Boundary error | Building IoU | LPIPS |
|---|---|---:|---:|---:|
| A0 | Frozen checkpoint | 0.195863 | 0.533314 | 0.565909 |
| A1 | RGB and perceptual | 0.195721 | 0.452819 | 0.582288 |
| A2 | A1 plus building region | 0.194261 | 0.539867 | 0.567256 |
| A3 | A2 plus boundary | 0.191681 | 0.540232 | 0.567910 |

- **A3 versus A1:** boundary error falls **2.06%**, with paired geographic
  bootstrap 95% CI **[-0.004615, 0.015448]** for absolute improvement. The
  interval spans zero. IoU rises 8.741 percentage points and LPIPS falls 2.47%.
  This misses the preselected 10% boundary target even before QA requirements.
- **A2 versus A1:** boundary error falls 0.75%, CI [-0.005656, 0.008794]. IoU
  rises 8.705 points and LPIPS falls 2.58%. Region supervision accounts for
  almost all of the aggregate IoU/perceptual recovery from A1.
- **A3 versus A2:** the boundary term adds a 1.33% boundary reduction,
  CI [0.000026, 0.006890]; IoU rises only 0.037 points and LPIPS worsens 0.12%.
  The narrowly positive interval is a secondary development result, without
  multiplicity adjustment or confirmation across training seeds.
- **A3 versus A0:** boundary error falls 2.14%, CI [0.000697, 0.008674]; IoU
  rises 0.692 points and LPIPS worsens 0.35%. This post-run reference check
  cannot replace the prespecified A3/A1 test.

Intervals resample the ten development geographic groups, with 2,000 bootstrap
draws. They do not include uncertainty from training seeds, labels or pairing.

## Human-reviewed views contradict an unqualified structural claim

| Reviewed view | A0 boundary | A1 boundary | A2 boundary | A3 boundary |
|---|---:|---:|---:|---:|
| 1 | 0.080931 | 0.070316 | 0.095711 | 0.097738 |
| 2 | 0.061055 | 0.054296 | 0.062226 | 0.062253 |
| 3 | 0.214713 | 0.091441 | 0.215773 | 0.215406 |

A3 is worse than A1 and A0 on all three reviewed boundary scores. A3 is also
worse than A2 on two of the three. The aggregate gain therefore is not
corroborated by these reviewed targets. These are three previously inspected
development views, not a representative blind benchmark.

Assistant visual inspection used all three targets in their original review
order, beside A0/A1/A2/A3. A1 is visibly blurred and contains ghosted structures;
A2/A3 resemble A0 much more closely. In view 1 the facade/entrance layout still
differs from the real target. In view 2 the foreground building mass and height
remain wrong. View 3 still misses the real tall buildings; its A3 IoU is 0.0783.
A1 has a much smaller boundary score there despite IoU of only 0.0639 and visible
failure. Boundary distance alone must not be interpreted as building recovery.
These observations are not additional human approvals or verified instance counts.

## Target-label strata and annotation limitations

| Model | Boundary: 22 nonempty targets | IoU: 22 nonempty targets | False-positive area: 2 empty targets (%) |
|---|---:|---:|---:|
| A0 | 0.217625 | 0.481460 | 1.6070 |
| A1 | 0.217468 | 0.392022 | 1.0753 |
| A2 | 0.215845 | 0.488741 | 1.6286 |
| A3 | 0.212979 | 0.489147 | 1.6060 |

Each stratum weights its own geographic groups equally, so stratum means do
not sum to the overall mean. All four models have the same empty-target
boundary penalty of 0.5. Thus the earlier experiment's discontinuous 13-pixel
penalty change is not the source of this revision's A3/A1 difference. However,
A3 has more labelled false-positive area than A1 on these two targets.

These categories describe the **fixed scoring labels**, not certified scene
contents. The separate human review already found buildings in one originally
empty pseudo-labelled target and a bridge incorrectly marked as a building in
another view. See [label-review findings](../checkpoint-diagnosis/label-review-notes.md).
Seven of 18 additional mask pairs are accepted; eleven remain pending. All
accepted indices and mask/target hashes were checked. None was inserted into
this experiment. Any later expanded-label evaluation must be separately
versioned and apply the same corrected masks to every compared model.

## Decision and reproducibility

**The structural hypothesis is not supported by this diagnosed pilot. Server
scaling remains unjustified.** This is an inconclusive/negative development
result for this adapter and budget, not proof that structural supervision can
never work. No further sweep, longer budget or extra seeds were launched.

Pairing, footprint/scale, date and near-duplicate QA remain incomplete. The
pretrained model may have seen these development locations. Seattle, the
sealed audit and calibration imagery were not opened. Reliability, metric
geometry, publication novelty and generalization are unestablished.

Training-loop times were 2,257.6/2,270.7/2,179.4 seconds for A1/A2/A3, with
597.5/668.1/668.1 MiB peak PyTorch allocated memory using existing feature
caches. These exclude initialization and desktop/driver allocations and are
not controlled speed comparisons. The serial job finished in about 114 minutes.

The next evidence step is to finish the remaining human review and evaluate
the saved predictions under one clearly versioned expanded label set before
considering any further experiment. Original results remain preserved.

Verification: **89 Python tests passed, two optional CUDA tests skipped**;
JavaScript editor syntax and behavior checks passed. The completed training
runs themselves exercised the GPU. The local gallery stays under ignored
`runs/conservative500-completion-audit-v2/`; no imagery or individual identifiers
are published.

To reproduce the completion checks, choose fresh output paths:

```powershell
.venv\python.exe scripts/audit_conservative500.py --output runs/fresh-completion.json --work runs/fresh-completion
```

[Unchanged primary report](primary/report.md), [ablations](summary.json),
[completion audit](completion-audit.json), and [current status](../CONSERVATIVE_RUN_STATUS.json).
The automatic report's `interpretation_pending` field records its state when
training ended; this separate document supplies the subsequent interpretation.
