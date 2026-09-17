# Completed mask batch: no reliable boundary improvement

The user completed all 18 additional building/validity mask pairs. Together
with the original three pairs, 21 of the same 24 development views now have
human-reviewed labels. Re-scoring the unchanged A0/A1/A2/A3 images gives **0.33%
lower boundary error for A3 versus A1 over all 24 views**, and **0.055% lower
error on the 21 reviewed views**. Both paired confidence intervals include zero.
This diagnostic does not establish the structural-supervision hypothesis or
justify server training. It is not a novelty or audited generalization result.

## What changed

The final batch-completion message authorized export of the exact saved versions.
All eleven newly completed submissions matched their completion marker, current
draft, target hash and binary 256 x 256 mask files. The seven earlier accepted
pairs retain their accepted versions, including the explicitly accepted bridge
correction. A new versioned index combines these 18 pairs with the unchanged
original three. Earlier indices, experiment reports and training outputs remain
preserved.

There was **no new training or image generation**. Checks verified identical
generation metadata, all 96 prediction images, all 96 predicted segmentations,
and all 96 LPIPS/PSNR/SSIM records. All models use identical target and validity
masks. The three previously reviewed rows and three still-pseudo-labelled rows
are unchanged. Target-building or valid-pixel counts changed in 17 of the 18
newly reviewed rows; the other row was accepted without pixel changes.

The large decrease in absolute boundary scores from the earlier roughly 0.19
to roughly 0.09 reflects **changed scoring labels**, not improved models.
These measurements must not be placed alongside the old measurements as a
before/after training gain.

## Matched comparisons

All means weight geographic groups equally. Intervals below are paired 95%
bootstrap intervals for the **absolute normalized boundary-error reduction**,
using 2,000 resamples of the ten geographic groups and bootstrap seed 17.
Positive reduction favors the proposed model; negative reduction means worse
boundaries. These exploratory intervals are not adjusted for multiple comparisons.

| Comparison | Boundary reduction, all 24 | Absolute-reduction 95% CI | IoU change | LPIPS change |
|---|---:|---:|---:|---:|
| A3 versus A1 | 0.334% | [-0.007147, 0.009717] | +3.628 percentage points | -2.469% |
| A2 versus A1 | -1.835% | [-0.007871, 0.004930] | +3.601 percentage points | -2.581% |
| A3 versus A2 | 2.130% | [-0.000031, 0.005244] | +0.027 percentage points | +0.115% |
| A3 versus frozen A0 | 2.106% | [-0.002371, 0.006650] | +0.853 percentage points | +0.353% |

Lower LPIPS is better. Region supervision in A2 accounts for almost all of
A3's IoU/perceptual recovery relative to ordinary adaptation A1. Adding the
boundary term gives a small point estimate over A2, but its interval now includes
zero. The narrowly positive secondary interval under the earlier label version
does not persist under the expanded human labels.

On the **21 reviewed views**, A3/A1 boundary reduction is 0.0546%, with absolute
CI [-0.007871, 0.009576]. A3/A2 reduction is 1.999%, CI [-0.000080, 0.005299];
A3/A0 reduction is 2.871%, CI [-0.001930, 0.007977]. None excludes zero.
A3 improves 10 reviewed boundary scores versus A1, worsens eight and ties three.
The original three reviewed boundary scores remain worse than A1 and A0.

## Building-containing and empty targets

The current labels define **21 nonempty scored building targets and three empty
targets**. This is different from the 21-reviewed-view cohort: that cohort
contains 18 nonempty and three empty targets, while the remaining three
pseudo-labelled targets are nonempty. No view was excluded from evaluation.

| Model | Nonempty-target boundary | Nonempty-target IoU | Empty-target predicted building pixels | Empty-target group-weighted false-positive area |
|---|---:|---:|---:|---:|
| A0 | 0.080176 | 0.429230 | 92 | 0.06359% |
| A1 | 0.077102 | 0.392449 | 55 | 0.03801% |
| A2 | 0.079275 | 0.440234 | 104 | 0.07188% |
| A3 | 0.077074 | 0.440670 | 99 | 0.06843% |

All four models have an empty-target boundary mean of 1/3: one of three empty
targets has a nonempty predicted building mask. Their binary empty-target
penalties therefore tie despite different false-positive areas. A3 has more
false-positive building pixels than A1 on these targets. The three empty targets
contain 134,838 valid pixels in total; group-weighted fractions above are not
computed by simply dividing the pooled pixel counts. All models predict some
building pixels on each nonempty target, which does not certify individual
building completeness.

## Local visual and annotation checks

All 21 reviewed target/model comparisons were inspected locally in fixed order.
A1 remains visibly blurred in several scenes; A2 and A3 often resemble A0.
Wrong facade layouts, incorrect building masses and missing distant or
industrial buildings remain. These assistant observations are diagnostic notes,
not certified instance-error counts or additional human annotations.

Two annotation qualifications remain visible in the record. Additional batch
view 9 may include an elevated transport structure under the building label;
its submitted mask is preserved pending a separate taxonomy review. In batch
view 16, 19 newly added building pixels are excluded by later manual validity
edits. Those exact validity choices are retained. Excluded residual building
pixels in two vegetation-only views also remain as submitted and do not score.
No masks were silently cleaned or rewritten to improve results.

These are human-reviewed machine proposals from the evaluation segmenter,
without an independent annotator or certified instance labels. Review was not
fully blinded to earlier model results. Pairing, geographic-footprint and
duplicate QA are incomplete. The cohort is small, previously inspected, and
uses one training seed; three targets still use pseudo-labels. Seattle, audit
and calibration imagery remain sealed. Reliability calibration has not been
established.

## Reproduction and decision

The new local results are in `runs/conservative500-reviewed21`; private approved
indices and masks stay in `data/review`. [The summary](summary.json) records
hashes, aggregate strata and paired comparisons. Per-review-number measurements
and imagery remain local. The original [conservative experiment](../conservative500/interpretation.md)
and [earlier experiment](../exploratory500/interpretation.md) are unchanged.

From the checkout root, the existing saved evaluation can be checked again
without running inference:

```powershell
& .\.venv\python.exe scripts/report_reviewed21.py --output runs/reviewed21-report-recheck --local-output runs/reviewed21-gallery-recheck
```

For a complete re-score on a machine with the same private data, approved index
and saved model images, use fresh output paths. This command uses the GPU for
evaluation only and should run after checking that no other GPU job is active:

```powershell
& .\.venv\python.exe scripts/evaluate_reviewed21.py --output runs/reviewed21-reproduction
& .\.venv\python.exe scripts/report_reviewed21.py --root runs/reviewed21-reproduction --output runs/reviewed21-reproduced-report --local-output runs/reviewed21-reproduced-gallery
```

The completed run used the same serial evaluation commands before the wrapper
was made reusable; its original status identity and logs are retained. Reporting
checks fail if predictions, labels, cohorts or non-label metrics differ.

**Decision: retain an inconclusive/negative pilot result.** The primary gain is
far below the planned 10% threshold and its interval includes zero. Full data
QA and three-seed confirmation are also absent, so a server gate cannot pass.
No further seed, sweep, longer run or dataset change was started. The experiment
monitor remains paused. Future work first needs an explicit decision about
pairing/coverage and annotation quality, informed by these failures; larger
hardware alone is not justified by this evidence.
