# Validity-aware surface loss candidate

This is one candidate for the original structural-supervision question, pending
the controlled and reviewed-gradient diagnosis. Implementation alone does not
select the method or authorize a successful-result claim. Existing experiments,
accepted masks and core loss code remain unchanged.

Distance-based boundary supervision is established. Kervadec et al. express
contour distances through a signed-distance regional integral
([MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html)); Karimi and
Salcudean investigate distance-transform Hausdorff surrogates
([paper](https://arxiv.org/abs/1904.10030)). The candidate below is an experimental
adaptation for partially valid masks, not either paper's exact formula or a
novelty claim.

## Fixed candidate

Let y be the unchanged binary building mask, p the frozen training segmenter's
building probability, and D the accepted validity eroded with the original 3x3
operator, with the outermost image perimeter explicitly removed. Interface
sources are endpoints of opposite-label four-neighbour pairs entirely in D.
Compute shortest-path distances from those sources through D using eight-neighbour
costs 1 and sqrt(2), with no diagonal movement through an invalid corner.
Invalid and unreachable pixels have no distance evidence.

Use a fixed radius of 8 pixels at 256 resolution. Within that radius set
w=(1+d)/9, and set w=0 elsewhere. Compute the mean weighted |p-y| error separately
for building and nonbuilding pixels, dividing each sum by the greater of its
class's weight mass and 32, then average the two class contributions. Use the
historical probability clamp of 1e-6 and a fixed surface coefficient of 0.5.
The radius, mass floor and coefficient are not tuned on fitting results.

This gives oriented foreground/background probability gradients without comparing
unsigned predicted edge magnitudes. The fixed mass floor bounds amplification
from very small fragments. Region BCE stays unchanged and is responsible for
distant false positives and components without a known interface. A component
with no admissible interface receives zero surface supervision, not an invented
boundary or distance. Report such components explicitly.

This policy can discard more contour support than the historical center-only
validity test, because interface neighbours must also belong to D. Audit that
loss of support and both class weight masses on all eight views before any
candidate training. More confident bookkeeping is not proof of more accurate
labels; approved masks, including unresolved QA concerns, remain preserved.

## Required evidence before training

- Correct synthetic geometry must have lower loss than displaced, missing or
  extra structures under the complete region-plus-surface objective. Inspect
  1, 2, 4 and 8 pixel displacements; clipped-band saturation must be reported.
- Gradients must restore foreground and remove false positives, while remaining
  finite for empty/full, tiny-fragment and crop-touching targets.
- Arbitrary probability changes confined to ignored pixels must not change the
  surface loss or its scored gradients. Invalid strips and corners must block
  distance paths; image borders must not create artificial interface sources.
- Actual reviewed-view support and one fixed, zero-update candidate-gradient
  pass at A0 must be measured after the historical 24-check probe. At least six
  of eight views must have an admissible interface and a finite, nonzero candidate
  adapter gradient. This is a predeclared feasibility rail, not evidence of label
  accuracy or improvement. Keep unsupported views in the cohort and report zero
  supervision. Nonfinite gradients, unresolved implementation discrepancies or
  harmful controlled-case behavior block training. Record gradient conflicts;
  their sign or magnitude alone cannot prove harm, efficacy or justify tuning
  the fixed coefficient.

If these checks support the correction, freeze a separate fresh A2/candidate-A3
pair of exactly 200 steps each, learning rate 0.00003 and seed 17, with matched
sampling and unchanged architecture, RGB/perceptual/region weights and style.
Keep the eight reviewed training examples, final-step scoring and frozen A0.

Declare both the historical full boundary distance and the already defined
nonperimeter distance as required outcomes for that future trial: at least 10%
relative improvement against both A2 and A0, at least four of eight views jointly
improving on both distances against A2, IoU decline at most one percentage point,
and relative LPIPS increase at most 5% against both references. Report all views,
both distances, missing strata and visual failures even if the criterion fails.
Do not retrospectively replace the completed experiment's metric or decision.

These are exploratory fitting gates, not a server gate or generalization claim.
No sweep, additional seed, longer budget, new dataset or server allocation is
included. Independent annotation, pairing and geographic QA remain incomplete.
