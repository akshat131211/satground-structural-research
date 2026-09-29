# Saturated valid-surface revision after the fixed-band failure

This separate revision follows the measured failure of `valid_surface_excess_v1`.
Its combined loss decreases as a false-positive building leaves radius-8 support.
The failed version, fixed cases, protocols, masks and zero-update results remain
unchanged. This is a concrete correction of that failure, not a radius or weight
sweep. VIGOR, Sat3DGen and the structural-supervision question remain unchanged.

## Definition fixed before measurement

Use the same unchanged target y, valid domain D, opposite-label four-neighbour
interface sources and eight-neighbour valid paths without corner cutting.
For components containing an observed interface, let d be shortest-path distance
to that interface and set w=(1+min(d,8))/9 everywhere in that component. Thus
distance weights saturate at one rather than disappear. Components without an
observed interface receive zero surface weight; no interface is invented through
unknown pixels or crop borders. Region BCE remains responsible there.

Keep the same two-class mean of weighted |p-y|, class-mass floor 32, probability
clamp 1e-6 and coefficient 0.5. Keep radius 8 as the saturation distance, not an
exclusive support band. No values are chosen from generated-image scores.

The fixed 52 synthetic cases must reproduce their historical inputs and losses.
Require the original near-shift checks, correctly directed finite interior
gradients, ignored-region invariance and correct geometry preferred over missing,
extra, inverted and collapsed predictions. Additionally require combined loss
to be nondecreasing over the complete fixed hard-translation family, including
40 to 44 pixels. Test validity barriers, corner cutting and boundary-free domains.
Any harmful ordering or nonfinite result blocks training; do not tune to rescue
the criterion. This screen cannot certify all possible geometries.

## Conditional bounded trial

After the current 24 historical and eight v1 zero-update checks complete, verify
their reports and preserve them. If this revision passes the controlled tests,
measure one separate eight-view A0 zero-update feasibility pass with the same
seeds 101 through 108. Require at least six supported views with finite, nonzero
adapter gradients. Feasibility alone does not establish an improvement.

Only after these gates, freeze one matched fresh A2/revised-A3 pair: exactly 200
steps each, learning rate 0.00003, training seed 17, unchanged eight fitting views,
architecture, camera, RGB/perceptual/region losses and training-derived style.
This is the single conditional training pair already planned after diagnosis;
the failed v1 candidate receives no optimizer updates.

Before launch, freeze all scripts, inputs, checkpoint handling and final-only
evaluation. Keep the original full boundary distance and the separate
nonperimeter distance as co-required outcomes: at least 10% improvement against
both A2 and A0; at least four of eight views jointly improving both distances
against A2; IoU decline at most one percentage point and relative LPIPS increase
at most 5% against both references. Preserve failed attempts and compatible
checkpoint recovery. Do not substitute a favorable metric or checkpoint.

No longer budget, extra seed, sweep, server allocation or dataset change is
included. Seattle, audit, calibration and the 21 reviewed development views
remain outside this fitting phase. User masks remain exact and QA limitations
remain recorded. These are exploratory training-example measurements, not a
server gate, novelty or generalization claim.
