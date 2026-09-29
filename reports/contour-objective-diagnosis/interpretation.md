# Completed contour-objective diagnosis — 29 September 2026

The completed eight-view fitting experiment is preserved: its declared fitting
criterion was not met. This separate diagnosis measures one possible weakness
before deciding whether another bounded experiment is justified.

The fixed 52-case, 13-derivative synthetic report reproduces byte-for-byte to a
fresh local path. At 40 and 44 pixels of displacement the current raw boundary
loss is identical (0.0296875), while the auxiliary normalized outline distance
rises from 0.135345 to 0.157442. The combined region-plus-boundary loss is also
identical in these two cases. This demonstrates a displacement plateau on these
maps, not invariance to every displacement or failure of the entire model.

A uniform missing-building probability map has boundary loss 0.015625 and zero
boundary probability gradient. The region gradient remains corrective. The
boundary-only missing penalty is lower than the displaced penalty; the displaced
map contains both missing and extra regions. These comparisons do not establish
that a generator will collapse or that its complete objective prefers deletion.

The same missing contour on a larger scored canvas receives a smaller penalty.
Changing confidence can reduce loss without moving the hard outline. Targets
with only crop-perimeter edges provide no internal training contour. The
auxiliary geometric score and probability gradients are diagnostic measures,
not independent image-quality or generalization evidence.

One fixed validity-aware surface candidate is specified separately. Its radius,
class normalization floor and coefficient are declared before actual fitting.
It must pass the same controlled cases, the support audit and eight zero-update
adapter-gradient checks after the historical 24-check diagnosis. Its local
distance band saturates; it does not replace region supervision for distant
false positives or invent labels in ignored regions.

That candidate passes the fixed near-shift checks, but its combined structural
loss falls from 0.469028 to 0.459674 when displacement grows from 40 to 44 pixels.
The geometric error grows in the same comparison. The false-positive building
leaves its local distance band and loses the extra penalty. This is an objective
ordering failure, not harmless saturation. The predeclared harmful-behavior rule
therefore blocks training. No radius, coefficient or screening gate was changed
to rescue the candidate. Zero-update gradient feasibility measurements may
characterize it, but cannot override this controlled failure.

All 24 historical and eight candidate checks completed with finite values,
unchanged adapter tensors and zero optimizer updates. Startup plus compatible
resume stayed within the declared counts. Both aggregate reports and the
completion audit reproduce exactly to fresh local paths. The historical 158
source/input pins and candidate pins remain intact; raw receipts, gradient
payloads, reviewed target support and provenance were rechecked independently.

| Adapter-gradient statistic | A0 | Final A2 | Final A3 |
|---|---:|---:|---:|
| Median weighted boundary/A2 objective norm | 3.726% | 4.346% | 4.157% |
| Median zero-contour/positive-contour norm | 41.788 | 39.939 | 37.142 |
| Boundary opposing A2 objective direction | 6/8 | 4/8 | 4/8 |

Here the A2 objective is RGB plus perceptual plus region at the same model state.
Zero target contour includes building interiors and nonbuilding regions; it
does not mean an empty target building mask. These are raw gradients before
optimizer moments and clipping, not reconstructed historical parameter updates.
Their norms and cosines alone cannot establish a causal source of poor geometry.

The candidate reaches the adapter on seven views; one has no admissible interface
and remains in the cohort. Its median weighted surface/A2 norm ratio is 0.428,
with a maximum of 4.090. Seven finite nonzero gradients pass the six-view
feasibility rail, but do not clear the harmful controlled-case ordering. The
maximum historical decomposition error is 8.93e-7 in RGB space and 3.85e-6 in
adapter space, within the declared tolerances. Repeated A2 gradients differ
slightly; no bitwise reproducibility claim is made.

This phase is complete with `candidate_training_blocked`. A separately specified
saturation revision retains distant valid-region penalties; it does not alter
this phase's method, counts, criteria or negative decision.

No new optimizer updates or generator improvement are reported here. The eight
views are reviewed training examples with incomplete independent QA. Sealed
sets and the reviewed development views remain outside this phase. No server
gate, novelty or audited-generalization conclusion follows from this diagnosis.

See [the fixed CPU evidence](controlled-cases.json),
[diagnostic protocol](../../docs/CONTOUR_OBJECTIVE_DIAGNOSIS.md),
[candidate specification](../../docs/VALID_SURFACE_CANDIDATE.md) and
[unchanged fitting result](../reviewed-fitting200/interpretation.md).
