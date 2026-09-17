# Development checkpoint diagnosis

Specified before running the trajectory evaluation on 17 September 2026.
This continues the existing structural-supervision research question.

- Re-evaluate A1 and A3 at saved steps 100, 200, 300 and 400 using the exact
  current 24 validation views, three reviewed target-mask pairs, B1 evaluation
  segmenter, camera poses and training-only illumination. Reuse the verified
  step-500 results and frozen A0 reference.
- Preserve the original final-step comparison. This retrospective development
  diagnosis does not choose a replacement primary checkpoint or open audit,
  calibration or Seattle images.
- Verify generation/checkpoint/image hashes, common label coverage, source
  revisions and rendering settings before aggregating.
- Retain all-view geographic means. Additionally separate targets with at least
  one scored building pixel from targets with zero scored building pixels, using
  the same target-derived partition at every checkpoint. Report counts and
  label provenance. These are scoring-label categories, not certified scene facts.
- For empty targets, report predicted building area / scored area, plus counts
  of nonempty predictions. For nonempty targets, report the original boundary
  error and IoU, and the count of completely missing building predictions.
  Do not remove tiny components or change the existing empty-boundary penalty.
- Show all three previously reviewed views separately and preserve their order.
  Inspect a fixed-view trajectory gallery. Sharpness is not geometric accuracy.
- Summarize 100-step training-log windows and adapter output-layer norms.
  Window averages contain sampled training views and are not a paired fixed-set
  generalization test. Do not infer overfitting solely from those averages.

The next training revision depends on these diagnostics. A smaller learning rate
is a candidate if update strength explains degradation; it is not yet validated.
Any new A1/A2/A3 comparison must share samples, budgets and optimization settings,
with changes specified before training. No server gate follows from this cohort.

Prepare 18 additional target-only development mask proposals using geographic
round-robin selection, excluding the three accepted views and avoiding duplicate
panoramas/tiles. Selection must not depend on model scores. Human review remains
pending until the user actually reviews the masks; pairing/footprint QA is separate.
