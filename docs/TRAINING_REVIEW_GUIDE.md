# Review 12 training scenes before another learning experiment

Double-click **Open Training Review.cmd** in the project folder. The launcher
reuses the running local server or restores saved progress after a restart.
All 12 images are in one page with an image selector and Save & Next navigation.

For each view:

1. Inspect the satellite context. White dot = released camera position; yellow
   arrow/wedge = the ground image whose mask you will edit. The other three
   real ground directions provide context. Open the full-size context or original
   satellite image for detail. Directions are based on released metadata; they
   do not independently establish heading, metric scale, visibility or identity.
2. Choose whether recognizable layout supports the pairing, it remains uncertain,
   or it appears mismatched. Assess whether the main visible buildings are inside
   the satellite crop. Similar colors alone are insufficient evidence. Uncertain
   is a useful answer; do not invent a correspondence.
3. Enter your reviewer name and a short evidence/uncertainty note. Scene notes
   save locally. Review the building and validity masks using the existing drawing
   controls. Building = visible building surfaces; validity = confidently
   assessable pixels. Exclude hidden walls, uncertain foliage, cars and people
   from validity. Bridges/supports are not buildings. An empty building mask is
   allowed. New building additions follow the existing automatic-scoring preference;
   you can still explicitly exclude uncertain pixels in validity afterward.
4. Select whether you checked both masks or they remain uncertain. Click
   **Save & Next**. Uncertain or mismatched cases can be submitted and retained;
   completion does not make them eligible for training. Next image alone keeps
   drafts without marking a view finished. Fill or cancel an unfinished polygon
   before switching images.
5. On the last view use **Save & Finish**. Tell Codex once **all training reviews
   are ready**. No message is needed between images. Your exact masks and scene
   responses will then be verified before any acceptance index is created.

Scene decisions and masks are versioned separately. Finishing binds the exact
mask version to the exact scene-review revision. Later edits invalidate completion;
finish the view again after editing. Previously saved versions are preserved.
Raw notes, reviewer identities, images and masks remain local and out of Git.

## Fixed selection and separation

The packet uses 12 views from the existing 100 training views: four geographic
groups per city, distinct satellite tiles and panoramas. Within each city, groups
and then sample IDs are ordered by SHA256 with the fixed `training-review12-v1:`
prefix. Selection uses metadata only, without consulting pixels, mask quality or
model scores. No rejected example is silently replaced. The usable fraction and
reasons for uncertainty must be reported after review.

Initial masks are exact binary exports of the existing cached training labels.
No segmenter or generator was rerun. Human changes stay in separate versioned
files and do not modify the training cache or the completed contour experiment.
The 21 human-reviewed development masks remain evaluation-only. Seattle, audit
and calibration imagery are not opened. This is not a new validation benchmark,
and reviewing these 12 examples does not complete dataset-wide QA.

## Gate before any additional training

No training is launched by this tool. First verify actual human decisions,
pairing evidence and exact mask/target hashes. Separate uncertain cases from
supported cases without deleting them from the review record. If too few views
are usable, report that failure rather than filling the set using model scores.

A later, separately specified 200-step A2/A3 fitting check may use suitable
reviewed training examples to assess whether the adapter can correct known
outlines. It would test learnability on those examples, not generalization or
paper readiness. No training configuration, checkpoint or budget is created here.

## Reproduction

Use fresh paths; existing packets and reports are never overwritten:

```powershell
.venv/python.exe scripts/prepare_training_review.py --output data/review/training12-new --report runs/training12-new-status.json
.venv/python.exe scripts/open_training_review.py --packet data/review/training12-new --output data/review/training12-new-edits --open-browser
```

The selection, context and UI additions live under `scripts/`. Core model code,
the established drawing implementation and completed runner inputs are unchanged.
The public preparation record is [TRAINING_REVIEW_STATUS.json](../reports/TRAINING_REVIEW_STATUS.json).
