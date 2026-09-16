# Review before the main experiment

The preliminary runs establish that real-image adaptation executes on this laptop. Three guided human pairing observations and three accepted practice building/validity mask pairs are recorded separately. The larger data and annotation review remains incomplete. Main training still requires alignment and spatial-separation evidence; the user-authorized [exploratory mode](EXPLORATORY_TRAINING.md) allows longer diagnostic training meanwhile. See the [mask review guide](MASK_REVIEW_GUIDE.md).

## Prepared packet

Run `python scripts/prepare_review_packet.py` from the project root. The current local output is `data/review/development200/`. Existing output directories are rejected to protect annotations.

The packet contains 200 exact perspective crops, their overhead images, 20 contact sheets, a pairing checklist, a pending manual-mask index, and manifest hashes. It samples 100 training and 100 validation views by cycling cities and geographic groups, with unique panoramas and tiles. Selection uses no prediction scores or content filters. Seattle, calibration, and audit images are excluded.

Open `START_HERE.md` in the packet. Review the full PNGs when drawing masks; the contact sheets are navigation aids. The building and valid masks must match the exact 256 x 256 target crop. The packet deliberately creates no placeholder masks that could be mistaken for annotations.

For each view:

- Check whether roads and visible building placement agree with the overhead image and recorded camera offsets/direction. Record uncertainty when roof-to-facade correspondence is ambiguous.
- Flag suspected temporal mismatch only when there is evidence; images alone may not establish acquisition dates.
- Mark the visible building extent. Do not infer hidden walls behind trees or vehicles as observed ground truth.
- Define valid static pixels, excluding moving objects, uncertain pixels, and occlusion where structure cannot be assessed. Keep genuine no-building scenes.
- Record the actual reviewer, completion time, and reasons for any corrections or exclusions.

Edit a copy of `manual-index.pending.json`; change `reviewed` only after the masks are completed and checked. A partial evaluation index must contain only completed entries. The evaluator rejects entries without reviewer/date information or masks of the correct dimensions. Manual annotation should be based on the target image before looking at generated alternatives. Evaluate training and validation subsets separately when reporting generalization.

Reviewed entries now also require the exact `target_image` path and SHA-256
fields `target_image_sha256`, `building_mask_sha256`, and `valid_mask_sha256`.
Masks must be single-channel PNGs with values 0 and 255. The evaluator rejects
changed masks and targets, rather than silently reusing an older approval.
The guided export command in the mask guide prepares these fields only after
both mask components have explicit review decisions. Previously generated
pending templates remain unapproved and must be completed with current hashes.

## Separate spatial and duplicate checks

The local `data/qa/near-duplicates-review.json` records two dHash candidates. This is a data-integrity review, not permission to inspect audit prediction quality. Determine whether the original image pairs show the same scene; use file/coordinate evidence and a designated data reviewer. Record decisions and rationale. If real leakage is found, revise geographic splits and regenerate the affected manifests, validation records, style, labels, and experimental outputs under new run names.

The 100 m footprint-radius bound is conservative relative to the coordinate/offset diagnostic, but that diagnostic is not independent calibration. Verify image scale against release metadata or reliable geographic references, and check that footprints cannot cross excluded split boundaries. Do not mark footprint verification complete based only on visual plausibility or a derived pixel ratio.

After those checks, copy `configs/data-qa-template.json` to `data/qa/review.json`, add the SHA-256 of the reviewed `reports/data-readiness.json`, fill the actual reviewer/date and supporting notes, and set only evidence-supported fields. A checksum can be obtained with:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath reports/data-readiness.json
```

Use lowercase hexadecimal checksums. Review records must follow the current manifests; changing data or splits invalidates old approvals. Default reviewed mode requires completed footprint, pairing and duplicate review beyond step 100. Explicit exploratory mode permits longer diagnostics with pending QA recorded in their provenance; it does not qualify their results for a server decision.

## What comes next

Once QA is complete, generate the main training-only illumination code and pseudo-labels, then run matched A1/A2/A3 configurations. Keep final audit imagery sealed until configuration selection is frozen. The manual development labels do not automatically certify final audit results.

All imagery, masks, per-location records, and review identities remain in ignored local directories. GitHub receives code and aggregate status only.
