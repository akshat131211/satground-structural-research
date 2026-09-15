# Review exact building and validity masks

Pairing observations answer whether two views might share a building. They do
not certify the building's pixel outline. This workflow keeps those decisions
separate and preserves every annotation proposal and response.

The current local packet starts at `data/review/mask-first3/`. It uses the first
three existing learning-check validation views in their fixed order. These are
teaching/annotation-feasibility examples, not a new benchmark. Model predictions
were previously shown for these views, so the review is explicitly **not blind**.

## What each card means

- **Building card:** red overlay and white mask pixels mean visible building.
  Include visible facade, windows, entrances, and attached architectural parts.
  Exclude cars, people, foliage, road, and hidden surfaces. A single building
  class does not separate individual building identities.
- **Validity card:** green overlay and white mask pixels mean suitable for
  structural scoring. Judge this independently. Exclude uncertain and dynamic
  regions; do not select validity based on where generated predictions look good.
  A valid pixel can be building or non-building.

Inspect the original target first. Report **needs correction**, **uncertain**,
or explicitly accept the displayed component. Mention missing/extra areas when
possible. Approval of building pixels alone does not approve validity pixels.
Small facade fragments behind leaves may need careful corrections or an explicit
uncertain region; confidently inventing the hidden wall is not valid annotation.

When a reviewer describes a gap as uncertain, preserve that uncertainty. A
requested addition is a proposal, not an accepted label. If it adds building
pixels beyond the accepted building mask, update both proposed components and
review the changed components; validity alone must not turn a missing building
label into scored non-building. The earlier accepted version remains archived.

The initial proposals use the pinned B1 evaluation segmenter at input size 256.
After rejection of view 1, a separate proposal was generated with input size
1024. That operates on an upsampled 256-pixel target; it does not add measured
image detail. The higher-resolution proposal still contained errors. Explicit
assistant polygon corrections were subsequently recorded in a new version.
All these remain proposals until the exact displayed masks receive human review.

Neither this change in annotation proposal nor a corrected ground-truth mask is
an improvement to Sat3DGen. Past evaluation results retain their original labels.
Any evaluation using reviewed masks must be a new, separately named output,
applying the same labels and validity regions to every compared model.

## Files and provenance

`packet.json` identifies exact target, building, validity, and card checksums.
`decisions/*.pending.json` contains no approvals. Actual user responses are saved
in separate `.review.json` records, with the raw wording and actual review time.
Revisions use new folders; they do not inherit acceptance from an older card.
Pixel corrections include their coordinates, source version, and reason.

An explicit component acceptance remains applicable if the exact target, mask,
and displayed card for that component are all verified byte-for-byte unchanged
while the other component is revised. In that case retain the original human
answer/time and prior decision hash, documenting the equality check. A changed
component always requires new review; general pairing feedback never substitutes
for mask acceptance.

Once both components of one sample have explicit acceptance, export an index:

```powershell
& .\.venv\python.exe scripts/finalize_mask_review.py --packet data/review/mask-first3-v3 --sample-id SAMPLE_ID --decision data/review/mask-first3-v3/decisions/SAMPLE_ID.review.json --output data/review/mask-first3-v3/approved-index.json
```

The exporter checks the packet/card hashes, both actual decisions, reviewer/time,
binary mask values, dimensions, target/mask hashes, and nonempty validity. It
records machine-proposal and correction provenance rather than describing the
masks as independently drawn from scratch. It never marks overall dataset QA
complete. A rejected or pending decision cannot be exported as approved.

The evaluator rechecks mask/target hashes and exact target pixels before using
any reviewed labels. Required index fields include `target_image`,
`target_image_sha256`, `building_mask_sha256`, and `valid_mask_sha256`, alongside
the existing paths, reviewer, and date. Old pending templates are not completed
annotations; fill these fields only from the actual reviewed files. The
`finalize_mask_review.py` command constructs these fields for the guided workflow.

A partially reviewed evaluation mixes reviewed and pseudo targets. The output
states that fact and records each sample's annotation method. Report reviewed
results separately before making claims about human-verified structural accuracy.

## Reproducible preparation and revisions

```powershell
& .\.venv\python.exe scripts/prepare_mask_review.py --output data/review/mask-review-new
& .\.venv\python.exe scripts/revise_mask_proposal.py --packet data/review/mask-first3 --sample-id SAMPLE_ID --input-size 1024 --output data/review/mask-revision-new
& .\.venv\python.exe scripts/patch_mask_proposal.py --packet data/review/mask-first3-v2 --plan data/review/mask-first3-v2/proposed-corrections.json --output data/review/mask-pixel-revision-new
```

Keep imagery, proposals, raw human responses, correction plans, and exact-location
records local. GitHub receives the code, documentation, and aggregate status.
The [main review requirements](REVIEW_GUIDE.md) and [building-identity review](IDENTITY_REVIEW_GUIDE.md)
remain separate tasks; a few mask approvals do not complete them.

## Inspect small gaps with regional comparisons

Prepare a local JSON file listing named rectangles in exact target pixels:

```json
{"regions": [{"name": "Upper tree", "box_xyxy": [76, 34, 152, 66]}]}
```

Then compare two immutable proposal versions:

```powershell
& .\.venv\python.exe scripts/preview_mask_regions.py --packet data/review/mask-first3-v7 --previous data/review/mask-first3-v6 --sample-id SAMPLE_ID --regions data/review/regions.json --output data/review/tree-detail-new.png
```

The left column is the exact target; the next columns show building and validity
proposals. Yellow marks additions and magenta marks removals. Nearest-neighbor
enlargement reveals the actual scoring pixels; it adds no measured detail.
The sidecar records input/output hashes and change counts. This command writes
only a visual aid and its provenance, never masks, decisions, or an approved
index. Use the complete mask cards alongside regional views for review.
