# Review the candidate multiple-view building set

The local folder `data/review/identity24/` contains 24 overhead tiles, two
distinct panorama positions per tile, and four existing perspective directions
per position: 192 real ground views. There are eight tiles per city and 24
different training geographic groups. Selection used metadata only, without
examining model scores, masks, or image content.

The inventory found 1,747 of the 2,000 training tiles with multiple camera
positions and complete directional crops. This establishes availability of
sparse multiple views, not correct building correspondence or dense trajectories.
The released camera convention uses -90 degrees for west; the selection code
recognizes it as equivalent to 270 without changing any source manifest row.

## Purpose of this packet

The density-only control can change geometry but still learns from a segmenter
applied to RGB. The proposed later method needs stronger evidence: which visible
building in a ground image corresponds to which supported overhead region, and
which building remains the same across camera positions.

The packet has no instance masks, assigned building identities, or certified
correspondences. `identity-review.pending.json` has empty fields and
`training_eligible` is false. Preparing or inspecting a contact sheet does not
complete human review. The eventual validated annotation format and identity
trainer remain future work.

## Review a small number of tiles first

1. Open the corresponding image in `sheets/`. Each row is a different real camera
   position; columns show the same four directions. Use the full-resolution
   `overhead/` and `targets/` images for close inspection.
2. Identify visible building boundaries that can be distinguished from trees,
   vehicles, walls/fences, and neighboring buildings. A continuous semantic
   building region may contain several separate buildings.
3. Record cross-camera identity only where evidence is clear. An identical facade
   color, a nearby map coordinate, or a matching model prediction is insufficient
   by itself. Use visible corners, structural layout, and camera relationships.
4. Link an overhead region only if its relationship to the visible building is
   supported. Label it as an observed roof region unless a true footprint has
   separate evidence. Do not infer metric height from the mask.
5. Record unresolved cases as unknown: outside-crop buildings, occlusion,
   off-nadir roof displacement, uncertain camera orientation, suspected temporal
   changes, or buildings that cannot be separated. Keep those views in the packet.
6. Work on a copy of the pending index. Add an actual reviewer and date only for
   completed work. Keep raw images, notes, and future masks local.

Start with a few tiles to learn whether useful correspondences can be reviewed
reliably. Expand to the remaining candidates only if that is feasible. This
candidate set is not selected for containing easy buildings; some tiles may
provide no trustworthy correspondence at all. Report the usable fraction and
retain rejected/unknown cases with reasons.

## Separate requirements

This training-only packet supports a possible new supervision method. It does
not replace the existing [200-view development review](REVIEW_GUIDE.md), its
evaluation masks, or the main data-quality checks. Neither review permits opening
Seattle or changing the frozen audit protocol. Assistant visual observations
remain separate from human-certified annotations.

Regenerate on a fresh directory using:

```powershell
& .\.venv\python.exe scripts/prepare_multiview_review.py --output data/review/identity24-new
```

The aggregate preparation and verification record is
[MULTIVIEW_REVIEW_STATUS.json](../reports/MULTIVIEW_REVIEW_STATUS.json).
