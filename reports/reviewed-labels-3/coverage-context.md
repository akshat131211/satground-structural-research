# Input coverage diagnosis for the difficult third view

A local context packet places the released camera and four viewing directions
over each reviewed satellite image beside four real panorama crops. These are
assistant inspection aids, not new human correspondence annotations.

In view 3, the camera marker falls on the road between a tree-lined building and
an open construction area. The yaw-180 crop shows a nearby building behind
trees; yaw 0 looks across an open lot toward distant taller buildings. Under
the release's orientation convention, the provided satellite crop shows the
nearby road, trees and construction area, but not those distant tall buildings.
This supports **limited input coverage as a plausible contributor**, alongside
possible capture-time changes. Independent heading, metric scale, capture dates
and exact building identities remain unverified.

The center ray reaches the top of the 640-pixel input after 228 image-coordinate
pixels. The pinned model has padding of 0.125 per side and position scale 0.8,
giving an equivalent distance of 308 pixels to its extended scene boundary.
These are coordinate distances, not metres, object depths or measured visible
coverage. The padding is learned context, not additional observed satellite RGB.
A ray crossing a footprint boundary does not determine which ground-view pixels
correspond to objects outside it.

This is not an unclaimed research gap: [Sat3DGen's method](https://arxiv.org/html/2605.14984#S3.SS1)
already uses spatial tokens to address footprint mismatch. Adding padding alone
would overlap. A more specific future question is whether a frozen estimate of
input-supported versus extrapolated structure can diagnose or reduce building
errors without degrading supported geometry. It requires comparison with the
existing padding, uncertainty baselines and independently checked correspondence.
This remains a hypothesis, not a validated improvement.

No example was removed or camera optimized against its target. The
[longer exploratory experiment](../../docs/EXPLORATORY_TRAINING.md) keeps its
original inputs and losses to test the effect of more optimization.

Rebuild locally with a fresh output directory:

```powershell
& .\.venv\python.exe scripts/prepare_camera_context.py --output data/review/new-camera-context
```

The packet records source, target and script hashes. Tests check cardinal
directions and ray-edge calculations. Each reviewed yaw-zero target is checked
against its accepted mask index. Images and identifiers stay off GitHub.
