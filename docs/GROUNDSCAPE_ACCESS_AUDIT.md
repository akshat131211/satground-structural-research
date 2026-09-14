# Can we use the supplied GroundScape paper's dataset?

**Inactive reference:** the user subsequently confirmed VIGOR access and an ongoing download on 14 September 2026, choosing to continue with VIGOR. The access investigation below is retained as background; no GroundScape migration or author outreach is planned.

Reviewed 14 September 2026. **Yes, conditionally.** Its public release is useful, but it is not a complete replacement for the overhead/real-ground pairs required by our current training experiment. No academic login was required for the inspected metadata and viewer endpoints.

## Public release evidence

| Release | Rows | Download size, decimal GB | Revision |
|---|---:|---:|---|
| [Perspective](https://huggingface.co/datasets/GDAOSU/Sat2GroundScape-Perspective) | 99,825 | 4.74 | `90991bbd090a9f516d2aa6ed14b854f3d33ad988` |
| [Panorama](https://huggingface.co/datasets/GDAOSU/Sat2GroundScape-Panorama) | 25,503 | 12.54 | `8cfd15c893e5533968c9dbc1ca68ded040ee696e` |

Both inspected dataset APIs report `gated: false`. The live viewer schemas and dataset cards agree:

- Perspective fields: `condition, lat, lon, heading, elevation, panoid, theta, phi, fov`.
- Panorama fields: `condition, lat, lon, heading, elevation, panoid`.
- `condition` is satellite texture rendered from a ground camera: 256 x 256 for perspective and 1024 x 512 for panorama.
- Real target RGB is not an image column. The cards instead instruct separate panorama retrieval by ID through a Street View package and a Google Maps API key.
- Original overhead image tiles, meshes and depth maps are not fields in these inspected releases; no such separate files appeared in their repository trees.

The [linked GitHub repository](https://github.com/GDAOSU/sat2groundscape), default branch `sat2groundscape`, currently contains the project website and media. It did not provide model-training code or checkpoints in the inspected tree. No claim is made about private author resources.

## Why the distinction matters

The [paper, sections 3.2 and 3.4](https://arxiv.org/html/2504.15786v1), describes reconstructing and texturing a mesh from multiple satellite views, then rendering it into ground cameras before generative refinement. Its source satellite imagery comes from the 2019 Data Fusion Contest around Jacksonville. A provided rendered condition therefore already incorporates geometry from several images.

Two valid experiments would be different:

1. **Rendered satellite condition to realistic street image:** fits the published conditions, but needs real target photographs and a suitable refinement-model implementation. It does not demonstrate prediction from one raw overhead RGB image.
2. **One overhead RGB image to street image:** retains our intended task. It additionally needs georeferenced source imagery, pairing to the ground cameras and verified scale/orientation. Multi-view-rendered conditions could be studied separately as extra-information supervision, with that distinction reported.

Neither setting can currently run the proposed supervised real-data comparison using these archives alone.

## Target retrieval is a separate dependency

The cards' Apache 2.0 label does not independently authorize acquiring new Google Street View imagery for training. [Google Maps Platform terms, section 3.2.3(c)(vii)](https://cloud.google.com/maps-platform/terms), restrict training, testing, validation and fine-tuning uses. A normal paid API key does not itself settle this issue. Ask the authors for the applicable research-access arrangement or an authorized complete subset; do not infer that their historical acquisition and our prospective acquisition have the same permissions.

## Concrete next step

Request a small complete research subset, preferably 100-500 locations for initial feasibility, along with camera conventions, satellite-tile mapping and usable target access. Also request model weights/code and split definitions if paper reproduction is intended. Start with camera/pair alignment and laptop profiling once those assets are available. Preserve the existing VIGOR pipeline and the [open-data fallback](DATA_ALTERNATIVES.md) while this is unresolved.

An [author-request draft](GROUNDSCAPE_AUTHOR_REQUEST.md) is prepared. It has **not been sent**. This review inspected cards, file inventories and viewer schemas, not the full image archives; it did not implement a GroundScape loader or run training.
