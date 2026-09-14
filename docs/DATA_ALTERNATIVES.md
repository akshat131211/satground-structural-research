# Alternatives if VIGOR access is unavailable

Access review: 14 September 2026. This is a dataset recommendation and feasibility record, not a completed dataset migration or training result.

The user's supplied GroundScape dataset was also checked in both release formats. See the [GroundScape access audit](GROUNDSCAPE_ACCESS_AUDIT.md) for the missing real targets and raw-overhead distinction. The IGN route below is a fallback if complete usable benchmark pairs cannot be obtained.

## Recommended real-data pilot: IGN Panoramax + BD ORTHO

Pair publicly accessible ground panoramas from the **IGN-hosted Panoramax instance** with licensed IGN BD ORTHO overhead images covering the same positions. This removes dependence on approval from the VIGOR authors. It creates a new paired research subset rather than reproducing an existing benchmark.

The [IGN Panoramax catalog](https://panoramax.ign.fr/api) links its Etalab 2.0 licence. An unauthenticated query to `/api/search?limit=2` returned HTTP 200 on the review date, and streaming the beginning of one returned standard-definition image produced HTTP 200, `image/jpeg`, and a JPEG signature. Only metadata and the initial image bytes were inspected; image content, building visibility, and alignment have not been reviewed.

The two inspected records included:

| Field | Record 1 | Record 2 |
|---|---|---|
| Public image ID | `e020681d-f0b0-4af1-a671-4cdfffdfc722` | `2f8ab4bb-0aad-4249-9240-61dd057274b0` |
| Capture date | 2026-09-12 | 2026-09-12 |
| Licence | `etalab-2.0` | `etalab-2.0` |
| Camera | QooCam 3 Ultra, 360 degrees | QooCam 3 Ultra, 360 degrees |
| Original dimensions | 13,888 x 6,944 | 13,888 x 6,944 |
| Reported azimuth | 242 degrees | 239 degrees |
| Reported horizontal accuracy | 3.6 m | 3.3 m |

These are provider metadata, not independently measured pose accuracy. The inspected EXIF identifies equirectangular panoramas. Access does not establish that heading, pitch, roll, camera height, or GPS position are sufficiently accurate for boundary evaluation. The recent upload dates are not a reason to select these exact records for training; an older panorama may match the available overhead capture date better.

IGN identifies BD ORTHO as high-resolution aerial imagery in its [current open-data overview](https://www.ign.fr/institut/des-donnees-et-logiciels-ouverts-au-service-de-la-nation). Its [product metadata](https://geoservices.ign.fr/sites/default/files/2022-10/IGNF_BDORTHOr_2-0.html) specifies the Etalab open licence. Verify the licence, resolution, footprint and capture date of the actual tiles selected, retaining attribution and provenance. Matching overhead tiles have not yet been acquired or inspected.

**Input distinction:** BD ORTHO consists of aircraft photographs rectified into map coordinates. It supports the overhead-to-ground objective, but results must be described as aerial-to-ground; a satellite-specific transfer claim requires a separate satellite-image evaluation.

### Small feasibility gate before adaptation

1. Discover candidate sequences in several urban areas, initially targeting about 100 building-containing pairs. Select public images with explicit licences, usable orientation and suitable camera metadata.
2. Match overhead coverage and dates, check building changes, verify map scale, and inspect several synthetic and real camera landmarks. Reject uncertain camera alignment instead of inventing calibration values.
3. Convert a few panoramas into prescribed perspective crops. Confirm target-free prediction: fixed illumination comes only from the eventual training partition.
4. Group shared panoramas, overlapping overhead footprints, neighboring locations and whole sequences together when assigning geographic holdouts. Expand across independent areas before claiming generalization; 100 nearby frames are not 100 independent scenes.
5. Run the frozen checkpoint first. Proceed to a 1,000-2,000-location adaptation pilot only if pair alignment and baseline behavior support the structural experiment. Choose final counts after measuring independent coverage.

The model, structural adapter, losses and evaluation utilities are reusable. The existing preparation and camera path are VIGOR-specific: implement a separate manifest/data adapter, camera normalization and scale conversion, new split configuration, and training-only illumination preparation. Do not relabel French locations as VIGOR cities or reuse its normalized camera constants without validation. Keep VIGOR and its sealed Seattle test separate.

## Other routes

| Route | Access and usefulness | Limitation |
|---|---|---|
| Procedurally generated buildings | Render paired overhead/street views with known cameras, masks, depth and building instances from scenes we create. | Useful for controlled structural experiments; synthetic gains do not establish real-world improvement. This dataset has not been built. |
| KartaView + open orthophotos | [Official terms](https://kartaview.org/terms) license street images under CC BY-SA 4.0; [API documentation](https://kartaview.org/doc/faq) describes public access. | Must construct pairs and validate image access, calibration, geography and dates. Not a ready paired benchmark. |
| CVUSA / CVACT | Established paired alternatives. | [CVUSA](https://mvrl.cse.wustl.edu/datasets/cvusa/) requires an access request; [CVACT](https://github.com/Liumouliu/OriCNN) requires contacting the authors and restricts use to research. Neither guarantees a way around the current access issue. |
| KITTI-360 + open orthophotos | Calibrated street imagery and useful annotations. | The [registration page](https://www.cvlibs.net/datasets/kitti-360/user_register.php) requests an institutional email. Overhead pairing is additional work. |
| Authors' Sat2Density++ demo samples | [Official repository](https://github.com/qianmingduowan/Sat2Densitypp) includes a few CVACT/CVUSA example pairs. | Useful for inference checks subject to source terms, not an independent training/evaluation corpus. |

## Public downloads that still need clarification

The [2025 field-robot cross-view dataset](https://springernature.figshare.com/articles/dataset/A_multi-modality_ground-to-air_cross-view_pose_estimation_dataset_for_field_robots/28528868) has public archives, a CC BY 4.0 listing, ground images, camera calibration and position labels. However, its actual [`data_description.txt`](https://ndownloader.figshare.com/files/53276423) identifies the overhead images as Google Earth. The listing alone does not resolve third-party overhead training rights. Do not call it an unrestricted substitute without resolving this provenance issue. Its LiDAR/camera-fused BEV arrays are also not overhead RGB scene inputs.

[DReSS](https://github.com/SummerpanKing/DReSS) links public Hugging Face archives, but its README says that only ground panorama IDs and aerial coordinates are supplied to comply with source policies. The Hugging Face image archives and licence labels do not resolve that discrepancy. Treat this as unverified, not as a clean drop-in replacement.

No full alternative dataset was downloaded, no replacement loader was implemented, and no real-data training was run for this access review. VIGOR remains an optional benchmark route; the recommended next feasibility study is the IGN pair construction above.
