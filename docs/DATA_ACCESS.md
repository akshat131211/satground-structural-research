# Data access and provenance

## Current dependency

On 14 September 2026, the user confirmed that they obtained VIGOR access and started downloading the dataset. They explicitly chose to continue the original VIGOR + Sat3DGen approach. Download completion, the local data path and file validation are pending the user's update. Do not treat the dataset as locally ready yet.

The previously downloaded public supplement contains metadata and optional derived resources, not the original paired satellite and street RGB photographs. The [alternative-data review](DATA_ALTERNATIVES.md) is retained as background; no replacement data adapter or alternative acquisition is planned.

For future reproduction, the [official VIGOR form](https://docs.google.com/forms/d/e/1FAIpQLScTSD6AFZgre3yLNbl7OqBcjrJF0-u2cpwgubqSQnyVzPKJzA/viewform?usp=sf_link) and [maintainers' dataset instructions](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md) document the access route. The repository states academic use and no redistribution. No new access request is needed for the user's current download.

Once the download finishes, identify its local folder and extract as needed. `data/vigor` is one possible location; an existing folder can be passed through `--data-root` without moving the dataset. Keep the original folder and filename structure:

```text
data/vigor/
  Chicago/satellite/...
  Chicago/panorama/...
  Chicago/pano_sky_mask/...
  NewYork/satellite/...
  NewYork/panorama/...
  NewYork/pano_sky_mask/...
  SanFrancisco/satellite/...
  SanFrancisco/panorama/...
  SanFrancisco/pano_sky_mask/...
```

The training sky masks come from the [Sat3DGen supplement](https://huggingface.co/datasets/qian43/VIGOR_SAT3DGEN_add_skymask_DSM_satdepth). Only the three training-city sky-mask archives are needed for the fixed illumination code. Satellite depth and Seattle DSM archives are not needed for this adapter pilot. Bootstrap does not download them automatically.

## Prepared local pilot

The public training metadata has 78,188 pairs from Chicago, New York, and San Francisco. The separate test metadata lists Seattle. The pilot currently samples 2,000 training tiles, 250 validation tiles, 250 calibration tiles, and 500 audit tiles, retaining associated panoramas and four prescribed headings. It produces 22,488 view records and requires 8,622 distinct original RGB files.

Splits use 1 km local geographic blocks, a provisional 100 m boundary guard, and exclusion of panoramas spanning blocks. This is metadata preparation, not completed geographic QA. Confirm that image footprints fit the guard bound using actual imagery scale metadata before the main experiment. Verify orientation, temporal/pose alignment, corrupt files, exact duplicate hashes, and near-duplicate review candidates after RGB access. A perceptual-hash match is only a review candidate.

Pilot holdouts may have been seen by the pretrained model. They assess adaptation generalization, not an untouched foundation-model test. Keep Seattle images sealed until server-stage configuration selection is complete.

## Other datasets

[Sat2GroundScape-Perspective](https://huggingface.co/datasets/GDAOSU/Sat2GroundScape-Perspective) remains a later comparison option. Its rendered conditions and panorama identifiers are not a complete interchangeable single-overhead/ground RGB dataset. Complete geometry and authorized target access must be verified first.

The [current GroundScape access audit](GROUNDSCAPE_ACCESS_AUDIT.md) checks both Perspective and Panorama downloads and lists the concrete requirements for using the supplied paper's dataset. An author-request draft is available there; no message has been sent.

Do not substitute bulk Google Street View acquisition for missing research data. The [Google Maps Platform terms](https://cloud.google.com/maps-platform/terms), section 3.2.3, restrict model training and evaluation uses of Maps content; a dataset-card license alone does not establish permission for that separate acquisition route. CVUSA, VIGOR++, and HoliCity are later candidates with their own access and compatibility checks.
