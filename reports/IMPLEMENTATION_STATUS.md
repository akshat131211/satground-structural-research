# Implementation status — updated 15 September 2026

The laptop research infrastructure now runs on real VIGOR inputs. The user supplied the archive folder on 15 September 2026. The selected pilot has been imported and decoded, and a 24-view A0 baseline has been generated and evaluated. Preliminary adaptation checks are underway. No controlled three-seed improvement, calibrated reliability result, or server gate has been established.

## Real-data milestone on 15 September 2026

- Six training-city RGB archives passed full gzip EOF/CRC checks and received SHA-256 records.
- All 3,000 satellite tiles and 5,622 panoramas required by the existing pilot were found and decoded: zero missing, invalid, or exact cross-split duplicates.
- Two dHash near-duplicate candidate pairs are pending review; these are not confirmed duplicates.
- All 3,747 training panorama sky masks were retrieved from the pinned supplement. Original downloads remain in `dataset/`; the selected files are in `data/vigor/`. Neither directory is tracked by Git.
- Seattle archive member names were inspected to identify the city; its imagery was not extracted, decoded, or used.
- A fixed preliminary subset has 100 training views and 24 validation views from 10 validation geographic groups. Selection preceded predictions and metrics.
- The 24-view frozen baseline finished at 256 x 256: 1,949.8 MiB peak PyTorch allocation and 36.2 seconds for the generation loop, including cache misses but excluding model startup. These timings are specific to this small run.
- Baseline group-weighted pseudo-label metrics: building IoU 0.5316, normalized boundary error 0.2040, LPIPS 0.5659, SSIM 0.1566. The independent B1 evaluation segmenter differs from B0 training supervision; zero masks have human verification. This small adaptation holdout may have been seen by the pretrained model.
- Illumination preprocessing now matches the pinned 512 x 128 RGB/mask resize before averaging training histograms; the reference equivalence test passes.
- Protocol, archive integrity/path handling, and illumination checks: 28 tests passed.

See [machine-readable ingest and baseline evidence](DATA_INGEST_STATUS.json). Full manifests, archive/member hashes, predictions, and per-view results remain local under ignored data/run directories.

## Verified locally on 13 September 2026

- Pinned Sat3DGen source and safetensors checkpoint downloaded and loaded on native Windows.
- NVIDIA RTX 4050 Laptop GPU; PyTorch 2.6.0+cu124 reports CUDA available.
- Zero-initialized adapter reproduces the frozen baseline exactly (maximum RGB difference 0).
- Full RGB/LPIPS/building/boundary objective passed synthetic 128- and 256-pixel tests.
- 256-pixel test: 10 optimizer steps with accumulation 8, then a resumed step; finite losses and gradients, frozen base parameters.
- Peak allocated VRAM: 1904.0 MiB; peak PyTorch reserved memory: 2220.0 MiB. These exclude the desktop/driver's separate allocations.
- Trainable adapter parameters: 4,448; frozen model parameters: 381,022,223.
- Median warm synthetic optimizer-step time: 3.78 seconds. This is not a real-dataset or server throughput estimate; frozen features were already cached in memory.
- Protocol and CUDA reference-renderer checks: `24 passed in 9.95s`.
- Dependency consistency check passed. Real-data training rejects missing original images before creating a run.
- Public metadata split preparation produced 22,488 fixed-view records from 3,000 selected overhead tiles: 2,000 train, 250 validation, 250 calibration, 500 audit. RGB validation reports 8,622 missing files. Seattle imagery was not opened.

The controlled native-renderer equality check disables importance resampling to avoid differences in random-number request ordering; normal profiles retain the checkpoint's 48 coarse and 48 importance samples. Full-image super-resolution is applied after ray chunks are stitched.

## What remains

1. Complete and inspect the preliminary 100-view learning and checkpoint/resume checks.
2. Review real image scale/footprints, the two near-duplicate candidates, camera alignment and dates; annotate development building boundaries.
3. Run matched main A1/A2/A3 experiments with seeds 17/29/43 and validation-only selection after the required review.
4. Freeze the audit, complete independent/manual evaluation, and assess structural and reliability gates.
5. Start server, external-geometry, and 3D phases only if the preceding gates pass.

Current server decision: **insufficient evidence**. The 13 September GPU profiles below are synthetic diagnostics. The new baseline uses real images, but the main controlled adaptation study and human review remain incomplete. Optional KID has not been run on a real dataset. Human annotations have not been fabricated.

## Implementation corrections during verification

The panorama pitch sign was aligned with the pinned release and checked against reference crops. Temporary test files were moved into the workspace to avoid a pre-existing Windows shared-temp permission problem. NVIDIA's pinned SegFormer release uses legacy weight files, loaded with PyTorch's restricted weights-only loader; no safetensors file was assumed. The unmodified upstream code and checksum-pinned assets remain separate from this project's source.

See [the runbook](../docs/RUNBOOK.md), [data access](../docs/DATA_ACCESS.md), [verification records](verification.json), and [the main resource profile](profile256-accum8.json).
