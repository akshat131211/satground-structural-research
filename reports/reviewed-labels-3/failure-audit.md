# Three-view annotation and preprocessing diagnostic

The three guided mask pairs are complete. These were previously inspected
development views, with human edits of machine proposals. They do not replace
the larger review or certify building identity.

View 3 shows a visible target-image mismatch: the tall buildings in the real
image are largely missing from both saved predictions. Its building IoU is
7.50% for GJ and 7.35% for GD. The [three-view report](report.md) retains every
reviewed view and gives its scoring coverage; all 24 original evaluation views
remain in each rerun. A local gallery shows the three targets and both models
in the original review order, without selecting favourable outputs.

The [source/camera audit](camera-source-audit.json) checks these exact three
views. Their pairing fields match the pinned training metadata, their reviewed
targets match crops from the original panoramas, and their crop pixels and
rotation matrices match the released implementation with maximum difference
zero. The audit checks implementation consistency. It does **not** independently
verify true camera heading, satellite footprint scale, capture dates, or whether
the overhead image and visible facade correspond in the real world.

The visible error therefore warrants a pairing/camera-versus-model investigation.
It does not by itself establish a novel method or justify server training.
Independent correspondence evidence, the outstanding duplicate/footprint checks,
and the larger development review remain necessary. Keep the difficult example
in the cohort while investigating it.

Reproduce the software check with fresh output paths:

```powershell
& .\.venv\python.exe scripts/audit_reviewed_cameras.py --index data/review/manual-scored-first3/approved-index.json --manifest data/manifests/learning100/validation.jsonl --output data/review/new-camera-audit.json --aggregate reports/new-camera-audit.json
```

Exact image identifiers, camera fields, masks, and human responses remain local.
The published aggregate contains counts, errors, evidence hashes, and explicit
unverified fields.
