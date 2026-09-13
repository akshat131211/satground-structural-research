from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_REV = "70763aff4d9fc5508e67685c906f84d3cebb34fe"
MODEL_REV = "fa0ed75aff7f42dcd749764787c0184fcab0b12c"
MODEL_REPO = "qian43/Sat3DGen"
DATA_REPO = "qian43/VIGOR_SAT3DGEN_add_skymask_DSM_satdepth"
TRAIN_CITIES = {"Chicago", "NewYork", "SanFrancisco"}
SPLITS = {"train", "validation", "calibration", "audit"}


class ResearchError(RuntimeError):
    """An unmet data/protocol requirement; never silently fabricate a substitute."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_data_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ResearchError(f"Path escapes dataset root: {relative}")
    return path


def require_development(rows, allowed_splits=None):
    if not rows:
        raise ResearchError("Empty manifest.")
    ids = set()
    for row in rows:
        if row["city"] not in TRAIN_CITIES or "Seattle" in str(row):
            raise ResearchError("Seattle is sealed. Pilot commands cannot open test imagery.")
        if row["split"] not in (allowed_splits or SPLITS):
            raise ResearchError(f"Split {row['split']} is not allowed for this command.")
        if row["sample_id"] in ids:
            raise ResearchError("Duplicate sample ID.")
        ids.add(row["sample_id"])


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def provenance():
    import importlib.metadata
    packages = {}
    for name in ["torch", "torchvision", "numpy", "transformers", "diffusers", "scipy", "lpips"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'src' / 'satground').glob('*.py'))}
    return {"created_utc": utc_now(), "python": platform.python_version(), "pipeline_source_hash": object_hash(source_hashes),
            "platform": platform.platform(), "code_revision": CODE_REV,
            "model_revision": MODEL_REV, "packages": packages}


def check_vendor(vendor=None):
    vendor = Path(vendor or ROOT / "external" / "Sat3DGen").resolve()
    if not (vendor / "source" / "generator.py").is_file():
        raise ResearchError("Pinned Sat3DGen checkout is missing. Run bootstrap.")
    actual = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != CODE_REV or dirty.strip():
        raise ResearchError("Sat3DGen checkout differs from the pinned unmodified revision.")
    return vendor
