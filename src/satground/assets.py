from __future__ import annotations

import subprocess
from pathlib import Path

from .common import (ROOT, CODE_REV, MODEL_REV, MODEL_REPO, DATA_REPO,
                     check_vendor, load_json, save_json, sha256, utc_now)


def bootstrap(model=True):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    vendor = ROOT / "external" / "Sat3DGen"
    if not vendor.exists():
        subprocess.run(["git", "clone", "https://github.com/qianmingduowan/Sat3DGen.git", str(vendor)], check=True)
        subprocess.run(["git", "-C", str(vendor), "checkout", "--detach", CODE_REV], check=True)
    check_vendor(vendor)
    api = HfApi()
    previous = load_json(ROOT / 'artifacts' / 'asset-lock.json') if (ROOT / 'artifacts' / 'asset-lock.json').exists() else {}
    data_rev = previous.get('dataset_metadata_revision') or api.dataset_info(DATA_REPO).sha
    assets = []
    metadata = ROOT / "data" / "metadata"
    for name in ["train__corrected_all_3city_remove_building.txt", "test_remove_building.txt", "README.md"]:
        path = hf_hub_download(DATA_REPO, name, repo_type="dataset", revision=data_rev, local_dir=metadata)
        assets.append({"file": Path(path).relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    if model:
        model_dir = ROOT / "artifacts" / "Sat3DGen"
        snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=model_dir,
                          allow_patterns=["config.json", "diffusion_pytorch_model.safetensors", "README.md"], max_workers=2)
        for name in ["config.json", "diffusion_pytorch_model.safetensors"]:
            assets.append({"file": (model_dir / name).relative_to(ROOT).as_posix(), "sha256": sha256(model_dir / name)})
    else:
        assets.extend(row for row in previous.get('files', []) if row['file'].replace('\\', '/').startswith('artifacts/Sat3DGen/'))
    result = {"created_utc": utc_now(), "code_revision": CODE_REV, "model_revision": MODEL_REV,
              "dataset_metadata_revision": data_rev, "files": assets,
              "rgb_downloaded": False, "test_images_opened": False}
    save_json(ROOT / "artifacts" / "asset-lock.json", result)
    return result
