"""VIGOR/Sat3DGen camera conventions, independent of CUDA and target images."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

from .common import ResearchError


def camera_matrices(h_offset, w_offset, yaw=0.0, pitch=0.0, fov=90.0, size=128, sr_factor=2):
    if not (10 <= fov <= 150) or abs(pitch) > 60:
        raise ResearchError("Unsupported field of view or pitch.")
    if abs(h_offset) > 320 or abs(w_offset) > 320:
        raise ResearchError("VIGOR camera position is outside the tile.")
    cv_to_xyz = Rotation.from_euler("X", -90, degrees=True)
    rot = (cv_to_xyz.inv() * Rotation.from_euler("YXZ", [0, pitch, yaw], degrees=True)).inv().as_matrix()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [w_offset / 320, -h_offset / 320, -0.85]
    focal = size * 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    k = np.array([[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1]], dtype=np.float32)
    low_k = k.copy()
    low_k[:2] /= sr_factor  # Preserve the pinned model's training convention.
    return c2w, k, low_k


def crop_panorama(image, yaw=0.0, pitch=0.0, fov=90.0, size=128, nearest=False):
    """Reproduce the release's perspective crop; handles the panorama wrap seam."""
    import cv2
    _, k, _ = camera_matrices(0, 0, yaw, pitch, fov, size)
    y, x = np.mgrid[0:size, 0:size]
    rays = np.stack([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
    # Pinned Equirectangular projection uses yaw around y and pitch around rotated x.
    r1 = cv2.Rodrigues(np.array([0, 1, 0], np.float32) * np.deg2rad(yaw))[0]
    r2 = cv2.Rodrigues((r1 @ np.array([1, 0, 0], np.float32)) * np.deg2rad(-pitch))[0]
    rays = rays @ (r2 @ r1).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(rays[..., 1], -1, 1))
    a = np.asarray(image)
    u = (lon / (2 * np.pi) + 0.5) * (a.shape[1] - 1)
    v = (lat / np.pi + 0.5) * (a.shape[0] - 1)
    return cv2.remap(a, u.astype(np.float32), v.astype(np.float32),
                     cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def satellite_tensor(path):
    import torch
    a = np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    a = (a - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
