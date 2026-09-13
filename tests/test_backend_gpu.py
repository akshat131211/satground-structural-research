"""Opt-in integration check against the unmodified official rendering path."""
import os

import numpy as np
import pytest
import torch

from satground.backend import Sat3DBackend
from satground.camera import camera_matrices
from satground.common import seed_everything


@pytest.mark.skipif(os.getenv('SATGROUND_GPU_TESTS') != '1', reason='Opt-in CUDA/checkpoint integration test')
def test_chunked_renderer_matches_native_before_sr():
    backend = Sat3DBackend(size=64, chunk_rows=4)
    backend.set_style(np.ones(270) / 90)
    # Remove importance resampling only for this numerical-equivalence check.
    # Otherwise splitting random-number requests changes Monte Carlo sample ordering.
    backend.model.num_importance = 0
    seed_everything(17)
    raw = torch.randn(1, 64, 320, 320, device='cuda') * .1
    camera = dict(h_offset=12, w_offset=-20, yaw=45, pitch=10, fov=90)
    c2w, _, k = camera_matrices(**camera, size=64)
    with torch.no_grad():
        seed_everything(101)
        ours = backend.render(raw, camera)
        seed_everything(101)
        native = backend.model.from_3D_to_results(
            [raw[:, :32].unsqueeze(1), raw[:, 32:].mean(2)],
            c2w=backend.model.c2w_prepare(torch.from_numpy(c2w)[None].cuda()),
            w_sky=backend.w_sky, sky_feature_2D=backend.sky, syn_pano=False, syn_per=True,
            intrinsics=torch.from_numpy(k)[None].cuda()).per_output.sr_image
    torch.testing.assert_close(ours, native, atol=3e-5, rtol=3e-5)
    assert ours.shape == (1, 3, 64, 64)
