"""Opt-in renderer intervention, gradient, and checkpoint recomputation checks."""
import os

import numpy as np
import pytest
import torch

from satground.backend import Sat3DBackend
from satground.common import seed_everything
from satground.fields import render_fields


@pytest.mark.skipif(os.getenv('SATGROUND_GPU_TESTS') != '1', reason='Opt-in CUDA/checkpoint integration test')
def test_shared_fields_match_baseline_and_isolate_geometry():
    backend = Sat3DBackend(size=64, chunk_rows=4)
    backend.set_style(np.ones(270) / 90)
    seed_everything(17)
    # Use the encoder's feature scale; arbitrary near-zero random features can
    # saturate the pretrained color head and make an RGB intervention invisible.
    raw = backend.encode(torch.zeros(1, 3, 256, 256))
    camera = dict(h_offset=12, w_offset=-20, yaw=45, pitch=10, fov=90)
    assert backend.model.num_importance == 48
    with torch.no_grad():
        seed_everything(101)
        native = backend.render(raw, camera)
        seed_everything(101)
        initial = render_fields(backend, raw, camera)
        torch.testing.assert_close(initial['base']['rgb'], native, atol=3e-5, rtol=3e-5)
        for mode in initial:
            for key in initial[mode]:
                torch.testing.assert_close(initial[mode][key], initial['base'][key], atol=0, rtol=0)
        # A nonzero intervention must change images, with geometry invariant only
        # when changing the appearance field; density and joint share geometry.
        for parameter in backend.adapter.parameters():
            parameter.add_(torch.randn_like(parameter) * .015)
        seed_everything(101)
        altered = render_fields(backend, raw, camera)
        for key in ('opacity', 'radial'):
            assert torch.equal(altered['appearance'][key], altered['base'][key])
            assert torch.equal(altered['density'][key], altered['joint'][key])
        assert not torch.equal(altered['density']['radial'], altered['base']['radial'])
        assert not torch.equal(altered['appearance']['rgb'], altered['base']['rgb'])

    # RGB loss can only update density in this mode. Checkpoint recomputation must
    # preserve both fine samples and closure-bound ray chunks during backward.
    def gradients(checkpointed):
        backend.checkpoint_rays = checkpointed
        backend.adapter.zero_grad(set_to_none=True)
        seed_everything(101)
        result = render_fields(backend, raw, camera, ('density',))['density']
        loss = result['rgb'].square().mean()
        loss.backward()
        backend.assert_frozen()
        values = [p.grad.clone() for p in backend.adapter.parameters()]
        assert all(torch.isfinite(g).all() for g in values)
        assert any(g.abs().max() > 0 for g in values)
        return loss.detach(), values

    loss_a, grad_a = gradients(False)
    loss_b, grad_b = gradients(True)
    torch.testing.assert_close(loss_a, loss_b, atol=0, rtol=0)
    for a, b in zip(grad_a, grad_b):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
