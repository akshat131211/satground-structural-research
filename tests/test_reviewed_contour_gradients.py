"""CPU autograd and immutable-record tests; no CUDA calls or research inputs."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import probe_reviewed_contour_gradients as probe
from satground.common import ResearchError, object_hash
from satground.losses import structural_losses


PROTOCOL = dict(weights={'rgb': 1., 'lpips': .1, 'region': .5, 'boundary': .5},
                image_gradient_relative_tolerance=5e-5, adapter_gradient_relative_tolerance=2e-4)


@pytest.fixture(scope='module', autouse=True)
def small_cpu_cases():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def masks(size=9):
    building = torch.zeros(1, 1, size, size)
    building[:, :, 2:6, 3:7] = 1
    return building, torch.ones_like(building)


def test_partition_matches_scalar_and_probability_gradients():
    building, valid = masks()
    probability = torch.linspace(.05, .95, 81).reshape(1, 1, 9, 9).requires_grad_()
    parts, counts = probe.boundary_partition(probability, building, valid)
    _, expected = structural_losses(probability, building, valid)
    assert float(sum(parts.values()).detach()) == pytest.approx(float(expected.detach()))
    a = torch.autograd.grad(sum(parts.values()), probability, retain_graph=True)[0]
    b = torch.autograd.grad(expected, probability)[0]
    assert torch.equal(a, b)
    assert counts['positive_target_contour_pixels'] + counts['zero_target_contour_pixels'] == counts['boundary_valid_pixels']


@pytest.mark.parametrize('value', [0., 1.])
def test_clamping_is_exactly_the_original_loss(value):
    building, valid = masks()
    probability = torch.full_like(building, value, requires_grad=True)
    parts, _ = probe.boundary_partition(probability, building, valid)
    _, expected = structural_losses(probability, building, valid)
    assert torch.equal(sum(parts.values()), expected)


def test_ignored_neighborhood_and_empty_support_remain_empty():
    building, valid = masks()
    valid.zero_()
    valid[:, :, 4, :] = 1  # Pixel-valid stripe has no fully scored 3x3 neighborhoods.
    parts, counts = probe.boundary_partition(torch.rand_like(building), building, valid)
    assert counts['boundary_valid_pixels'] == 0
    assert all(float(v) == 0 for v in parts.values())
    valid.fill_(1)
    parts, counts = probe.boundary_partition(torch.rand_like(building), torch.zeros_like(building), valid)
    assert counts['positive_target_contour_pixels'] == 0 and float(parts['boundary_positive']) == 0


def test_partition_rejects_nonbinary_reviewed_masks():
    building, valid = masks()
    valid[:, :, 1, 1] = .5
    with pytest.raises(ResearchError, match='reviewed labels'):
        probe.boundary_partition(torch.rand_like(building), building, valid)


def test_zero_and_conflicting_gradients_are_reported_honestly():
    vectors = {k: torch.zeros(3) for k in probe.COMPONENTS}
    empty = probe.gradient_summary(vectors)
    assert all(v == 0 for v in empty['weighted_norms'].values())
    assert all(v is None for v in empty['cosines'].values())
    assert all(v is None for v in empty['norm_ratios'].values())
    vectors.update(rgb=torch.tensor([1., 0., 0.]), boundary_positive=torch.tensor([2., 0., 0.]),
                   boundary_zero=torch.tensor([-1., 0., 0.]))
    result = probe.gradient_summary(vectors)
    assert result['cosines']['boundary_positive_vs_boundary_zero'] == -1
    assert result['norm_ratios']['zero_vs_positive'] == .5


class FakeObjective:
    """Small analytic CPU surrogate tests graph connectivity, not research results."""
    segmenter = type('Semantic', (), {'probabilities': staticmethod(lambda x: x.softmax(1))})()
    perceptual = staticmethod(lambda a, b: (a - b).square().mean().reshape(1))

    def __call__(self, prediction, target, labels):
        region, boundary = structural_losses(self.segmenter.probabilities(prediction)[:, 2:3], labels['building'], labels['valid'])
        terms = dict(rgb=F.l1_loss(prediction, target), lpips=self.perceptual(prediction * 2 - 1, target * 2 - 1).mean(),
                     region=region, boundary=boundary)
        terms['total'] = terms['rgb'] + .1 * terms['lpips'] + .5 * (region + boundary)
        return terms['total'], {k: float(v.detach()) for k, v in terms.items()}


def test_direct_shared_graph_decomposition_reaches_parameters_without_updates(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA called'))
    parameter = torch.nn.Parameter(torch.linspace(.1, .9, 3 * 9 * 9).reshape(1, 3, 9, 9))
    before = parameter.detach().clone()
    prediction = parameter.sigmoid()  # Original connected render surrogate; not a detached leaf.
    target = torch.zeros_like(prediction)
    building, valid = masks()
    record, vectors = probe.measure(FakeObjective(), prediction, target, {'building': building, 'valid': valid}, (parameter,), PROTOCOL)
    assert torch.equal(parameter, before) and parameter.grad is None
    assert record['gradient_sum_relative_error']['adapter'] < 2e-4
    assert vectors['adapter']['rgb'].norm() > 0
    assert record['support']['positive_target_contour_pixels'] > 0


def test_wrong_values_and_nonfinite_vectors_are_rejected():
    components = {k: 1. for k in probe.COMPONENTS}
    with pytest.raises(ResearchError, match='Component values'):
        probe.verify_loss_values(components, dict(rgb=1., lpips=1., region=1., boundary=1., total=3.), PROTOCOL['weights'])
    vectors = {k: torch.zeros(3) for k in probe.COMPONENTS}
    vectors['boundary_zero'][0] = float('nan')
    with pytest.raises(ResearchError, match='Invalid gradient'):
        probe.gradient_summary(vectors)


def record_fixture(tmp_path):
    plan = dict(state='A0', fitting_view=1, sample_id='SYNTHETIC-ONLY', seed=101)
    identity = dict(protocol=PROTOCOL, plan=[plan])
    state = dict(identity=identity, identity_hash=object_hash(identity), completed=1, receipts=[])
    vectors = {space: {k: torch.zeros(length) for k in probe.COMPONENTS}
               for space, length in (('image', 3 * 256 * 256), ('adapter', 4448))}
    vectors['actual_image'] = torch.zeros(3 * 256 * 256)
    vectors['actual_adapter'] = torch.zeros(4448)
    record = dict(plan=plan, parameters_unchanged=True, optimizer_updates=0,
        raw_components={k: 0. for k in probe.COMPONENTS}, objective_values={k: 0. for k in ('rgb', 'lpips', 'region', 'boundary', 'total')},
        gradient_sum_relative_error={'image': 0., 'adapter': 0.},
        image_gradients=probe.gradient_summary(vectors['image']), adapter_gradients=probe.gradient_summary(vectors['adapter']))
    state['receipts'] = [probe.commit_record(tmp_path, 1, record, vectors, state['identity_hash'])]
    return state


def test_compatible_resume_verifies_exact_completed_records(tmp_path):
    state = record_fixture(tmp_path)
    assert len(probe.verified_records(tmp_path, state)) == 1
    with pytest.raises(ResearchError, match='Partial check'):
        probe.commit_record(tmp_path, 1, {}, {}, state['identity_hash'])
    (tmp_path / 'checks/0002').mkdir()
    with pytest.raises(ResearchError, match='Uncommitted partial'):
        probe.verified_records(tmp_path, state)


@pytest.mark.parametrize('filename', ['record.json', 'gradients.pt', 'receipt.json'])
def test_resume_rejects_any_changed_row_bytes(tmp_path, filename):
    state = record_fixture(tmp_path)
    path = tmp_path / 'checks/0001' / filename
    path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    with pytest.raises(ResearchError, match='changed|hashes differ'):
        probe.verified_records(tmp_path, state)


def test_partial_diagnosis_cannot_be_promoted_to_public_summary():
    with pytest.raises(ResearchError, match='completed24'):
        probe.aggregate([], {'completed': 0, 'expected': 24})
