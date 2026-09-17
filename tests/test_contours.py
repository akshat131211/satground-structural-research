import numpy as np
import pytest
import torch

from satground.common import ResearchError
from satground.contours import contour_support
from satground.losses import structural_losses
from satground.runs import read_config


def scene():
    classes=np.zeros((24,24),np.uint8);classes[:,12:]=2
    confidence=np.full((24,24),.9,np.float32);confidence[:,11:13]=.5
    return classes==2,(confidence>=.7)&(classes<11),classes,confidence


def test_confident_sides_restore_only_positive_targets_not_region_pixels():
    b,v,c,q=scene();result=contour_support(b,v,c,q)
    assert (result['target_boundary'] & result['legacy_valid']).sum()==0
    assert result['added_positive'].sum()==48
    assert not np.any(result['legacy_valid'] & ~result['boundary_valid'])
    assert not np.any(result['added_positive'] & ~result['target_boundary'])
    assert not v[:,11:13].any()


@pytest.mark.parametrize('occluder',[8,11,13])
def test_vegetation_and_dynamic_neighbors_cannot_bridge_edges(occluder):
    b,v,c,q=scene();c[:,11]=occluder;b=c==2;v=(q>=.7)&(c<11)
    assert not contour_support(b,v,c,q)['added_positive'].any()


def test_absent_confident_side_and_wide_unknown_band_stay_unsupervised():
    b,v,c,q=scene();q[:,12:]=.5;v=(q>=.7)&(c<11)
    assert not contour_support(b,v,c,q)['added_positive'].any()
    b,v,c,q=scene();q[:,7:17]=.5;v=(q>=.7)&(c<11)
    assert not contour_support(b,v,c,q)['added_positive'].any()


def test_empty_target_does_not_acquire_positive_boundaries():
    c=np.zeros((16,16),np.uint8);q=np.ones_like(c,dtype=np.float32)
    result=contour_support(c==2,q>.7,c,q)
    assert not result['target_boundary'].any() and not result['added_positive'].any()


def test_invalid_inputs_and_changed_region_validity_rejected():
    b,v,c,q=scene();q[0,0]=np.nan
    with pytest.raises(ResearchError):contour_support(b,v,c,q)
    b,v,c,q=scene();v[0,0]=False
    with pytest.raises(ResearchError,match='unchanged'):contour_support(b,v,c,q)


def test_correction_preserves_region_gradient_and_penalizes_missing_contour():
    b,v,c,q=scene();s=contour_support(b,v,c,q)
    tensor=lambda a:torch.tensor(a,dtype=torch.float32)[None,None]
    bt,vt,mask=tensor(b),tensor(v),tensor(s['boundary_valid'])
    p=(bt*.8+.1).requires_grad_()
    old_region,old_boundary=structural_losses(p,bt,vt)
    region,boundary=structural_losses(p,bt,vt,mask)
    assert torch.equal(region,old_region)
    assert torch.equal(torch.autograd.grad(region,p,retain_graph=True)[0],
                       torch.autograd.grad(old_region,p,retain_graph=True)[0])
    smooth=torch.full_like(p,.5,requires_grad=True)
    smooth_boundary=structural_losses(smooth,bt,vt,mask)[1]
    assert boundary < smooth_boundary
    assert torch.isfinite(torch.autograd.grad(boundary,p)[0]).all()
    assert old_boundary==0
    # Explicit legacy support reproduces the historical loss exactly.
    reference=structural_losses(smooth,bt,vt)
    explicit=structural_losses(smooth,bt,vt,tensor(s['legacy_valid']))
    assert all(torch.equal(a,b) for a,b in zip(reference,explicit))


def test_matched_contour_configs_change_only_named_objective():
    a,b=(read_config(f'configs/contour500/{name}.yaml') for name in ('A2','A3'))
    assert {k:v for k,v in a.items() if k not in ('experiment','boundary_weight')}=={
        k:v for k,v in b.items() if k not in ('experiment','boundary_weight')}
    assert a['steps']==b['steps']==500 and a['seed']==b['seed']==17
    assert a['boundary_weight']==0 and b['boundary_weight']==.5
    assert a['boundary_mode']==b['boundary_mode']=='anchored_contour_v1'
