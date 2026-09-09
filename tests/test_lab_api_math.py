"""Tests for mjlab.utils.lab_api.math module."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.lab_api.math import apply_delta_pose


@pytest.fixture
def device():
  return get_test_device()


def test_apply_delta_pose_zero_rotation_is_finite_and_identity(device):
  """Zero rotation delta should return finite values and preserve input pose."""
  source_pos = torch.zeros(2, 3, device=device)
  source_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
  delta_pose = torch.zeros(2, 6, device=device)

  target_pos, target_rot = apply_delta_pose(source_pos, source_rot, delta_pose)

  assert torch.isfinite(target_pos).all()
  assert torch.isfinite(target_rot).all()
  assert torch.allclose(target_pos, source_pos)
  assert torch.allclose(target_rot, source_rot)
