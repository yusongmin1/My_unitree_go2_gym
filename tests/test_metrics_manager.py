"""Tests for metrics manager functionality."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.metrics_manager import (
  MetricsManager,
  MetricsTermCfg,
  NullMetricsManager,
)


class SimpleTestMetric:
  """A class-based metric that tracks state."""

  def __init__(self, cfg: MetricsTermCfg, env):
    self.call_count = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, **kwargs):
    self.call_count += 1
    return torch.ones(env.num_envs, device=env.device) * 0.5

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.call_count[env_ids] = 0


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.scene = {"robot": Mock()}
  return env


def test_episode_averages_and_reset(mock_env):
  """Compute for N steps, reset a subset, verify averages and zeroing."""
  cfg = {
    "term": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device) * 0.5,
      params={},
    )
  }
  manager = MetricsManager(cfg, mock_env)

  for _ in range(10):
    manager.compute()

  info = manager.reset(env_ids=torch.tensor([0, 1]))

  # Each env: sum=5.0, count=10, avg=0.5. Mean across 2 reset envs = 0.5.
  assert info["Episode_Metrics/term"].item() == pytest.approx(0.5)
  # Reset envs zeroed; non-reset envs untouched.
  assert manager._episode_sums["term"][0] == 0.0
  assert manager._step_count[0] == 0
  assert manager._episode_sums["term"][2] == pytest.approx(5.0)
  assert manager._step_count[2] == 10


def test_early_termination_uses_per_env_step_count(mock_env):
  """Envs with different episode lengths get correct per-step averages."""
  step = [0]

  def step_dependent_metric(env):
    step[0] += 1
    return torch.full((env.num_envs,), float(step[0]), device=env.device)

  cfg = {"m": MetricsTermCfg(func=step_dependent_metric, params={})}
  manager = MetricsManager(cfg, mock_env)

  # 4 steps for all envs: values are 1, 2, 3, 4.
  for _ in range(4):
    manager.compute()
  # Env 0: sum=10, count=4. Reset it (env 1 keeps accumulating).
  manager.reset(env_ids=torch.tensor([0]))

  # 2 more steps: values are 5, 6.
  for _ in range(2):
    manager.compute()
  # Env 0: sum=11, count=2, avg=5.5.
  # Env 1: sum=21, count=6, avg=3.5.
  info = manager.reset(env_ids=torch.tensor([0, 1]))
  # Mean of [5.5, 3.5] = 4.5.
  assert info["Episode_Metrics/m"].item() == pytest.approx(4.5)


def test_class_based_metric_reset_targets_correct_envs(mock_env):
  """Class-based term's reset() is called with the correct env_ids."""
  cfg = {"term": MetricsTermCfg(func=SimpleTestMetric, params={})}
  manager = MetricsManager(cfg, mock_env)
  term = manager._class_term_cfgs[0].func

  for _ in range(10):
    manager.compute()

  manager.reset(env_ids=torch.tensor([0, 2]))

  assert term.call_count[0] == 0
  assert term.call_count[1] == 10
  assert term.call_count[2] == 0
  assert term.call_count[3] == 10


def test_null_metrics_manager(mock_env):
  """NullMetricsManager doesn't crash and returns empty dict on reset."""
  manager = NullMetricsManager()
  manager.compute()
  assert manager.reset(env_ids=torch.tensor([0])) == {}


def test_none_terms_are_skipped(mock_env):
  """None terms in config are skipped without error."""
  cfg: dict[str, MetricsTermCfg | None] = {
    "valid": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device),
      params={},
    ),
    "skipped": None,
  }
  manager = MetricsManager(cfg, mock_env)  # type: ignore[arg-type]
  assert manager._term_names == ["valid"]
