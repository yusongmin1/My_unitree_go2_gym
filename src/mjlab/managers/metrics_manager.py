"""Metrics manager for logging custom per-step metrics during training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MetricsTermCfg(ManagerTermBaseCfg):
  """Configuration for a metrics term."""

  pass


class MetricsManager(ManagerBase):
  """Accumulates per-step metric values, reports episode averages.

  Unlike rewards, metrics have no weight, no dt scaling, and no
  normalization by episode length. Episode values are true per-step
  averages (sum / step_count), so a metric in [0,1] stays in [0,1]
  in the logger.
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, MetricsTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[MetricsTermCfg] = list()
    self._class_term_cfgs: list[MetricsTermCfg] = list()

    self.cfg = deepcopy(cfg)
    super().__init__(env=env)

    self._episode_sums: dict[str, torch.Tensor] = {}
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._step_values = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Metrics Terms"
    table.field_names = ["Index", "Name"]
    table.align["Name"] = "l"
    for index, name in enumerate(self._term_names):
      table.add_row([index, name])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  # Methods.

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    extras = {}
    counts = self._step_count[env_ids].float()
    # Avoid division by zero for envs that haven't stepped.
    safe_counts = torch.clamp(counts, min=1.0)
    for key in self._episode_sums:
      episode_avg = torch.mean(self._episode_sums[key][env_ids] / safe_counts)
      extras["Episode_Metrics/" + key] = episode_avg
      self._episode_sums[key][env_ids] = 0.0
    self._step_count[env_ids] = 0
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self) -> None:
    self._step_count += 1
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      value = term_cfg.func(self._env, **term_cfg.params)
      self._episode_sums[name] += value
      self._step_values[:, term_idx] = value

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_values[env_idx, idx].cpu().item()]))
    return terms

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: MetricsTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)


class NullMetricsManager:
  """Placeholder for absent metrics manager that safely no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self.cfg = None

  def __str__(self) -> str:
    return "<NullMetricsManager> (inactive)"

  def __repr__(self) -> str:
    return "NullMetricsManager()"

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    return {}

  def compute(self) -> None:
    pass
