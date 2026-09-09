"""Electric actuator utilities."""

import math
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class ElectricActuator:
  """Electric actuator parameters."""

  reflected_inertia: float
  velocity_limit: float
  effort_limit: float


def reflected_inertia(
  rotor_inertia: float,
  gear_ratio: float,
) -> float:
  """Compute reflected inertia of a single-stage gearbox."""
  return rotor_inertia * gear_ratio**2


def reflected_inertia_from_two_stage_planetary(
  rotor_inertia: tuple[float, float, float],
  gear_ratio: tuple[float, float, float],
) -> float:
  """Compute reflected inertia of a two-stage planetary gearbox."""
  assert gear_ratio[0] == 1
  r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
  r2 = rotor_inertia[1] * gear_ratio[2] ** 2
  r3 = rotor_inertia[2]
  return r1 + r2 + r3


def rpm_to_rad(rpm: float) -> float:
  """Convert revolutions per minute to radians per second."""
  return (rpm * 2 * math.pi) / 60


class LinearJointProperties(NamedTuple):
  """Linear joint properties reflected from a rotary actuator."""

  armature: float  # kg
  velocity_limit: float  # m/s
  effort_limit: float  # N


def reflect_rotary_to_linear(
  armature_rotary: float,
  velocity_limit_rotary: float,
  effort_limit_rotary: float,
  transmission_ratio: float,
) -> LinearJointProperties:
  """Reflect rotary motor properties through a transmission to linear joint properties.

  Converts motor specs from rotary coordinates (rad, rad/s, Nm) to equivalent linear
  properties (kg, m/s, N) for simulation. Uses energy and power equivalence:
    - Linear mass: m = I / r²
    - Linear velocity: v = r · ω
    - Linear force: F = τ / r
  where r is the transmission ratio [m/rad].

  Args:
    armature_rotary: Reflected inertia at motor output [kg⋅m²].
    velocity_limit_rotary: Maximum angular velocity [rad/s].
    effort_limit_rotary: Maximum torque [Nm].
    transmission_ratio: Linear displacement per radian [m/rad].

  Returns:
    Linear joint properties (armature [kg], velocity_limit [m/s], effort_limit [N])
  """
  armature_linear = armature_rotary / (transmission_ratio**2)
  velocity_limit_linear = velocity_limit_rotary * transmission_ratio
  effort_limit_linear = effort_limit_rotary / transmission_ratio

  return LinearJointProperties(
    armature_linear, velocity_limit_linear, effort_limit_linear
  )
