"""Ballistics utilities for Orbit Wars with scenario-based solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.ballistics_comet import solve_comet_intercept
from orbit_wars.army.ballistics_evaluator import ShotEvaluation, evaluate_shot
from orbit_wars.army.ballistics_rotating import solve_rotating_planet_intercept
from orbit_wars.army.ballistics_static import solve_static_planet_intercept
from orbit_wars.army.physics import fleet_speed, is_rotating, planet_position_at_step
from orbit_wars.observatory import timed_calc


@dataclass(slots=True)
class ShotPlan:
    """Planned shot metadata including validity/evaluation details."""

    angle: float
    scenario: str
    valid: bool
    eta_steps: float | None = None
    reason: str | None = None
    evaluation: ShotEvaluation | None = None


def _as_initial_by_id(initial_planets: Sequence[Planet] | Mapping[int, Planet] | None) -> dict[int, Planet]:
    """Normalize `initial_planets` into an id-indexed dictionary."""
    if initial_planets is None:
        return {}
    if isinstance(initial_planets, Mapping):
        return dict(initial_planets)
    return {p.id: p for p in initial_planets}


def _is_comet_target(to_planet: Planet, comet_planet_ids: Sequence[int] | None) -> bool:
    """Return whether the target planet id corresponds to an active comet body."""
    return bool(comet_planet_ids) and to_planet.id in comet_planet_ids


def classify_scenario(
    to_planet: Planet,
    *,
    comet_planet_ids: Sequence[int] | None,
) -> str:
    """Classify the target as static, rotating planet, or comet scenario."""
    if _is_comet_target(to_planet, comet_planet_ids):
        return "comet"
    if is_rotating(to_planet):
        return "moving_planet"
    return "static_planet"


def plan_shot(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    current_step: int,
    angular_velocity: float,
    *,
    initial_to_planet: Planet | None = None,
    planets: Sequence[Planet] | None = None,
    initial_planets: Sequence[Planet] | Mapping[int, Planet] | None = None,
    comets: Sequence[Any] | None = None,
    comet_planet_ids: Sequence[int] | None = None,
    max_speed: float = 6.0,
    max_iterations: int = 100,
    step_tolerance: float = 0.05,
    evaluation_horizon: int = 220,
) -> ShotPlan:
    """Create a scenario-specific shot plan and evaluate its validity."""
    with timed_calc(
        "ballistics.plan_shot",
        source=from_planet.id,
        target=to_planet.id,
        ships=ships,
        step=current_step,
    ):
        scenario = classify_scenario(to_planet, comet_planet_ids=comet_planet_ids)
        eta_steps: float | None = None

        if scenario == "static_planet":
            angle, eta_steps = solve_static_planet_intercept(
                from_planet,
                to_planet,
                ships,
                max_speed=max_speed,
            )
        elif scenario == "moving_planet":
            angle, eta_steps = solve_rotating_planet_intercept(
                from_planet,
                to_planet,
                ships,
                current_step,
                angular_velocity,
                initial_to_planet=initial_to_planet or to_planet,
                max_speed=max_speed,
                max_iterations=max_iterations,
                step_tolerance=step_tolerance,
            )
        else:
            angle, eta_steps, reason = solve_comet_intercept(
                from_planet,
                to_planet,
                ships,
                comets=list(comets) if comets is not None else None,
                max_speed=max_speed,
            )
            if angle is None:
                return ShotPlan(
                    angle=math.atan2(to_planet.y - from_planet.y, to_planet.x - from_planet.x),
                    scenario=scenario,
                    valid=False,
                    eta_steps=eta_steps,
                    reason=reason or "comet_solver_failed",
                )

        # When no world-state was provided, we can only return the raw solver result.
        if planets is None:
            return ShotPlan(
                angle=angle,
                scenario=scenario,
                valid=True,
                eta_steps=eta_steps,
            )

        initial_by_id = _as_initial_by_id(initial_planets)
        if not initial_by_id and initial_to_planet is not None:
            initial_by_id[to_planet.id] = initial_to_planet

        evaluation = evaluate_shot(
            from_planet=from_planet,
            target_id=to_planet.id,
            angle=angle,
            ships=ships,
            current_step=current_step,
            angular_velocity=angular_velocity,
            planets=planets,
            initial_by_id=initial_by_id,
            comets=comets,
            max_speed=max_speed,
            max_steps=evaluation_horizon,
        )
        return ShotPlan(
            angle=angle,
            scenario=scenario,
            valid=evaluation.valid,
            eta_steps=eta_steps,
            reason=evaluation.reason,
            evaluation=evaluation,
        )


def aim_angle(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    current_step: int,
    angular_velocity: float,
    *,
    initial_to_planet: Planet | None = None,
    max_speed: float = 6.0,
    max_iterations: int = 100,
    step_tolerance: float = 0.05,
    comets: Sequence[Any] | None = None,
    comet_planet_ids: Sequence[int] | None = None,
) -> float:
    """
    Backward-compatible angle API.

    This returns only the launch angle. Use `plan_shot` when you need scenario
    classification and invalid-target checks.
    """
    plan = plan_shot(
        from_planet,
        to_planet,
        ships,
        current_step,
        angular_velocity,
        initial_to_planet=initial_to_planet,
        planets=None,
        initial_planets=None,
        comets=comets,
        comet_planet_ids=comet_planet_ids,
        max_speed=max_speed,
        max_iterations=max_iterations,
        step_tolerance=step_tolerance,
    )
    return plan.angle


def estimated_travel_steps(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    max_speed: float = 6.0,
) -> float:
    """Approximate travel steps to the target's current position."""
    d = math.hypot(from_planet.x - to_planet.x, from_planet.y - to_planet.y)
    launch_offset = from_planet.radius + 0.1
    return max(0.0, d - launch_offset) / fleet_speed(ships, max_speed)


__all__ = [
    "ShotPlan",
    "aim_angle",
    "plan_shot",
    "classify_scenario",
    "estimated_travel_steps",
    "fleet_speed",
    "is_rotating",
    "planet_position_at_step",
]
