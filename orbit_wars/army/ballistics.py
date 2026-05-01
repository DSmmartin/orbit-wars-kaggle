"""
Ballistics utilities for Orbit Wars.

Fleets travel in **straight lines** at constant speed — no gravity, no curves.

Speed formula (from the environment source):
    speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ** 1.5
    speed = min(speed, max_speed)          # default max_speed = 6.0

Planet rotation (only planets where orbital_radius + radius < 50):
    position(step) = (50 + r*cos(initial_angle + ω*step),
                      50 + r*sin(initial_angle + ω*step))
    where  r            = orbital radius from initial_planets
           initial_angle = atan2 of initial planet relative to sun centre
           ω             = obs.angular_velocity  (rad/step, same for all planets)

For static planets aim_angle() returns a direct atan2 shot.
For rotating planets it iterates to find the future intercept position.

Key implementation notes:
- Float steps are used throughout iteration (no round()) to avoid quantisation
  oscillation between adjacent integers.
- The fleet spawns planet_radius + 0.1 ahead of the planet centre, so the
  actual travel distance is center-to-target minus that offset.
- planet_position_at_step() accepts float steps for smooth convergence.
"""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

# Mirror the constants used by the environment
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0  # orbital_radius + planet_radius must be < this to rotate


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    """Return the travel speed (units/step) for a fleet of *ships* ships."""
    if ships <= 1:
        return 1.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)


def is_rotating(planet: Planet) -> bool:
    """Return True if *planet* orbits the sun and therefore moves each step."""
    orbital_radius = math.sqrt((planet.x - CENTER) ** 2 + (planet.y - CENTER) ** 2)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def planet_position_at_step(
    initial_planet: Planet,
    step: float,
    angular_velocity: float,
) -> tuple[float, float]:
    """
    Return the (x, y) position of a planet at absolute game *step*.

    Uses *initial_planet* to derive orbital radius and initial angle — the same
    formula the environment applies (orbit_wars.py lines 563-570).
    Accepts a float step for smooth interpolation during ballistic iteration.
    For static planets the position is constant and equals initial_planet.(x, y).
    """
    dx = initial_planet.x - CENTER
    dy = initial_planet.y - CENTER
    orbital_radius = math.sqrt(dx ** 2 + dy ** 2)

    if orbital_radius + initial_planet.radius >= ROTATION_RADIUS_LIMIT:
        # Static planet — never moves
        return initial_planet.x, initial_planet.y

    initial_angle = math.atan2(dy, dx)
    angle = initial_angle + angular_velocity * step
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )


# ---------------------------------------------------------------------------
# Main ballistics solver
# ---------------------------------------------------------------------------

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
) -> float:
    """
    Return the launch angle (radians) needed to intercept *to_planet*.

    The angle can be passed directly as the second element of an action:
        action = [[from_planet.id, angle, ships]]

    Parameters
    ----------
    from_planet       : Source planet at the current step (from obs.planets).
    to_planet         : Target planet at the current step (from obs.planets).
    ships             : Fleet size (determines travel speed).
    current_step      : Current game step (obs.step).
    angular_velocity  : Shared angular velocity in rad/step (obs.angular_velocity).
    initial_to_planet : Initial position of the target from obs.initial_planets.
                        Required for accurate rotating-planet intercepts.
                        Falls back to *to_planet* when omitted (less accurate).
    max_speed         : Maximum fleet speed (configuration.shipSpeed, default 6.0).
    max_iterations    : Iteration cap for the rotating-planet solver.
    step_tolerance    : Convergence threshold in steps (default 0.05 steps).

    Returns
    -------
    float
        Launch angle in radians.  0 = right (+x), π/2 = down (+y) in board coords.
    """
    speed = fleet_speed(ships, max_speed)
    fx, fy = from_planet.x, from_planet.y

    if initial_to_planet is None:
        initial_to_planet = to_planet

    # --- Static target: direct shot ---
    if not is_rotating(to_planet):
        return math.atan2(to_planet.y - fy, to_planet.x - fx)

    # --- Rotating target: iterative intercept ---
    #
    # The fleet spawns at planet_center + (planet_radius + 0.1) * direction,
    # so the actual travel distance is center-to-target minus that offset.
    # Using round() on t_est quantises the arrival step and can cause the loop
    # to oscillate between adjacent integers and never converge. We use float
    # steps throughout; planet_position_at_step accepts floats and interpolates
    # the rotation angle continuously.
    #
    launch_offset = from_planet.radius + 0.1

    tx, ty = to_planet.x, to_planet.y
    d = math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2)
    t_est = max(0.0, d - launch_offset) / speed

    for _ in range(max_iterations):
        arrival_step = current_step + t_est          # float — no rounding
        tx, ty = planet_position_at_step(initial_to_planet, arrival_step, angular_velocity)

        d_new = math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2)
        t_new = max(0.0, d_new - launch_offset) / speed

        if abs(t_new - t_est) < step_tolerance:
            break
        t_est = t_new

    return math.atan2(ty - fy, tx - fx)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def estimated_travel_steps(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    max_speed: float = 6.0,
) -> float:
    """
    Approximate steps for a fleet to reach *to_planet* in a straight-line shot
    to its *current* position.  Accounts for the fleet spawn offset so the
    estimate matches what aim_angle uses internally.
    """
    d = math.sqrt(
        (from_planet.x - to_planet.x) ** 2 + (from_planet.y - to_planet.y) ** 2
    )
    launch_offset = from_planet.radius + 0.1
    return max(0.0, d - launch_offset) / fleet_speed(ships, max_speed)
