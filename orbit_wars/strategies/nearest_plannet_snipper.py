import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from orbit_wars.army.ballistics import aim_angle


def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    raw_initial = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    initial_by_id = {p[0]: Planet(*p) for p in raw_initial}

    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity
    current_step = obs.get("step", 0) if isinstance(obs, dict) else obs.step

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = min(targets, key=lambda t: math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2))

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            angle = aim_angle(
                mine, nearest,
                ships=ships_needed,
                current_step=current_step,
                angular_velocity=angular_velocity,
                initial_to_planet=initial_by_id.get(nearest.id),
            )
            moves.append([mine.id, angle, ships_needed])

    return moves
