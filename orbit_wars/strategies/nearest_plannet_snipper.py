import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.ballistics import plan_shot
from orbit_wars.observatory import record_decision


def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    raw_initial = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    initial_by_id = {p[0]: Planet(*p) for p in raw_initial}

    raw_comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets
    comet_planet_ids = obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids

    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity
    current_step = obs.get("step", 0) if isinstance(obs, dict) else obs.step

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        ordered_targets = sorted(
            targets,
            key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y),
        )

        for target in ordered_targets:
            ships_needed = max(target.ships + 1, 20)
            if mine.ships < ships_needed:
                continue

            shot = plan_shot(
                mine,
                target,
                ships=ships_needed,
                current_step=current_step,
                angular_velocity=angular_velocity,
                initial_to_planet=initial_by_id.get(target.id),
                planets=planets,
                initial_planets=initial_by_id,
                comets=raw_comets,
                comet_planet_ids=comet_planet_ids,
            )

            if not shot.valid:
                continue

            moves.append([mine.id, shot.angle, ships_needed])
            record_decision(
                step=current_step,
                player=player,
                source_planet_id=mine.id,
                target_planet_id=target.id,
                angle_rad=shot.angle,
                ships=ships_needed,
            )
            break

    return moves
