from typing import Any

_DECISIONS: list[dict[str, Any]] = []
_ENEMY_FLEETS: list[dict[str, Any]] = []
_SEEN_FLEET_IDS: set[int] = set()


def clear_decisions() -> None:
    _DECISIONS.clear()
    _ENEMY_FLEETS.clear()
    _SEEN_FLEET_IDS.clear()


def record_decision(
    *,
    step: int,
    player: int,
    source_planet_id: int,
    target_planet_id: int,
    angle_rad: float,
    ships: int,
) -> None:
    _DECISIONS.append(
        {
            "step": step,
            "player": player,
            "source_planet_id": source_planet_id,
            "target_planet_id": target_planet_id,
            "angle_rad": round(angle_rad, 4),
            "ships": ships,
        }
    )


def record_enemy_fleet(
    *,
    step: int,
    fleet_id: int,
    target_planet_id: int,
    time_arrival: int,
    ships: int,
    angle_rad: float,
) -> None:
    """Record the first detection of an enemy fleet in this episode."""
    if fleet_id in _SEEN_FLEET_IDS:
        return
    _SEEN_FLEET_IDS.add(fleet_id)
    _ENEMY_FLEETS.append(
        {
            "step": step,
            "fleet_id": fleet_id,
            "target_planet_id": target_planet_id,
            "time_arrival": time_arrival,
            "ships": ships,
            "angle_rad": round(angle_rad, 4),
        }
    )


def decisions_snapshot() -> list[dict[str, Any]]:
    return list(_DECISIONS)


def enemy_fleets_snapshot() -> list[dict[str, Any]]:
    return list(_ENEMY_FLEETS)
