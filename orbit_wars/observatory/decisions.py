from typing import Any

_DECISIONS: list[dict[str, Any]] = []


def clear_decisions() -> None:
    _DECISIONS.clear()


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


def decisions_snapshot() -> list[dict[str, Any]]:
    return list(_DECISIONS)
