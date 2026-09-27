"""Planogram configuration schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanogramSlot:
    """An expected SKU at a normalized horizontal position in one shelf row."""

    slot_id: str
    expected_sku: str
    x_center: float
    tolerance: float = 0.08


@dataclass(frozen=True)
class PlanogramRow:
    """One shelf row and its ordered planogram slots."""

    row_id: str
    y_center: float
    slots: tuple[PlanogramSlot, ...]
    y_tolerance: float = 0.20


@dataclass(frozen=True)
class Planogram:
    """Named planogram containing one or more shelf rows."""

    shelf_id: str
    rows: tuple[PlanogramRow, ...]


def parse_planogram(data: Any) -> Planogram:
    """Validate and parse a planogram JSON-compatible mapping.

    Schema: {"shelf_id": str, "rows": [{"row_id": str, "y_center": 0..1,
    "slots": [{"slot_id": str, "expected_sku": str, "x_center": 0..1,
    "tolerance": 0..0.5}]}]}.
    """
    if not isinstance(data, dict):
        raise ValueError("Planogram must be a JSON object.")
    shelf_id = _non_empty_string(data.get("shelf_id"), "shelf_id")
    rows_data = data.get("rows")
    if not isinstance(rows_data, list) or not rows_data:
        raise ValueError("Planogram must contain at least one row.")
    rows: list[PlanogramRow] = []
    seen_rows: set[str] = set()
    for row in rows_data:
        if not isinstance(row, dict):
            raise ValueError("Each row must be a JSON object.")
        row_id = _non_empty_string(row.get("row_id"), "row_id")
        if row_id in seen_rows:
            raise ValueError(f"Duplicate row_id: {row_id}")
        seen_rows.add(row_id)
        y = _unit_interval(row.get("y_center"), f"{row_id}.y_center")
        y_tolerance = float(row.get("y_tolerance", 0.20))
        if not 0 <= y_tolerance <= 0.5:
            raise ValueError(f"{row_id}.y_tolerance must be in [0, 0.5].")
        slot_data = row.get("slots")
        if not isinstance(slot_data, list) or not slot_data:
            raise ValueError(f"Row {row_id} must contain at least one slot.")
        slots: list[PlanogramSlot] = []
        seen_slots: set[str] = set()
        for item in slot_data:
            if not isinstance(item, dict):
                raise ValueError(f"Slots in {row_id} must be JSON objects.")
            slot_id = _non_empty_string(item.get("slot_id"), "slot_id")
            sku = _non_empty_string(item.get("expected_sku"), "expected_sku")
            if slot_id in seen_slots:
                raise ValueError(f"Duplicate slot_id in {row_id}: {slot_id}")
            seen_slots.add(slot_id)
            x = _unit_interval(item.get("x_center"), f"{slot_id}.x_center")
            tolerance = float(item.get("tolerance", 0.08))
            if not 0 <= tolerance <= 0.5:
                raise ValueError(f"{slot_id}.tolerance must be in [0, 0.5].")
            slots.append(PlanogramSlot(slot_id, sku, x, tolerance))
        slots.sort(key=lambda slot: slot.x_center)
        rows.append(PlanogramRow(row_id, y, tuple(slots), y_tolerance))
    rows.sort(key=lambda row: row.y_center)
    return Planogram(shelf_id, tuple(rows))


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _unit_interval(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number from 0 to 1.") from exc
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be from 0 to 1.")
    return number
