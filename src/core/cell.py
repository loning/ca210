"""Reversible lattice cells (supporting nested rings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .rule import Rule


@dataclass
class Cell:
    """Cell node; leaf or composite with optional children forming a ring."""

    x_prev: Optional[int] = None
    x_cur: Optional[int] = None
    children: Optional[List["Cell"]] = None  # None for leaf, list for ring
    _stage_next: Optional[int] = field(init=False, default=None, repr=False)
    _stage_prev: Optional[int] = field(init=False, default=None, repr=False)

    def is_leaf(self) -> bool:
        return self.children is None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def leaf(x_prev: int, x_cur: int) -> "Cell":
        _validate_bit(x_prev)
        _validate_bit(x_cur)
        return Cell(x_prev=x_prev & 1, x_cur=x_cur & 1, children=None)

    @staticmethod
    def node(x_prev: int, x_cur: int, children: List["Cell"]) -> "Cell":
        if not children:
            raise ValueError("Composite cell requires at least one child")
        _validate_bit(x_prev)
        _validate_bit(x_cur)
        return Cell(x_prev=x_prev & 1, x_cur=x_cur & 1, children=list(children))

    @staticmethod
    def ring(children: List["Cell"]) -> "Cell":
        if not children:
            raise ValueError("Ring must contain at least one child cell")
        return Cell(children=list(children))  # container/root

    # ------------------------------------------------------------------
    # Evolution API
    # ------------------------------------------------------------------
    def step(self, rule: Rule) -> None:
        """Advance one timestep using the provided rule."""

        if self.children is None:
            raise RuntimeError("Leaf cell cannot step without ring context")
        _step_ring(self.children, rule)

    def step_inverse(self, rule: Rule) -> None:
        """Reverse one timestep using the provided rule."""

        if self.children is None:
            raise RuntimeError("Leaf cell cannot step without ring context")
        _step_inverse_ring(self.children, rule)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_leaf(x_prev: int, x_cur: int) -> Cell:
    return Cell.leaf(x_prev, x_cur)


def make_node(x_prev: int, x_cur: int, children: List[Cell]) -> Cell:
    return Cell.node(x_prev, x_cur, children)


def make_ring(cells: List[Cell]) -> Cell:
    return Cell.ring(cells)


def _validate_bit(value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"Bit value must be 0 or 1, got {value!r}")


def _step_ring(cells: List[Cell], rule: Rule) -> None:
    _ensure_ring(cells)
    n = len(cells)
    # Stage 1: compute x_next for every leaf without mutating state
    for idx, cell in enumerate(cells):
        left = cells[(idx - 1) % n]
        right = cells[(idx + 1) % n]
        l = _read_bit(left.x_cur)
        c = _read_bit(cell.x_cur)
        r = _read_bit(right.x_cur)
        d = _read_bit(cell.x_prev)
        x_next = (d ^ rule.local(l, c, r)) & 1
        cell._stage_next = x_next

    # Stage 2: commit (swap registers)
    for cell in cells:
        x_old_cur = _read_bit(cell.x_cur)
        cell.x_prev = x_old_cur
        cell.x_cur = cell._stage_next
        cell._stage_next = None

    # Recurse into children after parent update
    for cell in cells:
        if cell.children:
            _step_ring(cell.children, rule)


def _step_inverse_ring(cells: List[Cell], rule: Rule) -> None:
    _ensure_ring(cells)
    n = len(cells)
    # Undo nested rings first (reverse order of forward step)
    for cell in cells:
        if cell.children:
            _step_inverse_ring(cell.children, rule)

    # Stage 1: compute previous x_prev (old state) without mutating
    for idx, cell in enumerate(cells):
        left = cells[(idx - 1) % n]
        right = cells[(idx + 1) % n]
        l = _read_bit(left.x_prev)
        c = _read_bit(cell.x_prev)
        r = _read_bit(right.x_prev)
        x_next = _read_bit(cell.x_cur)
        old_x_prev = (x_next ^ rule.local(l, c, r)) & 1
        cell._stage_prev = old_x_prev

    # Stage 2: commit back to old state
    for cell in cells:
        old_x_cur = _read_bit(cell.x_prev)
        cell.x_cur = old_x_cur
        cell.x_prev = cell._stage_prev
        cell._stage_prev = None


def _ensure_ring(cells: List[Cell]) -> None:
    if not cells:
        raise ValueError("Ring must contain at least one cell")
    for cell in cells:
        if cell.x_prev not in (0, 1):
            raise ValueError("Cell missing x_prev bit")
        if cell.x_cur not in (0, 1):
            raise ValueError("Cell missing x_cur bit")


def _read_bit(value: Optional[int]) -> int:
    if value not in (0, 1):
        raise ValueError("Bit value must be 0 or 1")
    return value
