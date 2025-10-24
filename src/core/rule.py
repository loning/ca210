"""Rule definitions for the P0 physics layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Rule(Protocol):
    """Local update rule operating on a three-cell neighbourhood."""

    def local(self, left: int, center: int, right: int) -> int:
        """Return the next-state contribution for the centre cell."""


@dataclass(frozen=True)
class Rule110:
    """Default elementary cellular automaton rule (Rule 110)."""

    def local(self, left: int, center: int, right: int) -> int:
        idx = (left << 2) | (center << 1) | right
        return (0b01101110 >> idx) & 1

