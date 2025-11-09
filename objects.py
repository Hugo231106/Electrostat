"""Physical entities that can be manipulated in the 2D scene."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PointCharge:
    """A point charge defined by its position and magnitude."""

    x: float
    y: float
    q: float

    @property
    def position(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class LineCharge:
    """A straight line charge described by two points and its linear density."""

    x1: float
    y1: float
    x2: float
    y2: float
    linear_density: float

    @property
    def start(self) -> Tuple[float, float]:
        return self.x1, self.y1

    @property
    def end(self) -> Tuple[float, float]:
        return self.x2, self.y2


@dataclass
class CurrentWire:
    """A wire segment carrying an electric current."""

    x1: float
    y1: float
    x2: float
    y2: float
    current: float

    @property
    def start(self) -> Tuple[float, float]:
        return self.x1, self.y1

    @property
    def end(self) -> Tuple[float, float]:
        return self.x2, self.y2
