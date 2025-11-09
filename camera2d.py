"""Simple 2D camera helper for pygame scenes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple


@dataclass
class Camera2D:
    """Represent a 2D camera with translation and zoom."""

    MIN_ZOOM: ClassVar[float] = 0.05
    MAX_ZOOM: ClassVar[float] = 120.0

    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0

    def world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates into screen coordinates."""

        sx = (x - self.offset_x) * self.zoom
        sy = (y - self.offset_y) * self.zoom
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """Convert screen coordinates back to world coordinates."""

        if self.zoom == 0:
            return self.offset_x, self.offset_y
        wx = sx / self.zoom + self.offset_x
        wy = sy / self.zoom + self.offset_y
        return wx, wy

    def zoom_at(self, focus_x: float, focus_y: float, zoom_factor: float) -> None:
        """Zoom while keeping a given focus point stationary on screen."""

        if zoom_factor <= 0:
            return
        world_focus_x, world_focus_y = self.screen_to_world(focus_x, focus_y)
        self.zoom *= zoom_factor
        self.zoom = max(self.MIN_ZOOM, min(self.zoom, self.MAX_ZOOM))
        self.offset_x = world_focus_x - focus_x / self.zoom
        self.offset_y = world_focus_y - focus_y / self.zoom
