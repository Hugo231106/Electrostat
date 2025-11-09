"""Computation helpers for electric and magnetic fields in the 2D scene."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from objects import CurrentWire, LineCharge, PointCharge

K_E = 1.0  # Coulomb-like constant (arbitrary units for the visualisation)
MU_OVER_4PI = 1.0  # Simplified constant for the Biot-Savart approximation
EPSILON = 1e-6
SEGMENTS_PER_LINE = 20

Vector = Tuple[float, float]


def _normalize(vec: Vector) -> Vector:
    mag = math.hypot(vec[0], vec[1])
    if mag < EPSILON:
        return 0.0, 0.0
    return vec[0] / mag, vec[1] / mag


def compute_E_at_point(
    point: Vector,
    charges: Sequence[PointCharge],
    line_charges: Sequence[LineCharge],
) -> Vector:
    """Compute the electric field (Ex, Ey) generated at ``point``."""

    px, py = point
    field_x = 0.0
    field_y = 0.0

    for charge in charges:
        dx = px - charge.x
        dy = py - charge.y
        r_sq = dx * dx + dy * dy
        if r_sq < EPSILON:
            continue
        r_mag = math.sqrt(r_sq)
        intensity = K_E * charge.q / r_sq
        field_x += intensity * dx / r_mag
        field_y += intensity * dy / r_mag

    for line in line_charges:
        segments = max(4, SEGMENTS_PER_LINE)
        length = math.hypot(line.x2 - line.x1, line.y2 - line.y1)
        if length < EPSILON:
            continue
        dx_line = (line.x2 - line.x1) / segments
        dy_line = (line.y2 - line.y1) / segments
        dq = line.linear_density * length / segments
        for i in range(segments):
            sx = line.x1 + (i + 0.5) * dx_line
            sy = line.y1 + (i + 0.5) * dy_line
            dx = px - sx
            dy = py - sy
            r_sq = dx * dx + dy * dy
            if r_sq < EPSILON:
                continue
            r_mag = math.sqrt(r_sq)
            intensity = K_E * dq / r_sq
            field_x += intensity * dx / r_mag
            field_y += intensity * dy / r_mag

    return field_x, field_y


def compute_B_at_point(point: Vector, currents: Sequence[CurrentWire]) -> Vector:
    """Compute a simplified magnetic field in the plane from current segments."""

    px, py = point
    field_x = 0.0
    field_y = 0.0

    for wire in currents:
        segments = max(4, SEGMENTS_PER_LINE)
        length = math.hypot(wire.x2 - wire.x1, wire.y2 - wire.y1)
        if length < EPSILON:
            continue
        dx_line = (wire.x2 - wire.x1) / segments
        dy_line = (wire.y2 - wire.y1) / segments
        dl_mag = math.hypot(dx_line, dy_line)
        if dl_mag < EPSILON:
            continue
        for i in range(segments):
            sx = wire.x1 + (i + 0.5) * dx_line
            sy = wire.y1 + (i + 0.5) * dy_line
            dx = px - sx
            dy = py - sy
            r_sq = dx * dx + dy * dy
            if r_sq < EPSILON:
                continue
            # Tangential direction (perpendicular in plane)
            tangent_x = -dy
            tangent_y = dx
            tangent = _normalize((tangent_x, tangent_y))
            strength = MU_OVER_4PI * wire.current * dl_mag / (r_sq)
            field_x += strength * tangent[0]
            field_y += strength * tangent[1]

    return field_x, field_y


def compute_potential_at_point(
    point: Vector,
    charges: Sequence[PointCharge],
    line_charges: Sequence[LineCharge],
) -> float:
    """Compute the electrostatic potential at a given point."""

    px, py = point
    potential = 0.0

    for charge in charges:
        dx = px - charge.x
        dy = py - charge.y
        r = math.hypot(dx, dy)
        if r < EPSILON:
            continue
        potential += K_E * charge.q / r

    for line in line_charges:
        segments = max(4, SEGMENTS_PER_LINE)
        length = math.hypot(line.x2 - line.x1, line.y2 - line.y1)
        if length < EPSILON:
            continue
        dx_line = (line.x2 - line.x1) / segments
        dy_line = (line.y2 - line.y1) / segments
        dq = line.linear_density * length / segments
        for i in range(segments):
            sx = line.x1 + (i + 0.5) * dx_line
            sy = line.y1 + (i + 0.5) * dy_line
            dx = px - sx
            dy = py - sy
            r = math.hypot(dx, dy)
            if r < EPSILON:
                continue
            potential += K_E * dq / r

    return potential


def compute_field_magnitude(vec: Vector) -> float:
    return math.hypot(vec[0], vec[1])


def marching_squares(
    xs: Sequence[float],
    ys: Sequence[float],
    values: Sequence[Sequence[float]],
    levels: Iterable[float],
) -> List[List[Vector]]:
    """Generate contour polylines using a simple marching squares implementation."""

    contours: List[List[Vector]] = []
    nx = len(xs)
    ny = len(ys)
    if nx < 2 or ny < 2:
        return contours

    def interpolate(p1: Vector, p2: Vector, v1: float, v2: float, level: float) -> Vector:
        if abs(level - v1) < EPSILON:
            return p1
        if abs(level - v2) < EPSILON:
            return p2
        if abs(v1 - v2) < EPSILON:
            return p1
        t = (level - v1) / (v2 - v1)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    for level in levels:
        segments: List[List[Vector]] = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v1 = values[j][i]
                v2 = values[j][i + 1]
                v3 = values[j + 1][i + 1]
                v4 = values[j + 1][i]
                idx = 0
                if v1 > level:
                    idx |= 1
                if v2 > level:
                    idx |= 2
                if v3 > level:
                    idx |= 4
                if v4 > level:
                    idx |= 8
                if idx == 0 or idx == 15:
                    continue

                x1 = xs[i]
                x2 = xs[i + 1]
                y1 = ys[j]
                y2 = ys[j + 1]
                p1 = (x1, y1)
                p2 = (x2, y1)
                p3 = (x2, y2)
                p4 = (x1, y2)

                edge_points: List[Vector] = []
                if idx in (1, 14, 13, 2, 11, 4, 7, 8):
                    pass
                # Evaluate edges according to the marching squares cases
                if idx in (1, 5, 13, 9):
                    edge_points.append(interpolate(p1, p2, v1, v2, level))
                if idx in (3, 7, 11, 10):
                    edge_points.append(interpolate(p2, p3, v2, v3, level))
                if idx in (2, 6, 7, 10, 14):
                    edge_points.append(interpolate(p2, p3, v2, v3, level))
                if idx in (4, 6, 12, 14):
                    edge_points.append(interpolate(p3, p4, v3, v4, level))
                if idx in (8, 9, 12, 13):
                    edge_points.append(interpolate(p4, p1, v4, v1, level))
                if idx in (1, 3, 5, 7, 9, 11):
                    edge_points.append(interpolate(p4, p1, v4, v1, level))

                if len(edge_points) >= 2:
                    # Pair points into small segments
                    for k in range(0, len(edge_points) - 1, 2):
                        segments.append([edge_points[k], edge_points[k + 1]])
        contours.extend(segments)

    return contours
