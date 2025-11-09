"""Interactive 2D scene that lets users place electromagnetic entities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import pygame

from base_scene import BaseScene
from camera2d import Camera2D
from field import (
    compute_B_at_point,
    compute_E_at_point,
    compute_field_magnitude,
    compute_potential_at_point,
    marching_squares,
)
from objects import CurrentWire, LineCharge, PointCharge
from simulation_config import FieldType, SimulationConfig


@dataclass
class _TypeOption:
    label: str
    identifier: str
    description: str


class Scene2D(BaseScene):
    """Handle rendering, events and simulation logic for the 2D mode."""

    BG_COLOR = (18, 24, 54)
    WORKSPACE_BG = (10, 12, 26)
    PANEL_BG = (44, 46, 58)
    PANEL_BORDER = (84, 88, 110)

    def __init__(self, screen: pygame.Surface, config: SimulationConfig) -> None:
        super().__init__(screen, config)
        self.small_font = pygame.font.Font(None, 22)
        self.tiny_font = pygame.font.Font(None, 18)

        self.panel_width = 280
        width, height = self.screen.get_size()
        self.panel_rect = pygame.Rect(width - self.panel_width, 0, self.panel_width, height)
        self.workspace_rect = pygame.Rect(0, 0, width - self.panel_width, height)

        self.camera = Camera2D()
        self.is_panning = False

        self.charges: List[PointCharge] = []
        self.line_charges: List[LineCharge] = []
        self.currents: List[CurrentWire] = []

        self.selected_object: Optional[Tuple[str, int]] = None
        self.pending_start: Optional[Tuple[str, Tuple[float, float]]] = None

        self.type_options: Sequence[_TypeOption] = self._build_type_options()
        self.selected_type: str = self.type_options[0].identifier
        self.input_value: str = "1.0"
        self.input_active = False

        self.selected_input_value: str = ""
        self.selected_input_active = False

        self.show_field_lines_mode = "none"  # none, E, B, both
        self.show_potentials = False
        self.show_help = False

        self.field_lines_E: List[List[Tuple[float, float]]] = []
        self.field_lines_B: List[List[Tuple[float, float]]] = []
        self.potential_contours: List[List[Tuple[float, float]]] = []
        self.field_dirty = True

    def _build_type_options(self) -> Sequence[_TypeOption]:
        if self.config.field_type == FieldType.ELECTROSTATIC:
            return (
                _TypeOption("Charge ponctuelle", "point", "Cliquez pour placer une charge."),
                _TypeOption("Ligne de charge", "line", "Deux clics définissent la ligne."),
            )
        if self.config.field_type == FieldType.MAGNETOSTATIC:
            return (
                _TypeOption("Fil de courant", "current", "Deux clics définissent le fil."),
            )
        return (
            _TypeOption("Charge ponctuelle", "point", "Cliquez pour placer une charge."),
            _TypeOption("Ligne de charge", "line", "Deux clics définissent la ligne."),
            _TypeOption("Fil de courant", "current", "Deux clics définissent le fil."),
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not super().handle_event(event):
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.panel_rect.collidepoint(event.pos):
                    self._handle_panel_click(event.pos)
                elif self.workspace_rect.collidepoint(event.pos):
                    self._handle_workspace_click(event.pos)
            elif event.button == 3 and self.workspace_rect.collidepoint(event.pos):
                self.is_panning = True
            elif event.button in (4, 5) and self.workspace_rect.collidepoint(event.pos):
                self._handle_zoom(event.pos, 1.1 if event.button == 4 else 1 / 1.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                self.is_panning = False
        elif event.type == pygame.MOUSEMOTION and self.is_panning:
            dx, dy = event.rel
            self.camera.offset_x -= dx / self.camera.zoom
            self.camera.offset_y -= dy / self.camera.zoom
            self.field_dirty = True
        elif event.type == pygame.KEYDOWN:
            if self.input_active or self.selected_input_active:
                self._handle_input_key(event)
            elif event.key == pygame.K_DELETE:
                self._delete_selected_object()
            elif event.key == pygame.K_l:
                self._cycle_field_line_mode()
            elif event.key == pygame.K_p:
                self.show_potentials = not self.show_potentials
                self.field_dirty = True
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
        return True

    def _handle_panel_click(self, position: Tuple[int, int]) -> None:
        x, y = position
        self.input_active = False
        self.selected_input_active = False
        self.pending_start = None
        option_height = 52
        option_spacing = 12
        current_y = 120
        for option in self.type_options:
            option_rect = pygame.Rect(
                self.panel_rect.left + 16,
                current_y,
                self.panel_rect.width - 32,
                option_height,
            )
            if option_rect.collidepoint((x, y)):
                self.selected_type = option.identifier
                self.pending_start = None
                return
            current_y += option_height + option_spacing

        input_rect = self._value_input_rect()
        if input_rect.collidepoint((x, y)):
            self.input_active = True
            return

        delete_rect = self._delete_button_rect()
        if delete_rect.collidepoint((x, y)) and self.selected_object is not None:
            self._delete_selected_object()
            return

        if self.selected_object is not None:
            selected_rect = self._selected_value_input_rect()
            if selected_rect.collidepoint((x, y)):
                self.selected_input_active = True

    def _handle_workspace_click(self, position: Tuple[int, int]) -> None:
        self.input_active = False
        self.selected_input_active = False
        world_pos = self._screen_to_world((position[0] - self.workspace_rect.left, position[1] - self.workspace_rect.top))

        if self._select_object_at(world_pos):
            self.pending_start = None
            self._populate_selected_input()
            return

        if self.selected_type == "point":
            self._place_point_charge(world_pos)
        elif self.selected_type in {"line", "current"}:
            self._handle_segment_placement(world_pos)

    def _handle_segment_placement(self, position: Tuple[float, float]) -> None:
        if self.pending_start and self.pending_start[0] == self.selected_type:
            start_pos = self.pending_start[1]
            value = self._parse_value()
            if self.selected_type == "line":
                self.line_charges.append(
                    LineCharge(start_pos[0], start_pos[1], position[0], position[1], value)
                )
                self.selected_object = ("line", len(self.line_charges) - 1)
            else:
                self.currents.append(
                    CurrentWire(start_pos[0], start_pos[1], position[0], position[1], value)
                )
                self.selected_object = ("current", len(self.currents) - 1)
            self.pending_start = None
            self._populate_selected_input()
            self._mark_field_dirty()
        else:
            self.pending_start = (self.selected_type, position)
            self.selected_object = None

    def _place_point_charge(self, position: Tuple[float, float]) -> None:
        value = self._parse_value()
        self.charges.append(PointCharge(position[0], position[1], value))
        self.selected_object = ("charge", len(self.charges) - 1)
        self.pending_start = None
        self._populate_selected_input()
        self._mark_field_dirty()

    def _handle_zoom(self, position: Tuple[int, int], zoom_factor: float) -> None:
        local_x = position[0] - self.workspace_rect.left
        local_y = position[1] - self.workspace_rect.top
        self.camera.zoom_at(local_x, local_y, zoom_factor)
        self._mark_field_dirty()

    def _cycle_field_line_mode(self) -> None:
        modes = ["none", "E", "B", "both"]
        try:
            index = modes.index(self.show_field_lines_mode)
        except ValueError:
            index = 0
        self.show_field_lines_mode = modes[(index + 1) % len(modes)]
        self.field_dirty = True

    def _parse_value(self) -> float:
        try:
            return float(self.input_value)
        except ValueError:
            return 0.0

    def _populate_selected_input(self) -> None:
        if not self.selected_object:
            self.selected_input_value = ""
            self.selected_input_active = False
            return
        kind, index = self.selected_object
        obj = self._get_object(kind, index)
        if obj is None:
            self.selected_object = None
            self.selected_input_value = ""
            self.selected_input_active = False
            return
        if kind == "charge":
            value = obj.q
        elif kind == "line":
            value = obj.linear_density
        else:
            value = obj.current
        self.selected_input_value = f"{value:.4g}"

    def _apply_selected_input_value(self) -> None:
        if not self.selected_object:
            return
        try:
            new_value = float(self.selected_input_value)
        except ValueError:
            return
        kind, index = self.selected_object
        obj = self._get_object(kind, index)
        if obj is None:
            return
        if kind == "charge":
            if obj.q == new_value:
                return
            obj.q = new_value
        elif kind == "line":
            if obj.linear_density == new_value:
                return
            obj.linear_density = new_value
        else:
            if obj.current == new_value:
                return
            obj.current = new_value
        self._mark_field_dirty()

    def _select_object_at(self, position: Tuple[float, float]) -> bool:
        threshold_point = self._charge_radius_world()
        threshold_line = self._line_selection_threshold_world()

        # Charges
        for index, charge in enumerate(reversed(self.charges)):
            actual_index = len(self.charges) - 1 - index
            if self._point_within_radius(position, charge.position, threshold_point):
                self.selected_object = ("charge", actual_index)
                return True

        # Line charges
        for index, line in enumerate(reversed(self.line_charges)):
            actual_index = len(self.line_charges) - 1 - index
            if self._point_near_segment(position, line.start, line.end, threshold_line):
                self.selected_object = ("line", actual_index)
                return True

        # Currents
        for index, wire in enumerate(reversed(self.currents)):
            actual_index = len(self.currents) - 1 - index
            if self._point_near_segment(position, wire.start, wire.end, threshold_line):
                self.selected_object = ("current", actual_index)
                return True

        return False

    def _handle_input_key(self, event: pygame.event.Event) -> None:
        target_value = self.selected_input_value if self.selected_input_active else self.input_value

        if event.key == pygame.K_BACKSPACE:
            target_value = target_value[:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.selected_input_active:
                self.selected_input_active = False
                self._apply_selected_input_value()
            else:
                self.input_active = False
        else:
            if len(event.unicode) == 1 and event.unicode in "0123456789.-":
                if event.unicode == "-" and target_value:
                    return
                if event.unicode == "." and "." in target_value:
                    return
                target_value += event.unicode

        if self.selected_input_active:
            self.selected_input_value = target_value
        else:
            self.input_value = target_value

    def update(self) -> None:
        pass

    def draw(self) -> None:
        self.screen.fill(self.BG_COLOR)
        pygame.draw.rect(self.screen, self.WORKSPACE_BG, self.workspace_rect)
        pygame.draw.rect(self.screen, self.PANEL_BG, self.panel_rect)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, self.panel_rect, 2)

        self._draw_workspace()
        self._draw_panel()

    def _draw_workspace(self) -> None:
        self._draw_grid()
        self._ensure_field_visuals()
        self._draw_field_visuals()
        self._draw_pending_segment_preview()

        for index, charge in enumerate(self.charges):
            selected = self.selected_object == ("charge", index)
            self._draw_charge(charge, selected)

        for index, line in enumerate(self.line_charges):
            selected = self.selected_object == ("line", index)
            self._draw_line_charge(line, selected)

        for index, wire in enumerate(self.currents):
            selected = self.selected_object == ("current", index)
            self._draw_current_wire(wire, selected)

        self._draw_hud()

    def _draw_field_visuals(self) -> None:
        if self.show_field_lines_mode in {"E", "both"}:
            color = (255, 220, 160)
            width = max(1, int(2 * self.camera.zoom))
            for line in self.field_lines_E:
                if len(line) < 2:
                    continue
                points = [self._round_point(self._world_to_screen(p)) for p in line]
                pygame.draw.lines(self.screen, color, False, points, width)

        if self.show_field_lines_mode in {"B", "both"}:
            color = (140, 230, 255)
            width = max(1, int(2 * self.camera.zoom))
            for line in self.field_lines_B:
                if len(line) < 2:
                    continue
                points = [self._round_point(self._world_to_screen(p)) for p in line]
                pygame.draw.lines(self.screen, color, False, points, width)

        if self.show_potentials and self.potential_contours:
            color = (150, 120, 240)
            for segment in self.potential_contours:
                if len(segment) < 2:
                    continue
                start = self._round_point(self._world_to_screen(segment[0]))
                end = self._round_point(self._world_to_screen(segment[1]))
                pygame.draw.line(self.screen, color, start, end, 1)

    def _draw_pending_segment_preview(self) -> None:
        if not self.pending_start or self.pending_start[0] != self.selected_type:
            return
        mouse_pos = pygame.mouse.get_pos()
        if not self.workspace_rect.collidepoint(mouse_pos):
            return
        start = self.pending_start[1]
        end_world = self._screen_to_world(
            (mouse_pos[0] - self.workspace_rect.left, mouse_pos[1] - self.workspace_rect.top)
        )
        color = (240, 200, 80) if self.selected_type == "line" else (120, 230, 120)
        start_screen = self._round_point(self._world_to_screen(start))
        end_screen = self._round_point(self._world_to_screen(end_world))
        width = max(1, int(3 * self.camera.zoom))
        pygame.draw.line(self.screen, color, start_screen, end_screen, width)

    def _draw_grid(self) -> None:
        spacing = self._determine_grid_spacing()
        x_min, y_min, x_max, y_max = self._visible_world_rect(margin=spacing * 4)
        start_x = math.floor(x_min / spacing) * spacing
        start_y = math.floor(y_min / spacing) * spacing

        vertical_color = (40, 46, 78)
        axis_color = (110, 120, 180)

        x = start_x
        while x <= x_max:
            start = self._world_to_screen((x, y_min))
            end = self._world_to_screen((x, y_max))
            color = axis_color if abs(x) < spacing * 0.3 else vertical_color
            pygame.draw.line(
                self.screen,
                color,
                self._round_point(start),
                self._round_point(end),
                2 if color == axis_color else 1,
            )
            x += spacing

        y = start_y
        while y <= y_max:
            start = self._world_to_screen((x_min, y))
            end = self._world_to_screen((x_max, y))
            color = axis_color if abs(y) < spacing * 0.3 else vertical_color
            pygame.draw.line(
                self.screen,
                color,
                self._round_point(start),
                self._round_point(end),
                2 if color == axis_color else 1,
            )
            y += spacing

    def _determine_grid_spacing(self) -> float:
        min_spacing_px = 50
        spacing = 1.0
        zoom = self.camera.zoom
        while spacing * zoom < min_spacing_px:
            spacing *= 2.0
        while spacing * zoom > min_spacing_px * 4 and spacing > 1.0:
            spacing /= 2.0
        return spacing

    def _ensure_field_visuals(self) -> None:
        if not self.field_dirty:
            return
        bounds = self._visible_world_rect(margin=200 / self.camera.zoom)
        self.field_lines_E = []
        self.field_lines_B = []
        self.potential_contours = []

        if self.show_field_lines_mode in {"E", "both"} and self.charges:
            self.field_lines_E = self._generate_e_field_lines(bounds)
        if self.show_field_lines_mode in {"B", "both"} and self.currents:
            self.field_lines_B = self._generate_b_field_lines(bounds)
        if self.show_potentials and self.config.field_type != FieldType.MAGNETOSTATIC:
            self.potential_contours = self._generate_potential_contours(bounds)
        self.field_dirty = False

    def _generate_e_field_lines(self, bounds: Tuple[float, float, float, float]) -> List[List[Tuple[float, float]]]:
        lines: List[List[Tuple[float, float]]] = []
        seed_radius = self._charge_radius_world() * 1.4
        seeds_per_charge = 12
        field_func = lambda p: compute_E_at_point(p, self.charges, self.line_charges)
        for charge in self.charges:
            for i in range(seeds_per_charge):
                angle = 2 * math.pi * i / seeds_per_charge
                start = (charge.x + math.cos(angle) * seed_radius, charge.y + math.sin(angle) * seed_radius)
                if not (bounds[0] <= start[0] <= bounds[2] and bounds[1] <= start[1] <= bounds[3]):
                    continue
                for direction in (1, -1):
                    line = self._trace_field_line(start, field_func, bounds, avoid="E", direction=direction)
                    if len(line) > 1:
                        lines.append(line)
        return lines

    def _generate_b_field_lines(self, bounds: Tuple[float, float, float, float]) -> List[List[Tuple[float, float]]]:
        lines: List[List[Tuple[float, float]]] = []
        field_func = lambda p: compute_B_at_point(p, self.currents)
        for wire in self.currents:
            dx = wire.x2 - wire.x1
            dy = wire.y2 - wire.y1
            length = math.hypot(dx, dy)
            segments = max(4, int(length / max(self._field_line_step_world(), 1.0)))
            if segments <= 0:
                continue
            direction_vec = self._normalize_vector((dx, dy))
            normal = self._normalize_vector((-direction_vec[1], direction_vec[0]))
            if normal == (0.0, 0.0):
                normal = (0.0, 1.0)
            offset = self._line_selection_threshold_world() * 0.8
            for i in range(segments):
                t = (i + 0.5) / segments
                base = (wire.x1 + dx * t, wire.y1 + dy * t)
                for sign in (1, -1):
                    start = (base[0] + normal[0] * offset * sign, base[1] + normal[1] * offset * sign)
                    if not (bounds[0] <= start[0] <= bounds[2] and bounds[1] <= start[1] <= bounds[3]):
                        continue
                    line = self._trace_field_line(start, field_func, bounds, avoid="B", direction=sign)
                    if len(line) > 1:
                        lines.append(line)
        return lines

    def _generate_potential_contours(
        self, bounds: Tuple[float, float, float, float]
    ) -> List[List[Tuple[float, float]]]:
        x_min, y_min, x_max, y_max = bounds
        cols = 40
        rows = 30
        xs = [x_min + (x_max - x_min) * i / (cols - 1) for i in range(cols)]
        ys = [y_min + (y_max - y_min) * j / (rows - 1) for j in range(rows)]
        values: List[List[float]] = []
        for y in ys:
            row = []
            for x in xs:
                row.append(compute_potential_at_point((x, y), self.charges, self.line_charges))
            values.append(row)
        flat_values = [v for row in values for v in row]
        if not flat_values:
            return []
        min_v = min(flat_values)
        max_v = max(flat_values)
        if math.isclose(min_v, max_v, rel_tol=1e-6, abs_tol=1e-6):
            return []
        num_levels = 7
        levels = [min_v + (i + 1) * (max_v - min_v) / (num_levels + 1) for i in range(num_levels)]
        return marching_squares(xs, ys, values, levels)

    def _trace_field_line(
        self,
        start: Tuple[float, float],
        field_func: Callable[[Tuple[float, float]], Tuple[float, float]],
        bounds: Tuple[float, float, float, float],
        avoid: str,
        direction: int,
    ) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = [start]
        current = start
        step = self._field_line_step_world()
        max_steps = 600
        for _ in range(max_steps):
            field_vec = field_func(current)
            magnitude = compute_field_magnitude(field_vec)
            if magnitude < 1e-4:
                break
            norm = (field_vec[0] / magnitude, field_vec[1] / magnitude)
            if direction < 0:
                norm = (-norm[0], -norm[1])
            next_point = (current[0] + norm[0] * step, current[1] + norm[1] * step)
            if not (bounds[0] <= next_point[0] <= bounds[2] and bounds[1] <= next_point[1] <= bounds[3]):
                break
            if avoid == "E" and self._near_e_source(next_point):
                break
            if avoid == "B" and self._near_b_source(next_point):
                break
            points.append(next_point)
            current = next_point
        return points

    def _field_line_step_world(self) -> float:
        return max(4.0, 30.0 / max(self.camera.zoom, 0.2))

    def _near_e_source(self, point: Tuple[float, float]) -> bool:
        for charge in self.charges:
            if math.hypot(point[0] - charge.x, point[1] - charge.y) < self._charge_radius_world() * 0.9:
                return True
        for line in self.line_charges:
            if self._distance_to_segment(point, line.start, line.end) < self._line_selection_threshold_world() * 0.6:
                return True
        return False

    def _near_b_source(self, point: Tuple[float, float]) -> bool:
        for wire in self.currents:
            if self._distance_to_segment(point, wire.start, wire.end) < self._line_selection_threshold_world() * 0.6:
                return True
        return False

    def _draw_panel(self) -> None:
        title = self.font.render("Outils", True, (245, 245, 250))
        self.screen.blit(title, (self.panel_rect.left + 20, 24))

        mode_label = self.small_font.render(self.config.field_type.label(), True, (190, 195, 210))
        self.screen.blit(mode_label, (self.panel_rect.left + 20, 60))

        option_height = 52
        option_spacing = 12
        y = 120
        for option in self.type_options:
            option_rect = pygame.Rect(
                self.panel_rect.left + 16,
                y,
                self.panel_rect.width - 32,
                option_height,
            )
            hovered = option_rect.collidepoint(pygame.mouse.get_pos())
            color = (70, 72, 92)
            if option.identifier == self.selected_type:
                color = (110, 98, 210)
            elif hovered:
                color = (86, 88, 118)
            pygame.draw.rect(self.screen, color, option_rect, border_radius=10)

            label_surface = self.small_font.render(option.label, True, (240, 240, 245))
            self.screen.blit(label_surface, label_surface.get_rect(center=(option_rect.centerx, option_rect.centery - 8)))

            desc_surface = self.tiny_font.render(option.description, True, (200, 205, 220))
            desc_rect = desc_surface.get_rect(center=(option_rect.centerx, option_rect.centery + 14))
            self.screen.blit(desc_surface, desc_rect)

            y += option_height + option_spacing

        value_label = self.small_font.render("Valeur", True, (220, 225, 235))
        self.screen.blit(value_label, (self.panel_rect.left + 20, y + 10))

        input_rect = self._value_input_rect(y_offset=y + 34)
        pygame.draw.rect(
            self.screen,
            (90, 92, 116) if self.input_active else (70, 72, 96),
            input_rect,
            border_radius=8,
        )
        value_surface = self.small_font.render(self.input_value or "0", True, (250, 250, 255))
        value_rect = value_surface.get_rect(midleft=(input_rect.left + 12, input_rect.centery))
        self.screen.blit(value_surface, value_rect)
        y_cursor = input_rect.bottom + 24

        if self.selected_object:
            selected_label = self.small_font.render("Objet sélectionné", True, (225, 230, 245))
            self.screen.blit(selected_label, (self.panel_rect.left + 20, y_cursor))
            y_cursor += 26
            param_name = self._selected_parameter_name()
            param_surface = self.tiny_font.render(f"Paramètre ({param_name})", True, (200, 205, 220))
            self.screen.blit(param_surface, (self.panel_rect.left + 20, y_cursor))
            y_cursor += 20
            selected_rect = self._selected_value_input_rect(y_offset=y_cursor)
            pygame.draw.rect(
                self.screen,
                (110, 112, 150) if self.selected_input_active else (82, 84, 118),
                selected_rect,
                border_radius=8,
            )
            selected_surface = self.small_font.render(
                self.selected_input_value or "0", True, (250, 250, 255)
            )
            selected_rect_text = selected_surface.get_rect(
                midleft=(selected_rect.left + 12, selected_rect.centery)
            )
            self.screen.blit(selected_surface, selected_rect_text)
            y_cursor = selected_rect.bottom + 16

        delete_rect = self._delete_button_rect(y_offset=y_cursor)
        button_color = (150, 70, 80) if self.selected_object else (90, 50, 60)
        pygame.draw.rect(self.screen, button_color, delete_rect, border_radius=8)
        delete_label = self.small_font.render("Supprimer sélection", True, (255, 235, 240))
        self.screen.blit(delete_label, delete_label.get_rect(center=delete_rect.center))
        y_cursor = delete_rect.bottom + 28

        field_lines_label = self.small_font.render(
            f"Lignes de champ : {self._field_mode_label()}", True, (210, 215, 230)
        )
        self.screen.blit(field_lines_label, (self.panel_rect.left + 20, y_cursor))
        y_cursor += 26
        potentials_label = self.small_font.render(
            f"Équipotentielles : {'activées' if self.show_potentials else 'désactivées'}",
            True,
            (210, 215, 230),
        )
        self.screen.blit(potentials_label, (self.panel_rect.left + 20, y_cursor))
        y_cursor += 32

        instructions = [
            "Clic gauche : sélectionner / créer",
            "Clic droit + glisser : déplacer la vue",
            "Molette : zoom",
            "Touche L : cycle lignes de champ",
            "Touche P : afficher équipotentielles",
            "Suppr : supprimer la sélection",
            "Touche H : aide",
            "Échap : retour au menu",
        ]
        for line in instructions:
            surface = self.tiny_font.render(line, True, (200, 205, 220))
            self.screen.blit(surface, (self.panel_rect.left + 20, y_cursor))
            y_cursor += 20

    def _draw_hud(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hud_lines: List[str] = []
        if self.workspace_rect.collidepoint(mouse_pos):
            world_pos = self._screen_to_world(
                (mouse_pos[0] - self.workspace_rect.left, mouse_pos[1] - self.workspace_rect.top)
            )
            hud_lines.append(f"Position : x={world_pos[0]:.2f}, y={world_pos[1]:.2f}")
            e_vec = compute_E_at_point(world_pos, self.charges, self.line_charges)
            e_mag = compute_field_magnitude(e_vec)
            hud_lines.append(f"|E|={e_mag:.3f}  Ex={e_vec[0]:.3f}  Ey={e_vec[1]:.3f}")
            b_vec = compute_B_at_point(world_pos, self.currents)
            b_mag = compute_field_magnitude(b_vec)
            hud_lines.append(f"|B|={b_mag:.3f}  Bx={b_vec[0]:.3f}  By={b_vec[1]:.3f}")
            if self.config.field_type != FieldType.MAGNETOSTATIC:
                potential = compute_potential_at_point(world_pos, self.charges, self.line_charges)
                hud_lines.append(f"V={potential:.3f}")
        else:
            hud_lines.append("Curseur hors de la scène")

        if hud_lines:
            surfaces = [self.tiny_font.render(line, True, (230, 235, 245)) for line in hud_lines]
            max_width = max(surface.get_width() for surface in surfaces) if surfaces else 0
            total_height = sum(surface.get_height() for surface in surfaces) + (len(surfaces) - 1) * 4
            padding = 10
            hud_rect = pygame.Rect(
                self.workspace_rect.left + 12,
                self.workspace_rect.top + 12,
                max_width + padding * 2,
                total_height + padding * 2,
            )
            pygame.draw.rect(self.screen, (20, 24, 40), hud_rect, border_radius=8)
            pygame.draw.rect(self.screen, (70, 80, 120), hud_rect, 1, border_radius=8)
            y = hud_rect.top + padding
            for surface in surfaces:
                self.screen.blit(surface, (hud_rect.left + padding, y))
                y += surface.get_height() + 4

        if self.show_help:
            self._draw_help_overlay()

    def _draw_help_overlay(self) -> None:
        help_lines = [
            "Aide rapide :",
            "• Clic gauche : sélectionner ou créer l'objet sélectionné",
            "• Clic droit + déplacement : déplacer la caméra",
            "• Molette : zoomer/dézoomer (autour du curseur)",
            "• Touche L : alterner l'affichage des lignes de champ",
            "• Touche P : afficher ou masquer les équipotentielles",
            "• Touche Suppr : supprimer l'objet sélectionné",
            "Les lignes de champ suivent E ou B selon le mode.",
            "Les équipotentielles représentent les niveaux du potentiel V.",
        ]
        surfaces = [self.tiny_font.render(line, True, (240, 240, 250)) for line in help_lines]
        max_width = max(surface.get_width() for surface in surfaces)
        total_height = sum(surface.get_height() for surface in surfaces) + (len(surfaces) - 1) * 4
        padding = 14
        overlay_rect = pygame.Rect(
            self.workspace_rect.left + 20,
            self.workspace_rect.top + 140,
            max_width + padding * 2,
            total_height + padding * 2,
        )
        pygame.draw.rect(self.screen, (30, 34, 60), overlay_rect, border_radius=10)
        pygame.draw.rect(self.screen, (90, 100, 150), overlay_rect, 1, border_radius=10)
        y = overlay_rect.top + padding
        for surface in surfaces:
            self.screen.blit(surface, (overlay_rect.left + padding, y))
            y += surface.get_height() + 4
    def _value_input_rect(self, y_offset: Optional[int] = None) -> pygame.Rect:
        if y_offset is None:
            y_offset = self.panel_rect.top + 240
        return pygame.Rect(self.panel_rect.left + 16, y_offset, self.panel_rect.width - 32, 42)

    def _delete_button_rect(self, y_offset: Optional[int] = None) -> pygame.Rect:
        if y_offset is None:
            y_offset = self.panel_rect.top + 320
        return pygame.Rect(self.panel_rect.left + 16, y_offset, self.panel_rect.width - 32, 44)

    def _selected_value_input_rect(self, y_offset: Optional[int] = None) -> pygame.Rect:
        if y_offset is None:
            y_offset = self.panel_rect.top + 380
        return pygame.Rect(self.panel_rect.left + 16, y_offset, self.panel_rect.width - 32, 42)

    def _draw_charge(self, charge: PointCharge, selected: bool) -> None:
        color = (220, 70, 80) if charge.q >= 0 else (70, 120, 220)
        screen_pos = self._world_to_screen(charge.position)
        radius_world = self._charge_radius_world() * (1.1 if selected else 1.0)
        radius = max(2, int(radius_world * self.camera.zoom))
        pygame.draw.circle(self.screen, color, self._round_point(screen_pos), radius)
        pygame.draw.circle(self.screen, (250, 250, 255), self._round_point(screen_pos), radius, 2)
        value_surface = self.tiny_font.render(f"q={charge.q:.2f}", True, (230, 230, 240))
        text_rect = value_surface.get_rect(
            center=(int(screen_pos[0]), int(screen_pos[1] + radius + 14))
        )
        self.screen.blit(value_surface, text_rect)

    def _draw_line_charge(self, line: LineCharge, selected: bool) -> None:
        color = (240, 200, 80)
        width = max(1, int((4 if selected else 3) * self.camera.zoom))
        start_screen = self._round_point(self._world_to_screen(line.start))
        end_screen = self._round_point(self._world_to_screen(line.end))
        pygame.draw.line(self.screen, color, start_screen, end_screen, width)
        mid_world = ((line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2)
        mid_screen = self._world_to_screen(mid_world)
        value_surface = self.tiny_font.render(f"λ={line.linear_density:.2f}", True, (240, 235, 210))
        text_rect = value_surface.get_rect(
            center=(int(mid_screen[0]), int(mid_screen[1] - 16))
        )
        self.screen.blit(value_surface, text_rect)

    def _draw_current_wire(self, wire: CurrentWire, selected: bool) -> None:
        color = (120, 230, 120)
        width = max(1, int((4 if selected else 3) * self.camera.zoom))
        start_screen = self._round_point(self._world_to_screen(wire.start))
        end_screen = self._round_point(self._world_to_screen(wire.end))
        pygame.draw.line(self.screen, color, start_screen, end_screen, width)
        mid_world = ((wire.x1 + wire.x2) / 2, (wire.y1 + wire.y2) / 2)
        mid_screen = self._world_to_screen(mid_world)
        value_surface = self.tiny_font.render(f"I={wire.current:.2f}", True, (210, 240, 210))
        text_rect = value_surface.get_rect(
            center=(int(mid_screen[0]), int(mid_screen[1] - 16))
        )
        self.screen.blit(value_surface, text_rect)

    @staticmethod
    def _point_within_radius(point: Tuple[float, float], center: Tuple[float, float], radius: float) -> bool:
        return math.hypot(point[0] - center[0], point[1] - center[1]) <= radius

    @staticmethod
    def _point_near_segment(
        point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float], threshold: float
    ) -> bool:
        if start == end:
            return Scene2D._point_within_radius(point, start, threshold)
        px, py = point
        sx, sy = start
        ex, ey = end
        line_mag_sq = (ex - sx) ** 2 + (ey - sy) ** 2
        if line_mag_sq == 0:
            return Scene2D._point_within_radius(point, start, threshold)
        t = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / line_mag_sq
        t = max(0.0, min(1.0, t))
        closest = (sx + t * (ex - sx), sy + t * (ey - sy))
        return Scene2D._point_within_radius(point, closest, threshold)

    @staticmethod
    def _distance_to_segment(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
        if start == end:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        px, py = point
        sx, sy = start
        ex, ey = end
        line_mag_sq = (ex - sx) ** 2 + (ey - sy) ** 2
        if line_mag_sq == 0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        t = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / line_mag_sq
        t = max(0.0, min(1.0, t))
        closest = (sx + t * (ex - sx), sy + t * (ey - sy))
        return math.hypot(point[0] - closest[0], point[1] - closest[1])

    def _delete_selected_object(self) -> None:
        if not self.selected_object:
            return
        kind, index = self.selected_object
        if kind == "charge" and 0 <= index < len(self.charges):
            del self.charges[index]
        elif kind == "line" and 0 <= index < len(self.line_charges):
            del self.line_charges[index]
        elif kind == "current" and 0 <= index < len(self.currents):
            del self.currents[index]
        self.selected_object = None
        self.selected_input_value = ""
        self.selected_input_active = False
        self._mark_field_dirty()

    def _get_object(self, kind: str, index: int):
        if kind == "charge" and 0 <= index < len(self.charges):
            return self.charges[index]
        if kind == "line" and 0 <= index < len(self.line_charges):
            return self.line_charges[index]
        if kind == "current" and 0 <= index < len(self.currents):
            return self.currents[index]
        return None

    def _mark_field_dirty(self) -> None:
        self.field_dirty = True

    @staticmethod
    def _charge_radius_world() -> float:
        return 20.0

    @staticmethod
    def _line_selection_threshold_world() -> float:
        return 14.0

    @staticmethod
    def _normalize_vector(vec: Tuple[float, float]) -> Tuple[float, float]:
        length = math.hypot(vec[0], vec[1])
        if length < 1e-8:
            return (0.0, 0.0)
        return vec[0] / length, vec[1] / length

    def _selected_parameter_name(self) -> str:
        if not self.selected_object:
            return "-"
        kind, _ = self.selected_object
        if kind == "charge":
            return "q"
        if kind == "line":
            return "λ"
        if kind == "current":
            return "I"
        return "-"

    def _field_mode_label(self) -> str:
        mapping = {
            "none": "désactivées",
            "E": "E",
            "B": "B",
            "both": "E + B",
        }
        return mapping.get(self.show_field_lines_mode, "désactivées")

    def _world_to_screen(self, position: Tuple[float, float]) -> Tuple[float, float]:
        sx, sy = self.camera.world_to_screen(position[0], position[1])
        return self.workspace_rect.left + sx, self.workspace_rect.top + sy

    def _screen_to_world(self, position: Tuple[float, float]) -> Tuple[float, float]:
        return self.camera.screen_to_world(position[0], position[1])

    def _visible_world_rect(self, margin: float = 0.0) -> Tuple[float, float, float, float]:
        top_left = self.camera.screen_to_world(-margin, -margin)
        bottom_right = self.camera.screen_to_world(
            self.workspace_rect.width + margin, self.workspace_rect.height + margin
        )
        return (*top_left, *bottom_right)

    @staticmethod
    def _round_point(point: Tuple[float, float]) -> Tuple[int, int]:
        return int(point[0]), int(point[1])
