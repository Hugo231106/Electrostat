"""Interactive 2D scene that lets users place electromagnetic entities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

from base_scene import BaseScene
from objects import CurrentWire, LineCharge, PointCharge
from simulation_config import FieldType


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

        self.panel_width = 260
        width, height = self.screen.get_size()
        self.panel_rect = pygame.Rect(width - self.panel_width, 0, self.panel_width, height)
        self.workspace_rect = pygame.Rect(0, 0, width - self.panel_width, height)

        self.charges: List[PointCharge] = []
        self.line_charges: List[LineCharge] = []
        self.currents: List[CurrentWire] = []

        self.selected_object: Optional[Tuple[str, int]] = None
        self.pending_start: Optional[Tuple[str, Tuple[int, int]]] = None

        self.type_options: Sequence[_TypeOption] = self._build_type_options()
        self.selected_type: str = self.type_options[0].identifier
        self.input_value: str = "1.0"
        self.input_active = False

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

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.panel_rect.collidepoint(event.pos):
                self._handle_panel_click(event.pos)
            else:
                self._handle_workspace_click(event.pos)
        elif event.type == pygame.KEYDOWN and self.input_active:
            self._handle_input_key(event)
        return True

    def _handle_panel_click(self, position: Tuple[int, int]) -> None:
        x, y = position
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
        else:
            self.input_active = False

        delete_rect = self._delete_button_rect()
        if delete_rect.collidepoint((x, y)) and self.selected_object is not None:
            self._delete_selected_object()

    def _handle_workspace_click(self, position: Tuple[int, int]) -> None:
        if self._select_object_at(position):
            return

        if self.selected_type == "point":
            self._place_point_charge(position)
        elif self.selected_type in {"line", "current"}:
            self._handle_segment_placement(position)

    def _handle_segment_placement(self, position: Tuple[int, int]) -> None:
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
        else:
            self.pending_start = (self.selected_type, position)
            self.selected_object = None

    def _place_point_charge(self, position: Tuple[int, int]) -> None:
        value = self._parse_value()
        self.charges.append(PointCharge(position[0], position[1], value))
        self.selected_object = ("charge", len(self.charges) - 1)
        self.pending_start = None

    def _parse_value(self) -> float:
        try:
            return float(self.input_value)
        except ValueError:
            return 0.0

    def _select_object_at(self, position: Tuple[int, int]) -> bool:
        # Charges
        for index, charge in enumerate(reversed(self.charges)):
            actual_index = len(self.charges) - 1 - index
            if self._point_within_radius(position, charge.position, 18):
                self.selected_object = ("charge", actual_index)
                return True

        # Line charges
        for index, line in enumerate(reversed(self.line_charges)):
            actual_index = len(self.line_charges) - 1 - index
            if self._point_near_segment(position, line.start, line.end, 12):
                self.selected_object = ("line", actual_index)
                return True

        # Currents
        for index, wire in enumerate(reversed(self.currents)):
            actual_index = len(self.currents) - 1 - index
            if self._point_near_segment(position, wire.start, wire.end, 12):
                self.selected_object = ("current", actual_index)
                return True

        return False

    def _handle_input_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_BACKSPACE:
            self.input_value = self.input_value[:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.input_active = False
        else:
            if len(event.unicode) == 1 and event.unicode in "0123456789.-":
                if event.unicode == "-" and self.input_value:
                    return
                if event.unicode == "." and "." in self.input_value:
                    return
                self.input_value += event.unicode

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
        # Grid for better spatial orientation
        grid_color = (24, 32, 68)
        for x in range(0, self.workspace_rect.width, 40):
            pygame.draw.line(
                self.screen,
                grid_color,
                (x, 0),
                (x, self.workspace_rect.height),
                1,
            )
        for y in range(0, self.workspace_rect.height, 40):
            pygame.draw.line(
                self.screen,
                grid_color,
                (0, y),
                (self.workspace_rect.width, y),
                1,
            )

        # Draw pending segment preview
        if self.pending_start and self.pending_start[0] == self.selected_type:
            start_pos = self.pending_start[1]
            end_pos = pygame.mouse.get_pos()
            if not self.panel_rect.collidepoint(end_pos):
                color = (240, 200, 80) if self.selected_type == "line" else (120, 230, 120)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

        for index, charge in enumerate(self.charges):
            selected = self.selected_object == ("charge", index)
            self._draw_charge(charge, selected)

        for index, line in enumerate(self.line_charges):
            selected = self.selected_object == ("line", index)
            self._draw_line_charge(line, selected)

        for index, wire in enumerate(self.currents):
            selected = self.selected_object == ("current", index)
            self._draw_current_wire(wire, selected)

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

        delete_rect = self._delete_button_rect(y_offset=input_rect.bottom + 24)
        button_color = (150, 70, 80) if self.selected_object else (90, 50, 60)
        pygame.draw.rect(self.screen, button_color, delete_rect, border_radius=8)
        delete_label = self.small_font.render("Supprimer sélection", True, (255, 235, 240))
        self.screen.blit(delete_label, delete_label.get_rect(center=delete_rect.center))

        instructions = [
            "Clic gauche : placer / sélectionner",
            "ESC : retour au menu",
        ]
        y_text = delete_rect.bottom + 40
        for line in instructions:
            surface = self.tiny_font.render(line, True, (200, 205, 220))
            self.screen.blit(surface, (self.panel_rect.left + 20, y_text))
            y_text += 22

    def _value_input_rect(self, y_offset: Optional[int] = None) -> pygame.Rect:
        if y_offset is None:
            y_offset = self.panel_rect.top + 240
        return pygame.Rect(self.panel_rect.left + 16, y_offset, self.panel_rect.width - 32, 42)

    def _delete_button_rect(self, y_offset: Optional[int] = None) -> pygame.Rect:
        if y_offset is None:
            y_offset = self.panel_rect.top + 320
        return pygame.Rect(self.panel_rect.left + 16, y_offset, self.panel_rect.width - 32, 44)

    def _draw_charge(self, charge: PointCharge, selected: bool) -> None:
        color = (220, 70, 80) if charge.q >= 0 else (70, 120, 220)
        position = (int(charge.x), int(charge.y))
        radius = 16 if not selected else 18
        pygame.draw.circle(self.screen, color, position, radius)
        pygame.draw.circle(self.screen, (250, 250, 255), position, radius, 2)
        value_surface = self.tiny_font.render(f"q={charge.q:.2f}", True, (230, 230, 240))
        text_rect = value_surface.get_rect(center=(position[0], position[1] + radius + 14))
        self.screen.blit(value_surface, text_rect)

    def _draw_line_charge(self, line: LineCharge, selected: bool) -> None:
        color = (240, 200, 80)
        width = 4 if selected else 3
        pygame.draw.line(self.screen, color, line.start, line.end, width)
        mid = ((line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2)
        value_surface = self.tiny_font.render(f"λ={line.linear_density:.2f}", True, (240, 235, 210))
        text_rect = value_surface.get_rect(center=(mid[0], mid[1] - 16))
        self.screen.blit(value_surface, text_rect)

    def _draw_current_wire(self, wire: CurrentWire, selected: bool) -> None:
        color = (120, 230, 120)
        width = 4 if selected else 3
        pygame.draw.line(self.screen, color, wire.start, wire.end, width)
        mid = ((wire.x1 + wire.x2) / 2, (wire.y1 + wire.y2) / 2)
        value_surface = self.tiny_font.render(f"I={wire.current:.2f}", True, (210, 240, 210))
        text_rect = value_surface.get_rect(center=(mid[0], mid[1] - 16))
        self.screen.blit(value_surface, text_rect)

    @staticmethod
    def _point_within_radius(point: Tuple[int, int], center: Tuple[float, float], radius: float) -> bool:
        return math.hypot(point[0] - center[0], point[1] - center[1]) <= radius

    @staticmethod
    def _point_near_segment(
        point: Tuple[int, int], start: Tuple[float, float], end: Tuple[float, float], threshold: float
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
