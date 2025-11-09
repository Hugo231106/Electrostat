"""Main menu interface built with the iGame library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import igame

from simulation_config import DimensionMode, FieldType, SimulationConfig


@dataclass
class MenuButton:
    """Simple structure describing a clickable area and its value."""

    label: str
    rect: Any
    group: str
    value: Any = None
    is_selected: bool = False


class MainMenu:
    """Handle rendering and interaction for the launcher main menu."""

    BG_COLOR = (18, 18, 22)
    BTN_COLOR = (55, 55, 65)
    BTN_COLOR_HOVER = (90, 90, 110)
    BTN_COLOR_SELECTED = (120, 80, 200)
    TEXT_COLOR = (235, 235, 245)
    SUBTITLE_COLOR = (160, 160, 190)

    def __init__(self, screen: igame.Surface, config: SimulationConfig) -> None:
        self.screen = screen
        self.config = config
        self.font = igame.font.Font(None, 28)
        self.small_font = igame.font.Font(None, 22)
        self.title_font = igame.font.Font(None, 52)
        self.clock = igame.time.Clock()

        self.dimension_buttons: List[MenuButton] = []
        self.field_buttons: List[MenuButton] = []
        self.action_buttons: List[MenuButton] = []

        self._build_buttons()

        self._start_requested = False

    def _build_buttons(self) -> None:
        width, height = self.screen.get_size()
        center_x = width // 2

        button_width = 260
        button_height = 56
        button_spacing = 16

        # Dimension buttons
        top_y = height // 2 - 150
        for index, dimension in enumerate((DimensionMode.MODE_2D, DimensionMode.MODE_3D)):
            rect = igame.Rect(
                center_x - button_width // 2,
                top_y + index * (button_height + button_spacing),
                button_width,
                button_height,
            )
            button = MenuButton(dimension.label(), rect, "dimension", value=dimension)
            self.dimension_buttons.append(button)

        # Field type buttons
        top_y += 170
        field_types = [
            FieldType.ELECTROSTATIC,
            FieldType.MAGNETOSTATIC,
            FieldType.COUPLED,
        ]
        for index, field_type in enumerate(field_types):
            rect = igame.Rect(
                center_x - button_width // 2,
                top_y + index * (button_height + button_spacing),
                button_width,
                button_height,
            )
            button = MenuButton(field_type.label(), rect, "field", value=field_type)
            self.field_buttons.append(button)

        # Validate button
        validate_rect = igame.Rect(
            center_x - button_width // 2,
            height - 120,
            button_width,
            button_height,
        )
        self.action_buttons.append(MenuButton("Valider", validate_rect, "validate"))

        self._refresh_selections()

    def _refresh_selections(self) -> None:
        for button in self.dimension_buttons:
            button.is_selected = button.value == self.config.dimension
        for button in self.field_buttons:
            button.is_selected = button.value == self.config.field_type

    def reset(self) -> None:
        """Clear transient state (start requests) when returning to the menu."""

        self._start_requested = False
        self._refresh_selections()

    def _draw_button(self, button: MenuButton, mouse_pos: Any) -> None:
        hovered = button.rect.collidepoint(mouse_pos)
        color = self.BTN_COLOR
        if hovered:
            color = self.BTN_COLOR_HOVER
        if button.is_selected:
            color = self.BTN_COLOR_SELECTED

        igame.draw.rect(self.screen, color, button.rect, border_radius=12)
        label_surface = self.font.render(button.label, True, self.TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button.rect.center)
        self.screen.blit(label_surface, label_rect)

    def draw(self) -> None:
        """Render the complete main menu on the screen."""

        self.screen.fill(self.BG_COLOR)
        mouse_pos = igame.mouse.get_pos()

        title_surface = self.title_font.render("Electrostat Sim", True, self.TEXT_COLOR)
        title_rect = title_surface.get_rect(center=(self.screen.get_width() // 2, 80))
        self.screen.blit(title_surface, title_rect)

        subtitle_surface = self.small_font.render(
            "Choisissez le mode et la simulation à lancer", True, self.SUBTITLE_COLOR
        )
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen.get_width() // 2, 120))
        self.screen.blit(subtitle_surface, subtitle_rect)

        group_titles = [
            ("Mode de rendu", self.dimension_buttons[0].rect.top - 40),
            ("Type de simulation", self.field_buttons[0].rect.top - 40),
        ]
        for text, y in group_titles:
            label_surface = self.small_font.render(text, True, self.SUBTITLE_COLOR)
            label_rect = label_surface.get_rect(center=(self.screen.get_width() // 2, y))
            self.screen.blit(label_surface, label_rect)

        for button in self.dimension_buttons + self.field_buttons + self.action_buttons:
            self._draw_button(button, mouse_pos)

        summary_surface = self.small_font.render(self.config.describe(), True, self.TEXT_COLOR)
        summary_rect = summary_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 60))
        self.screen.blit(summary_surface, summary_rect)

    def handle_event(self, event: igame.event.Event) -> None:
        """React to user inputs and update the configuration accordingly."""

        if event.type == igame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for button in self.dimension_buttons:
                if button.rect.collidepoint(mouse_pos):
                    self.config.set_dimension(button.value)
                    self._refresh_selections()
                    return

            for button in self.field_buttons:
                if button.rect.collidepoint(mouse_pos):
                    self.config.set_field_type(button.value)
                    self._refresh_selections()
                    return

            for button in self.action_buttons:
                if button.rect.collidepoint(mouse_pos) and button.group == "validate":
                    self._start_requested = True
                    return

    def consume_start_request(self) -> bool:
        """Return whether the user pressed the validate button since the last check."""

        if self._start_requested:
            self._start_requested = False
            return True
        return False

    def tick(self, fps: int = 60) -> None:
        """Regulate the frame-rate of the menu."""

        self.clock.tick(fps)
