"""Main menu interface built with the pygame library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pygame

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

    def __init__(self, screen: pygame.Surface, config: SimulationConfig) -> None:
        self.screen = screen
        self.config = config
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 56)
        self.clock = pygame.time.Clock()

        self.dimension_buttons: List[MenuButton] = []
        self.field_buttons: List[MenuButton] = []
        self.action_buttons: List[MenuButton] = []

        self.container_rect = pygame.Rect(0, 0, 640, 420)

        self._rebuild_layout()

        self._start_requested = False

    def _build_buttons(self) -> None:
        self.dimension_buttons.clear()
        self.field_buttons.clear()
        self.action_buttons.clear()

        column_spacing = 32
        button_spacing = 18
        button_height = 58
        column_width = (self.container_rect.width - column_spacing * 3) // 2

        left_column_x = self.container_rect.left + column_spacing
        right_column_x = left_column_x + column_width + column_spacing
        buttons_top = self.container_rect.top + 140

        # Dimension buttons (left column)
        for index, dimension in enumerate((DimensionMode.MODE_2D, DimensionMode.MODE_3D)):
            rect = pygame.Rect(
                left_column_x,
                buttons_top + index * (button_height + button_spacing),
                column_width,
                button_height,
            )
            self.dimension_buttons.append(
                MenuButton(dimension.label(), rect, "dimension", value=dimension)
            )

        # Field type buttons (right column)
        field_types = [
            FieldType.ELECTROSTATIC,
            FieldType.MAGNETOSTATIC,
            FieldType.COUPLED,
        ]
        for index, field_type in enumerate(field_types):
            rect = pygame.Rect(
                right_column_x,
                buttons_top + index * (button_height + button_spacing),
                column_width,
                button_height,
            )
            self.field_buttons.append(MenuButton(field_type.label(), rect, "field", value=field_type))

        # Validate button centered at bottom of container
        validate_width = column_width
        validate_rect = pygame.Rect(
            self.container_rect.centerx - validate_width // 2,
            self.container_rect.bottom - button_height - 36,
            validate_width,
            button_height,
        )
        self.action_buttons.append(MenuButton("Lancer la simulation", validate_rect, "validate"))

        self._refresh_selections()

    def _rebuild_layout(self) -> None:
        width, height = self.screen.get_size()
        available_width = max(360, width - 120)
        container_width = min(780, available_width)
        available_height = max(320, height - 200)
        container_height = min(480, available_height)

        top_margin = max(60, (height - container_height) // 2)
        self.container_rect = pygame.Rect(
            (width - container_width) // 2,
            top_margin,
            container_width,
            container_height,
        )

        self._build_buttons()

    def on_resize(self, size: tuple[int, int]) -> None:
        """Update layout when the main window is resized."""

        self._rebuild_layout()

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
        if button.group == "validate":
            color = (140, 100, 220)
            if hovered:
                color = (160, 120, 240)
        else:
            if hovered:
                color = self.BTN_COLOR_HOVER
            if button.is_selected:
                color = self.BTN_COLOR_SELECTED

        pygame.draw.rect(self.screen, color, button.rect, border_radius=12)
        label_surface = self.font.render(button.label, True, self.TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button.rect.center)
        self.screen.blit(label_surface, label_rect)

    def draw(self) -> None:
        """Render the complete main menu on the screen."""

        self.screen.fill(self.BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        # Background accents
        overlay_rect = self.container_rect.inflate(80, 60)
        pygame.draw.rect(self.screen, (24, 24, 32), overlay_rect, border_radius=24)
        pygame.draw.rect(self.screen, (36, 36, 52), self.container_rect, border_radius=24)

        title_surface = self.title_font.render("Electrostat Sim", True, self.TEXT_COLOR)
        title_rect = title_surface.get_rect(midtop=(self.screen.get_width() // 2, 48))
        self.screen.blit(title_surface, title_rect)

        subtitle_surface = self.small_font.render(
            "Choisissez le mode et configurez votre simulation", True, self.SUBTITLE_COLOR
        )
        subtitle_rect = subtitle_surface.get_rect(midtop=(self.screen.get_width() // 2, 104))
        self.screen.blit(subtitle_surface, subtitle_rect)

        # Section headers inside the container
        dimension_header = self.small_font.render("Mode de rendu", True, self.SUBTITLE_COLOR)
        dimension_pos = (self.dimension_buttons[0].rect.left, self.container_rect.top + 96)
        self.screen.blit(dimension_header, dimension_pos)

        field_header = self.small_font.render("Type de simulation", True, self.SUBTITLE_COLOR)
        self.screen.blit(field_header, (self.field_buttons[0].rect.left, self.container_rect.top + 96))

        # Draw buttons
        for button in self.dimension_buttons + self.field_buttons + self.action_buttons:
            self._draw_button(button, mouse_pos)

        # Summary chips at bottom of container
        summary_texts = [
            f"Mode : {self.config.dimension.label()}",
            f"Simulation : {self.config.field_type.label()}",
        ]
        chip_y = self.container_rect.bottom - 140
        chip_x = self.container_rect.left + 40
        for text in summary_texts:
            surface = self.small_font.render(text, True, self.TEXT_COLOR)
            padding = pygame.Vector2(18, 10)
            rect = surface.get_rect(topleft=(chip_x, chip_y))
            rect.inflate_ip(padding.x, padding.y)
            pygame.draw.rect(self.screen, (64, 64, 92), rect, border_radius=12)
            self.screen.blit(surface, surface.get_rect(center=rect.center))
            chip_x = rect.right + 16

    def handle_event(self, event: pygame.event.Event) -> None:
        """React to user inputs and update the configuration accordingly."""

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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
