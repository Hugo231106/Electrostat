"""Shared scene base class for pygame simulations."""
from __future__ import annotations

import pygame

from simulation_config import SimulationConfig


class BaseScene:
    """Base class containing the main loop shared by all scenes."""

    BG_COLOR = (12, 12, 18)

    def __init__(self, screen: pygame.Surface, config: SimulationConfig) -> None:
        self.screen = screen
        self.config = config
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.running = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            self.running = False
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
        return True

    def update(self) -> None:
        """Hook for subclasses to update their simulation state."""

    def draw(self) -> None:
        """Hook for subclasses to render their scene."""

    def run(self) -> bool:
        """Main simulation loop. Returns False if the app should close."""

        while self.running:
            for event in pygame.event.get():
                should_continue = self.handle_event(event)
                if not should_continue:
                    return False

            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        return True
