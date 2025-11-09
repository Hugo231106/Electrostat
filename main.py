"""Entry point for the Electrostat launcher using the pygame framework."""
from __future__ import annotations

import math

import pygame

from base_scene import BaseScene
from scene2d import Scene2D
from simulation_config import DimensionMode, SimulationConfig
from ui_main_menu import MainMenu


class Scene3D(BaseScene):
    """Placeholder for a 3D electromagnetic simulation."""

    BG_COLOR = (16, 32, 24)

    def __init__(self, screen: pygame.Surface, config: SimulationConfig) -> None:
        super().__init__(screen, config)
        self.angle = 0

    def update(self) -> None:
        self.angle = (self.angle + 0.5) % 360

    def draw(self) -> None:
        self.screen.fill(self.BG_COLOR)
        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2

        cube_color = (120, 255, 200)
        size = 180
        rect = pygame.Rect(center_x - size // 2, center_y - size // 2, size, size)
        offset = int(25 * math.sin(math.radians(self.angle)))
        top_rect = rect.move(offset, -offset)
        pygame.draw.rect(self.screen, (40, 120, 100), top_rect, border_radius=12)
        pygame.draw.rect(self.screen, cube_color, rect, border_radius=12)
        pygame.draw.line(self.screen, (20, 60, 50), rect.topleft, top_rect.topleft, 4)
        pygame.draw.line(self.screen, (20, 60, 50), rect.topright, top_rect.topright, 4)
        pygame.draw.line(self.screen, (20, 60, 50), rect.topleft, rect.topright, 4)
        pygame.draw.line(self.screen, (20, 60, 50), top_rect.topleft, top_rect.topright, 4)

        text = self.font.render(
            f"Simulation 3D • {self.config.field_type.label()}", True, (255, 255, 255)
        )
        self.screen.blit(text, text.get_rect(center=(center_x, 80)))

        hint = self.font.render("ESC pour revenir au menu", True, (220, 220, 230))
        self.screen.blit(hint, hint.get_rect(center=(center_x, self.screen.get_height() - 80)))


def run_simulation(screen: pygame.Surface, config: SimulationConfig) -> bool:
    """Instantiate and run the proper scene based on the configuration."""

    if config.dimension == DimensionMode.MODE_2D:
        scene: BaseScene = Scene2D(screen, config)
    else:
        scene = Scene3D(screen, config)
    return scene.run()


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("Electrostat Launcher")

    config = SimulationConfig()
    menu = MainMenu(screen, config)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                menu.screen = screen
                menu.on_resize(event.size)
                continue
            menu.handle_event(event)

        if not running:
            break

        menu.draw()
        pygame.display.flip()
        menu.tick()

        if menu.consume_start_request():
            continue_running = run_simulation(screen, config)
            if not continue_running:
                running = False
            else:
                screen = pygame.display.get_surface()
                menu.screen = screen
                menu.on_resize(screen.get_size())
                menu.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
