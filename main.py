"""Entry point for the Electrostat launcher using the pygame framework."""
from __future__ import annotations

import math

import pygame

from simulation_config import DimensionMode, SimulationConfig
from ui_main_menu import MainMenu


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


class Scene2D(BaseScene):
    """Very simple placeholder for a 2D electromagnetic simulation."""

    BG_COLOR = (18, 24, 54)

    def __init__(self, screen: pygame.Surface, config: SimulationConfig) -> None:
        super().__init__(screen, config)
        self.rotation = 0

    def update(self) -> None:
        self.rotation = (self.rotation + 1) % 360

    def draw(self) -> None:
        self.screen.fill(self.BG_COLOR)
        center = (self.screen.get_width() // 2, self.screen.get_height() // 2)

        radius = 120
        pygame.draw.circle(self.screen, (240, 150, 80), center, radius, 4)
        pygame.draw.circle(self.screen, (120, 200, 255), center, radius // 3)

        direction = (
            center[0] + int(math.cos(math.radians(self.rotation)) * radius),
            center[1] + int(math.sin(math.radians(self.rotation)) * radius),
        )
        pygame.draw.line(self.screen, (255, 255, 255), center, direction, 3)

        text = self.font.render(
            f"Simulation 2D • {self.config.field_type.label()}", True, (255, 255, 255)
        )
        self.screen.blit(text, text.get_rect(center=(center[0], 80)))

        hint = self.font.render("Appuyez sur ESC pour revenir au menu", True, (220, 220, 230))
        self.screen.blit(hint, hint.get_rect(center=(center[0], self.screen.get_height() - 80)))


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
    screen = pygame.display.set_mode((960, 640))
    pygame.display.set_caption("Electrostat Launcher")

    config = SimulationConfig()
    menu = MainMenu(screen, config)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
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
                menu.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
