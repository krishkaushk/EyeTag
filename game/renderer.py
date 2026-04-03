# Handles all drawing

import pygame
from game.config import *


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20)

    def draw_background(self):
        # Fill with black each frame (clears previous frame)
        self.screen.fill(BLACK)

        # Draw a subtle grid for retro arcade feel
        for x in range(0, SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, (20, 20, 40), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, (20, 20, 40), (0, y), (SCREEN_WIDTH, y))

    def draw_hud(self, score, wave, kills_this_wave):
        # Score - top left
        score_text = self.font_large.render(f"SCORE  {score}", True, NEON_GREEN)
        self.screen.blit(score_text, (20, 15))

        # Wave - top right
        wave_text = self.font_large.render(f"WAVE  {wave}", True, NEON_BLUE)
        self.screen.blit(wave_text, (SCREEN_WIDTH - wave_text.get_width() - 20, 15))

        # Kill progress bar toward next wave - bottom
        bar_width = 300
        bar_x = (SCREEN_WIDTH - bar_width) // 2
        bar_y = SCREEN_HEIGHT - 30
        progress = min(kills_this_wave / KILLS_PER_WAVE, 1.0)

        # Background bar
        pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_width, 12))
        # Filled portion
        pygame.draw.rect(self.screen, NEON_GREEN, (bar_x, bar_y, int(bar_width * progress), 12))
        # Outline
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, 12), 1)

        # Label above bar
        label = self.font_small.render(f"WAVE PROGRESS", True, WHITE)
        self.screen.blit(label, (bar_x + bar_width // 2 - label.get_width() // 2, bar_y - 22))

    def draw_gaze_cursor(self, gaze_x, gaze_y):
        # Draw a crosshair where the player is looking
        size = 12
        pygame.draw.line(self.screen, NEON_GREEN,
                         (gaze_x - size, gaze_y), (gaze_x + size, gaze_y), 2)
        pygame.draw.line(self.screen, NEON_GREEN,
                         (gaze_x, gaze_y - size), (gaze_x, gaze_y + size), 2)
        pygame.draw.circle(self.screen, NEON_GREEN, (gaze_x, gaze_y), size, 1)

    def draw_game_over(self, score, wave):
        # Dark overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Game over text
        title = self.font_large.render("GAME  OVER", True, NEON_RED)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 80))

        # Final score
        final_score = self.font_large.render(f"SCORE  {score}", True, YELLOW)
        self.screen.blit(final_score, (SCREEN_WIDTH // 2 - final_score.get_width() // 2,
                                        SCREEN_HEIGHT // 2 - 20))

        # Wave reached
        final_wave = self.font_small.render(f"REACHED WAVE  {wave}", True, WHITE)
        self.screen.blit(final_wave, (SCREEN_WIDTH // 2 - final_wave.get_width() // 2,
                                       SCREEN_HEIGHT // 2 + 40))

        # Restart prompt
        restart = self.font_small.render("PRESS  R  TO  RESTART", True, NEON_GREEN)
        self.screen.blit(restart, (SCREEN_WIDTH // 2 - restart.get_width() // 2,
                                    SCREEN_HEIGHT // 2 + 90))

    def draw_calibration_screen(self):
        self.screen.fill(BLACK)
        lines = [
            "EYETAG",
            "",
            "Before the game starts,",
            "you need to calibrate the eye tracker.",
            "",
            "Follow the dots on screen with your eyes.",
            "Keep your head still during calibration.",
            "",
            "PRESS SPACE TO BEGIN",
        ]
        y = 160
        for line in lines:
            if line == "EYETAG":
                surf = self.font_large.render(line, True, NEON_BLUE)
            elif line == "PRESS SPACE TO BEGIN":
                surf = self.font_small.render(line, True, NEON_GREEN)
            else:
                surf = self.font_small.render(line, True, WHITE)
            self.screen.blit(surf, (SCREEN_WIDTH // 2 - surf.get_width() // 2, y))
            y += 40