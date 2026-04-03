# Main 

import pygame
import math
import random
import sys
from game.config import *
from game.entities import Ship, Bullet, Enemy
from game.renderer import Renderer
from gaze.estimator import GazeTracker


class Game:
    def __init__(self):
        pygame.init()

        # Create the window
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

        # Game objects
        self.ship = Ship()
        self.bullets = []
        self.enemies = []
        self.renderer = Renderer(self.screen)
        self.gaze = GazeTracker()

        # Game state
        self.score = 0
        self.wave = 1
        self.kills_this_wave = 0
        self.game_over = False
        self.running = True

        # Timers - track how much time has passed since last spawn/shot
        self.bullet_timer = 0
        self.enemy_timer = 0

        # Current wave stats
        self.enemy_speed = ENEMY_BASE_SPEED
        self.enemy_spawn_rate = ENEMY_SPAWN_RATE

        # Gaze position
        self.gaze_x = SCREEN_WIDTH // 2
        self.gaze_y = SCREEN_HEIGHT // 2


    def run_calibration(self):
            # Show instructions screen first
            waiting = True
            while waiting:
                self.renderer.draw_calibration_screen()
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False

            # Run the actual EyeTrax calibration
            self.gaze.calibrate()

    def update_gaze(self):
            coords = self.gaze.get_coords()
            if coords is not None:
                self.gaze_x, self.gaze_y = coords

    def update_bullets(self, dt):
        self.bullet_timer += dt

        if self.bullet_timer >= BULLET_RATE:
            self.bullet_timer = 0

            # Calculate angle from ship center to gaze position
            dx = self.gaze_x - (SCREEN_WIDTH // 2)
            dy = self.gaze_y - (SCREEN_HEIGHT // 2)
            angle = math.atan2(dy, dx)

            # Spawn a new bullet at the ship center
            self.bullets.append(Bullet(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, angle))

        # Update all bullets, remove inactive ones
        for bullet in self.bullets:
            bullet.update()
        self.bullets = [b for b in self.bullets if b.active]

    def update_enemies(self, dt):
        self.enemy_timer += dt

        if self.enemy_timer >= self.enemy_spawn_rate:
            self.enemy_timer = 0
            self.enemies.append(Enemy(self.enemy_speed))

        # Update all enemies, check if any reached the ship
        for enemy in self.enemies:
            enemy.update()

        # If any enemy deactivated by reaching center - game over
        for enemy in self.enemies:
            if not enemy.active:
                self.game_over = True

        self.enemies = [e for e in self.enemies if e.active]

    def check_collisions(self):
        for bullet in self.bullets:
            for enemy in self.enemies:
                # Calculate distance between bullet and enemy
                dx = bullet.x - enemy.x
                dy = bullet.y - enemy.y
                dist = math.sqrt(dx**2 + dy**2)

                # If close enough - both die
                if dist < enemy.size + bullet.radius:
                    bullet.active = False
                    enemy.active = False
                    self.score += 10
                    self.kills_this_wave += 1

        # Clean up destroyed objects
        self.bullets = [b for b in self.bullets if b.active]
        self.enemies = [e for e in self.enemies if e.active]


    def check_wave(self):
        if self.kills_this_wave >= KILLS_PER_WAVE:
            self.wave += 1
            self.kills_this_wave = 0

            # Each wave: enemies faster and spawn more frequently
            self.enemy_speed = ENEMY_BASE_SPEED + (self.wave * SPEED_INCREMENT)
            self.enemy_spawn_rate = max(
                MIN_SPAWN_RATE,
                ENEMY_SPAWN_RATE - (self.wave * SPAWN_RATE_DECREASE)
            )


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                # R to restart on game over
                if event.key == pygame.K_r and self.game_over:
                    self.restart()
                # ESC to quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False


    def restart(self):
        self.bullets = []
        self.enemies = []
        self.score = 0
        self.wave = 1
        self.kills_this_wave = 0
        self.game_over = False
        self.bullet_timer = 0
        self.enemy_timer = 0
        self.enemy_speed = ENEMY_BASE_SPEED
        self.enemy_spawn_rate = ENEMY_SPAWN_RATE


    def run(self):
        while self.running:
            # dt = time since last frame in seconds
            dt = self.clock.tick(FPS) / 1000.0

            self.handle_events()

            if not self.game_over:
                self.update_gaze()
                self.update_bullets(dt)
                self.update_enemies(dt)
                self.check_collisions()
                self.check_wave()

            # Draw everything
            self.renderer.draw_background()
            self.ship.draw(self.screen)

            for bullet in self.bullets:
                bullet.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)

            self.renderer.draw_gaze_cursor(self.gaze_x, self.gaze_y)
            self.renderer.draw_hud(self.score, self.wave, self.kills_this_wave)

            if self.game_over:
                self.renderer.draw_game_over(self.score, self.wave)

            # Flip pushes everything to the screen at once
            pygame.display.flip()

        # Cleanup
        self.gaze.release()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run_calibration()
    game.run()