# objects in the game

import pygame
import math
import random
from game.config import *


# --- SHIP ---
class Ship:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.size = SHIP_SIZE

    def draw(self, screen):
        points = [
            (self.x,              self.y - self.size),  # top
            (self.x + self.size,  self.y),              # right
            (self.x,              self.y + self.size),  # bottom
            (self.x - self.size,  self.y),              # left
        ]
        pygame.draw.polygon(screen, YELLOW, points)
        pygame.draw.polygon(screen, WHITE, points, 2)   # outline


# --- BULLET ---
class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.radius = BULLET_RADIUS
        self.active = True  # False when it goes off screen or hits enemy

        # Calculate x and y speed from the angle
        self.dx = math.cos(angle) * BULLET_SPEED
        self.dy = math.sin(angle) * BULLET_SPEED

    def update(self):
        # Move the bullet each frame
        self.x += self.dx
        self.y += self.dy

        # Deactivate if it leaves the screen
        if (self.x < 0 or self.x > SCREEN_WIDTH or
            self.y < 0 or self.y > SCREEN_HEIGHT):
            self.active = False

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.radius)


# --- ENEMY ---
class Enemy:
    def __init__(self, speed):
        self.size = ENEMY_SIZE
        self.active = True
        self.speed = speed

        # Spawn randomly on one of the four screen edges
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge == "top":
            self.x = random.randint(0, SCREEN_WIDTH)
            self.y = 0
        elif edge == "bottom":
            self.x = random.randint(0, SCREEN_WIDTH)
            self.y = SCREEN_HEIGHT
        elif edge == "left":
            self.x = 0
            self.y = random.randint(0, SCREEN_HEIGHT)
        elif edge == "right":
            self.x = SCREEN_WIDTH
            self.y = random.randint(0, SCREEN_HEIGHT)

        # Calculate angle toward the center (where the ship is)
        dx = (SCREEN_WIDTH // 2) - self.x
        dy = (SCREEN_HEIGHT // 2) - self.y
        angle = math.atan2(dy, dx)

        # Set velocity toward center
        self.dx = math.cos(angle) * self.speed
        self.dy = math.sin(angle) * self.speed

    def update(self):
        self.x += self.dx
        self.y += self.dy

        # Deactivate if it reaches the ship
        dist = math.sqrt((self.x - SCREEN_WIDTH // 2) ** 2 +
                         (self.y - SCREEN_HEIGHT // 2) ** 2)
        if dist < KILL_THRESHOLD:
            self.active = False

    def draw(self, screen):
        # Draw as a neon red triangle pointing toward center
        points = [
            (self.x,             self.y - self.size),
            (self.x + self.size, self.y + self.size),
            (self.x - self.size, self.y + self.size),
        ]
        pygame.draw.polygon(screen, NEON_RED, points)
        pygame.draw.polygon(screen, WHITE, points, 2)

