# All constants for EyeTag

# Screen — detect native resolution at import time
import pygame
pygame.display.init()
_info = pygame.display.Info()
SCREEN_WIDTH  = _info.current_w if _info.current_w  > 0 else 800
SCREEN_HEIGHT = _info.current_h if _info.current_h > 0 else 600
FPS = 60
TITLE = "EyeTag"

# Colours (RGB)
BLACK = (0,   0,   0)
WHITE = (255, 255, 255)
NEON_GREEN = (0,   255, 100)
NEON_RED = (255, 50,  50)
YELLOW = (255, 220, 0)
NEON_BLUE = (255, 220, 0)

# Ship
SHIP_SIZE = 20

# Bullets
BULLET_SPEED  = 16
BULLET_RADIUS = 4
BULLET_RATE = 0.2  # seconds between shots

# Enemies 
ENEMY_SIZE = 18
ENEMY_BASE_SPEED = 1.5
ENEMY_SPAWN_RATE = 2.0  # seconds between spawns
KILL_THRESHOLD = 40   # how close before game over

# --- Waves ---
KILLS_PER_WAVE = 10
SPEED_INCREMENT = 0.3 
SPAWN_RATE_DECREASE = 0.15
MIN_SPAWN_RATE = 0.5 