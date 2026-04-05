# Shows calibration dots, collects training data, trains the regression model

import pygame
import numpy as np
import time
import sys
from sklearn.linear_model import Ridge
from gaze.features import extract_features
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT

DURATION = 2.0
ALPHA_VAL = 0.2

CALIBRATION_POINTS = [
    (0.1, 0.1),   # top left
    (0.5, 0.1),   # top center
    (0.9, 0.1),   # top right
    (0.1, 0.5),   # middle left
    (0.5, 0.5),   # center
    (0.9, 0.5),   # middle right
    (0.1, 0.9),   # bottom left
    (0.5, 0.9),   # bottom center
    (0.9, 0.9),   # bottom right
]

class Calibrator:
    def __init__(self, screen, cap):
        self.screen = screen   # pygame screen to draw dots on
        self.cap = cap         # opencv webcam capture
        self.model = None      # will hold the trained Ridge model

        self.X = []  # input features - one vector
        self.Y = []  # output labels - one (x,y) 

        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20)

    def draw_dot(self, x, y, progress, countdown):
        self.screen.fill((0, 0, 0))

        # Outer ring — shrinks as time passes showing progress
        max_radius = 30
        current_radius = int(max_radius * (1 - progress))
        pygame.draw.circle(self.screen, (100, 100, 100),
                          (x, y), max_radius, 2)

        # Inner dot — always visible
        pygame.draw.circle(self.screen, (0, 255, 100), (x, y), 8)

        # Shrinking ring shows how long left to stare
        if current_radius > 8:
            pygame.draw.circle(self.screen, (255, 255, 255),
                              (x, y), current_radius, 2)

        # Instructions at top

        pygame.display.flip()



    def collect_dot(self, screen_x, screen_y):
        # Collect frames for DURATION seconds while user stares at dot
        samples_collected = 0
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            progress = elapsed / DURATION

            # Done collecting for this dot
            if elapsed >= DURATION:
                break

            # Draw the dot with progress animation
            self.draw_dot(screen_x, screen_y, progress, DURATION - elapsed)

            # Handle quit/escape
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            # Read webcam frame
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Extract features from this frame
            features, is_blinking = extract_features(frame)

            # Only collect if face detected and eyes open
            if features is not None and not is_blinking:
                self.X.append(features)
                self.Y.append([screen_x, screen_y])
                samples_collected += 1

        print(f"Dot at ({screen_x}, {screen_y}) — {samples_collected} samples")



    #TRAINING    
    def train(self):
        X_array = np.array(self.X)  #(frame, 34)
        Y_array = np.array(self.Y)  #(frame, 2)

        print(f"Training on {len(X_array)} samples...")

        # Create and train the Ridge regression model
        self.model = Ridge(alpha=ALPHA_VAL)
        self.model.fit(X_array, Y_array)

        print("Training complete")
        return self.model
    


    def run(self):
        # Show brief instruction screen
        self.screen.fill((0, 0, 0))
        lines = [
            "CALIBRATION",
            "",
            "9 dots will appear one at a time.",
            "Stare at each dot until it disappears.",
            "Keep your head still.",
            "",
            "PRESS SPACE TO BEGIN",
        ]
        y_pos = 150
        for line in lines:
            if line == "CALIBRATION":
                surf = self.font_large.render(line, True, (0, 180, 255))
            elif line == "PRESS SPACE TO BEGIN":
                surf = self.font_small.render(line, True, (0, 255, 100))
            else:
                surf = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (SCREEN_WIDTH//2 - surf.get_width()//2, y_pos))
            y_pos += 40

        pygame.display.flip()

        # Wait for spacebar
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_SPACE:
                        waiting = False

        time.sleep(0.5)

        # Show countdown before first dot
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))
            countdown = self.font_large.render(str(i), True, (255, 255, 255))
            self.screen.blit(countdown,
                (SCREEN_WIDTH//2 - countdown.get_width()//2,
                 SCREEN_HEIGHT//2 - countdown.get_height()//2))
            pygame.display.flip()
            end = time.time() + 1
            while time.time() < end:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                time.sleep(0.01)

        # Loop through all 9 calibration points
        for i, (fx, fy) in enumerate(CALIBRATION_POINTS):
            # Convert fractions to actual screen pixels
            screen_x = int(fx * SCREEN_WIDTH)
            screen_y = int(fy * SCREEN_HEIGHT)

            # Show static dot before collection starts — gives eye time to travel there
            self.draw_dot(screen_x, screen_y, 1.0, 0)
            end = time.time() + 0.8
            while time.time() < end:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                time.sleep(0.01)

            # Collect data for this dot
            self.collect_dot(screen_x, screen_y)

        # Train the model on all collected data
        model = self.train()

        # Show completion screen
        self.screen.fill((0, 0, 0))
        done = self.font_large.render("CALIBRATION COMPLETE", True, (0, 255, 100))
        self.screen.blit(done,
            (SCREEN_WIDTH//2 - done.get_width()//2,
             SCREEN_HEIGHT//2 - done.get_height()//2))
        pygame.display.flip()
        time.sleep(1.5)

        return model
