# Shows calibration dots, collects training data, trains the neural network

import math
import pygame
import numpy as np
import time
import sys
import torch
import torch.nn as nn
from gaze.model import GazeNet
from gaze.features import extract_features
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT

EPOCH_RANGE    = 1500
LEARNING       = 0.01
WEIGHTED_DECAY = 0

DURATION       = 1.0   # seconds to collect per dot
PRE_PAUSE      = 0.6   # seconds to show dot before collecting (eye settles)

CALIBRATION_POINTS = [
    (fx, fy)
    for fy in [0.1, 0.3, 0.5, 0.7, 0.9]
    for fx in [0.1, 0.3, 0.5, 0.7, 0.9]
]  # 25 points, row by row


class Calibrator:
    def __init__(self, screen, cap):
        self.screen = screen
        self.cap    = cap
        self.model  = None

        self.X = []
        self.Y = []

        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20)


    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_dot(self, x, y, progress):
        # progress 0→1 over DURATION: shrinking ring shows time left
        self.screen.fill((0, 0, 0))

        pygame.draw.circle(self.screen, (60, 60, 60), (x, y), 28, 2)

        remaining_r = int(28 * (1 - progress))
        if remaining_r > 9:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), remaining_r, 2)

        pygame.draw.circle(self.screen, (0, 255, 100), (x, y), 9)

        pygame.display.flip()

    def _draw_dot_static(self, x, y):
        # Plain dot with no ring — shown during pre-pause so eye can travel there
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (60, 60, 60), (x, y), 28, 2)
        pygame.draw.circle(self.screen, (0, 200, 80), (x, y), 9)
        pygame.display.flip()


    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_dot(self, screen_x, screen_y):
        # Pre-pause: show static dot, let eye settle, collect nothing
        self._draw_dot_static(screen_x, screen_y)
        end = time.time() + PRE_PAUSE
        while time.time() < end:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            time.sleep(0.01)

        # Collection: shrinking ring, gather features
        samples = 0
        start   = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= DURATION:
                break

            self._draw_dot(screen_x, screen_y, elapsed / DURATION)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            ret, frame = self.cap.read()
            if not ret:
                continue

            features, is_blinking = extract_features(frame)
            if features is not None and not is_blinking:
                self.X.append(features)
                self.Y.append([screen_x, screen_y])
                samples += 1

        print(f"  ({screen_x}, {screen_y}) — {samples} samples")


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        X_array = np.array(self.X)
        Y_array = np.array(self.Y)

        print(f"Training on {len(X_array)} samples...")

        Y_norm = Y_array / np.array([SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32)

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_norm,  dtype=torch.float32)

        torch.manual_seed(42)
        model     = GazeNet()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING, weight_decay=WEIGHTED_DECAY)

        for epoch in range(EPOCH_RANGE):
            predictions = model(X_tensor)
            loss        = criterion(predictions, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCH_RANGE} — loss: {loss.item():.4f}")

        print("Training complete")
        self.model = model
        return model


    # ------------------------------------------------------------------
    # Accuracy map
    # ------------------------------------------------------------------

    def show_accuracy_map(self):
        # Each calibration point: white dot = actual, coloured dot+arrow = predicted.
        # Green = accurate, red = large error.

        self.model.eval()

        X_array = np.array(self.X)
        Y_array = np.array(self.Y)

        with torch.no_grad():
            preds_norm = self.model(torch.tensor(X_array, dtype=torch.float32)).numpy()
        preds_px = preds_norm * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])

        # Group by calibration point
        point_map = {}
        for i, (ax, ay) in enumerate(Y_array):
            key = (int(ax), int(ay))
            if key not in point_map:
                point_map[key] = []
            point_map[key].append(preds_px[i])

        self.screen.fill((20, 20, 20))

        for gx in range(0, SCREEN_WIDTH,  SCREEN_WIDTH  // 10):
            pygame.draw.line(self.screen, (35, 35, 35), (gx, 0), (gx, SCREEN_HEIGHT))
        for gy in range(0, SCREEN_HEIGHT, SCREEN_HEIGHT // 10):
            pygame.draw.line(self.screen, (35, 35, 35), (0, gy), (SCREEN_WIDTH, gy))

        max_err = 200
        for (ax, ay), preds in point_map.items():
            mean_pred = np.mean(preds, axis=0)
            px, py    = int(mean_pred[0]), int(mean_pred[1])

            err    = math.sqrt((px - ax)**2 + (py - ay)**2)
            t      = min(err / max_err, 1.0)
            colour = (int(255 * t), int(255 * (1 - t)), 40)

            pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), 8)
            pygame.draw.circle(self.screen, colour,          (px, py), 6)
            pygame.draw.line(self.screen, colour, (ax, ay), (px, py), 2)
            label = self.font_small.render(f"{int(err)}px", True, colour)
            self.screen.blit(label, (px + 6, py - 6))

        legend = [
            "ACCURACY MAP — white = actual   dot = predicted",
            "green = accurate   red = large error",
            "press any key to continue",
        ]
        for i, txt in enumerate(legend):
            surf = self.font_small.render(txt, True, (160, 160, 160))
            self.screen.blit(surf, (20, 20 + i * 26))

        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting = False


    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        # Instruction screen
        self.screen.fill((0, 0, 0))
        lines = [
            "CALIBRATION",
            "",
            "25 dots will appear one at a time.",
            "Stare at each dot until it disappears.",
            "Keep your head still.",
            "",
            "PRESS SPACE TO BEGIN",
        ]
        y_pos = 140
        for line in lines:
            if line == "CALIBRATION":
                surf = self.font_large.render(line, True, (0, 180, 255))
            elif "PRESS SPACE" in line:
                surf = self.font_small.render(line, True, (0, 255, 100))
            else:
                surf = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (SCREEN_WIDTH//2 - surf.get_width()//2, y_pos))
            y_pos += 40
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_SPACE:
                        waiting = False

        time.sleep(0.5)

        # Countdown
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))
            surf = self.font_large.render(str(i), True, (255, 255, 255))
            self.screen.blit(surf, (SCREEN_WIDTH//2 - surf.get_width()//2,
                                    SCREEN_HEIGHT//2 - surf.get_height()//2))
            pygame.display.flip()
            end = time.time() + 1
            while time.time() < end:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                time.sleep(0.01)

        # 25 dots
        for fx, fy in CALIBRATION_POINTS:
            screen_x = int(fx * SCREEN_WIDTH)
            screen_y = int(fy * SCREEN_HEIGHT)
            self._collect_dot(screen_x, screen_y)

        model = self.train()

        self.show_accuracy_map()

        self.screen.fill((0, 0, 0))
        done = self.font_large.render("CALIBRATION COMPLETE", True, (0, 255, 100))
        self.screen.blit(done, (SCREEN_WIDTH//2 - done.get_width()//2,
                                SCREEN_HEIGHT//2 - done.get_height()//2))
        pygame.display.flip()
        time.sleep(1.5)

        return model
