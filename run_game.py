"""
Shape-Shift Racer: Real-Time Face Expression & Pose Controlled Morphing Car Game.

Entrypoint script:
Controls:
  - Face Smile           -> Morph FLAT (under low beams)
  - Face Open Mouth ("O")-> Morph NARROW (between pillar slits)
  - Face Head Turn/Tilt  -> Steer Left / Right
  - Keyboard Fallback    -> A/D or Arrows to Steer, Down to Flatten, Up to Narrow, Space to Start.
"""

import os
import sys
import time
import random
import platform
import ctypes
import pygame

os.environ["SDL_VIDEO_CENTERED"] = "1"

from game.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TITLE,
    PIP_X, PIP_Y, PIP_WIDTH, PIP_HEIGHT,
    BASE_SPEED, SPEED_INCREMENT, MAX_SPEED,
    COLOR_NEON_YELLOW, COLOR_NEON_CYAN, COLOR_NEON_RED, COLOR_NEON_GREEN,
    CAR_SHAPES
)
from game.road import RoadRenderer
from game.car import Car
from game.obstacles import ObstacleManager
from game.expression_worker import ExpressionWorker
from game.sound_manager import SoundManager


class ExplosionDebris:
    """Spark debris from collision impact."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6.0, 6.0)
        self.vy = random.uniform(-8.0, 3.0)
        self.color = random.choice([(255, 230, 0), (255, 60, 40), (255, 255, 255)])
        self.life = 1.0
        self.size = random.uniform(3.0, 7.0)

    def update(self, dt):
        self.x += self.vx * 60.0 * dt
        self.y += self.vy * 60.0 * dt
        self.vy += 12.0 * dt  # Gravity
        self.life -= 1.8 * dt
        return self.life > 0

    def draw(self, screen):
        alpha = int(255 * max(0.0, self.life))
        s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
        screen.blit(s, (self.x - self.size, self.y - self.size))


class ShapeShiftRacerGame:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self._bring_window_to_front()
        self.clock = pygame.time.Clock()

        # Audio Manager
        self.sound_mgr = SoundManager()

        # Game Engines
        self.road = RoadRenderer()
        self.car = Car()
        self.obstacles = ObstacleManager()

        # Game State
        self.state = "START"  # "START", "CALIBRATING", "PLAYING", "GAMEOVER"
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.speed = BASE_SPEED
        self.distance = 0.0
        self.invulnerable_timer = 0.0
        self.screen_flash_timer = 0.0
        self.debris = []

        # Calibration state
        self.calibration_timer = 3.0

        # Fonts
        self.font_title = pygame.font.SysFont("Impact", 54)
        self.font_sub = pygame.font.SysFont("Impact", 28)
        self.font_hud = pygame.font.SysFont("Impact", 24)
        self.font_body = pygame.font.SysFont("Arial", 18, bold=True)

        # Fallback PiP surface created safely on main thread
        self.fallback_pip_surface = self._create_fallback_pip_surface()

        # Vision background thread (started after display & fonts are initialized)
        self.vision = ExpressionWorker(camera_id=0)
        self.vision.start()

    def _bring_window_to_front(self):
        """Ensures the Pygame window is visible and focused on Windows."""
        if platform.system() == "Windows":
            try:
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 5)   # SW_SHOW
                    ctypes.windll.user32.ShowWindow(hwnd, 9)   # SW_RESTORE
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    def _create_fallback_pip_surface(self):
        """Creates a fallback HUD surface on the main thread when camera is initializing/unavailable."""
        surf = pygame.Surface((PIP_WIDTH, PIP_HEIGHT))
        surf.fill((25, 20, 38))
        f = pygame.font.SysFont("Arial", 14, bold=True)
        lines = [
            ("Camera Initializing / Standby", (255, 220, 80)),
            ("Keyboard Controls Active:", (200, 220, 255)),
            ("A / D  : Steer Left / Right", (0, 230, 255)),
            ("DOWN   : Morph FLAT (Smile)", (255, 80, 180)),
            ("UP     : Morph NARROW (O-Mouth)", (80, 255, 140)),
        ]
        for idx, (text, color) in enumerate(lines):
            surf.blit(f.render(text, True, color), (14, 22 + idx * 30))
        return surf

    def run(self):
        """Main game loop running at 60 FPS."""
        running = True
        last_time = time.time()

        while running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.08)  # Clamp against lag spikes

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.state == "START":
                            self.state = "CALIBRATING"
                            self.calibration_timer = 3.0
                            self.vision.trigger_calibration()
                        elif self.state == "CALIBRATING":
                            self._reset_game()
                            self.state = "PLAYING"
                        elif self.state == "GAMEOVER":
                            self._reset_game()
                            self.state = "PLAYING"
                    elif event.key == pygame.K_c:
                        # Quick recalibrate during game
                        self.vision.trigger_calibration()
                        self.sound_mgr.play("calibrated")

            # 2. State-Specific Update
            if self.state == "START":
                self._update_start_screen(dt)
            elif self.state == "CALIBRATING":
                self._update_calibration_screen(dt)
            elif self.state == "PLAYING":
                self._update_playing(dt)
            elif self.state == "GAMEOVER":
                self._update_game_over(dt)

            # 3. Render Everything
            self._render()

            pygame.display.flip()

        # Clean shutdown
        self.vision.stop()
        pygame.quit()
        sys.exit(0)

    def _reset_game(self):
        """Resets gameplay variables for a new round."""
        self.score = 0
        self.lives = 3
        self.speed = BASE_SPEED
        self.distance = 0.0
        self.obstacles.reset()
        self.invulnerable_timer = 0.0
        self.screen_flash_timer = 0.0
        self.debris.clear()
        self.car.norm_x = 0.0
        self.car.set_target_steer(0.0)
        self.car.set_shape("NORMAL")

    def _update_start_screen(self, dt):
        """Menu animation."""
        self.road.update(120.0, dt)
        self.car.update(self.road, dt)

    def _update_calibration_screen(self, dt):
        """Countdown for face neutral baseline calibration."""
        self.road.update(150.0, dt)
        self.car.update(self.road, dt)
        self.calibration_timer -= dt

        if self.calibration_timer <= 0:
            self.sound_mgr.play("calibrated")
            self._reset_game()
            self.state = "PLAYING"

    def _update_playing(self, dt):
        """Active gameplay loop."""
        vision_state = self.vision.get_state()

        # Keyboard Fallback Controls
        keys = pygame.key.get_pressed()
        steer_input = vision_state["steer_x"]
        shape_input = vision_state["shape"]

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer_input = -0.75
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer_input = 0.75

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            shape_input = "FLAT"
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            shape_input = "NARROW"

        # Apply inputs to car
        self.car.set_target_steer(steer_input)
        shape_changed = self.car.set_shape(shape_input)
        if shape_changed:
            self.sound_mgr.play("morph")

        # Update Road & Car
        self.road.update(self.speed, dt)
        self.car.update(self.road, dt)
        self.distance += (self.speed * dt) / 10.0

        # Invulnerability & flash timers
        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= dt
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= dt

        # Update Debris
        self.debris = [d for d in self.debris if d.update(dt)]

        # Update Obstacles & Collision Check
        points, hit = self.obstacles.update(self.speed, dt, self.car, self.sound_mgr)
        if points > 0:
            self.score += points
            # Gradually increase speed
            self.speed = min(MAX_SPEED, self.speed + SPEED_INCREMENT)

        if hit and self.invulnerable_timer <= 0:
            self.lives -= 1
            self.screen_flash_timer = 0.18
            self.invulnerable_timer = 1.4  # Grace period
            # Spawn explosion debris
            for _ in range(25):
                self.debris.append(ExplosionDebris(self.car.screen_x, self.car.screen_y - self.car.h / 2.0))

            if self.lives <= 0:
                self.high_score = max(self.high_score, self.score)
                self.state = "GAMEOVER"

    def _update_game_over(self, dt):
        """Game over screen animation."""
        self.road.update(60.0, dt)
        self.debris = [d for d in self.debris if d.update(dt)]

    def _render(self):
        """Master draw call."""
        # 1. 2.5D Road Environment
        self.road.render(self.screen)

        # 2. Obstacles
        self.obstacles.render(self.screen, self.road)

        # 3. Car (with invulnerability blinking)
        if self.invulnerable_timer <= 0 or int(self.invulnerable_timer * 10) % 2 == 0:
            self.car.render(self.screen)

        # 4. Collision Debris
        for d in self.debris:
            d.draw(self.screen)

        # 5. Red Collision Screen Flash
        if self.screen_flash_timer > 0:
            flash_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(140 * (self.screen_flash_timer / 0.18))
            flash_surf.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surf, (0, 0))

        # 6. HUD & Overlay Displays
        self._render_hud()

        # State-Specific Overlays
        if self.state == "START":
            self._render_start_overlay()
        elif self.state == "CALIBRATING":
            self._render_calibration_overlay()
        elif self.state == "GAMEOVER":
            self._render_game_over_overlay()

    def _render_hud(self):
        """Displays scoreboard, health hearts, shape badge, and webcam PiP."""
        # Top-Left Telemetry Box
        telemetry_rect = pygame.Rect(20, 20, 260, 110)
        tele_surf = pygame.Surface((telemetry_rect.width, telemetry_rect.height), pygame.SRCALPHA)
        tele_surf.fill((15, 12, 28, 200))
        pygame.draw.rect(tele_surf, (0, 230, 255), (0, 0, telemetry_rect.width, telemetry_rect.height), width=1, border_radius=6)
        self.screen.blit(tele_surf, telemetry_rect.topleft)

        # Score & Distance
        score_txt = self.font_hud.render(f"SCORE: {self.score}", True, (255, 230, 0))
        dist_txt = self.font_body.render(f"DISTANCE: {int(self.distance)} m", True, (200, 240, 255))
        speed_txt = self.font_body.render(f"SPEED: {int(self.speed)} km/h", True, (100, 255, 160))
        self.screen.blit(score_txt, (35, 30))
        self.screen.blit(dist_txt, (35, 65))
        self.screen.blit(speed_txt, (35, 90))

        # Lives / Vector Hearts (Impact font lacks Unicode heart glyph)
        for i in range(3):
            hx = 200 + i * 24
            hy = 40
            filled = i < self.lives
            col = (255, 50, 80) if filled else (70, 50, 65)
            pygame.draw.circle(self.screen, col, (hx - 4, hy - 2), 5)
            pygame.draw.circle(self.screen, col, (hx + 4, hy - 2), 5)
            pygame.draw.polygon(self.screen, col, [(hx - 9, hy), (hx + 9, hy), (hx, hy + 10)])

        # Bottom-Center Morph Shape Indicator
        shape_cfg = CAR_SHAPES[self.car.current_shape_key]
        shape_name = shape_cfg["name"]
        shape_col = shape_cfg["color_primary"]

        badge_w, badge_h = 320, 48
        badge_x = (SCREEN_WIDTH - badge_w) // 2
        badge_y = SCREEN_HEIGHT - badge_h - 15

        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        badge_surf.fill((18, 14, 30, 220))
        pygame.draw.rect(badge_surf, shape_col, (0, 0, badge_w, badge_h), width=2, border_radius=8)
        self.screen.blit(badge_surf, (badge_x, badge_y))

        morph_title = self.font_sub.render(f"MORPH: {shape_name}", True, shape_col)
        self.screen.blit(morph_title, (badge_x + (badge_w - morph_title.get_width()) // 2, badge_y + 8))

        # Top-Right Webcam Picture-in-Picture (converted safely on main thread)
        vision_state = self.vision.get_state()
        rgb_bytes = vision_state["pip_rgb_bytes"]
        if rgb_bytes is not None:
            pip_surf = pygame.image.frombuffer(rgb_bytes, (PIP_WIDTH, PIP_HEIGHT), "RGB")
        else:
            pip_surf = self.fallback_pip_surface

        pip_border = pygame.Rect(PIP_X - 2, PIP_Y - 2, PIP_WIDTH + 4, PIP_HEIGHT + 4)
        pygame.draw.rect(self.screen, (0, 255, 255), pip_border, width=2, border_radius=4)
        self.screen.blit(pip_surf, (PIP_X, PIP_Y))

    def _render_start_overlay(self):
        """Renders intro banner and instructions."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 5, 20, 180))
        self.screen.blit(overlay, (0, 0))

        # Title
        t1 = self.font_title.render("SHAPE-SHIFT RACER", True, (0, 230, 255))
        t2 = self.font_sub.render("FACE EXPRESSION & POSE CONTROLLED MORPHING CAR", True, (255, 230, 0))
        self.screen.blit(t1, ((SCREEN_WIDTH - t1.get_width()) // 2, 140))
        self.screen.blit(t2, ((SCREEN_WIDTH - t2.get_width()) // 2, 210))

        # Instructions Card
        card_w, card_h = 660, 240
        card_x = (SCREEN_WIDTH - card_w) // 2
        card_y = 270
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((20, 16, 38, 230))
        pygame.draw.rect(card_surf, (255, 0, 128), (0, 0, card_w, card_h), width=2, border_radius=10)
        self.screen.blit(card_surf, (card_x, card_y))

        rules = [
            ("SMILE BIG", "--> Morphs car FLAT (Slides under low horizontal barriers)"),
            ("OPEN MOUTH ('O')", "--> Morphs car NARROW (Squeezes through twin pillars)"),
            ("HEAD TURN / TILT", "--> Steers Left / Right to avoid lane roadblocks"),
            ("KEYBOARD BACKUP", "--> A/D to steer, DOWN to flatten, UP to narrow")
        ]

        for i, (action, desc) in enumerate(rules):
            a_txt = self.font_body.render(action, True, (255, 230, 0))
            d_txt = self.font_body.render(desc, True, (220, 240, 255))
            y_pos = card_y + 25 + i * 48
            self.screen.blit(a_txt, (card_x + 30, y_pos))
            self.screen.blit(d_txt, (card_x + 210, y_pos))

        # Press Space Prompt (blinking)
        if int(time.time() * 2) % 2 == 0:
            prompt = self.font_sub.render("PRESS SPACE TO CALIBRATE & START", True, (100, 255, 140))
            self.screen.blit(prompt, ((SCREEN_WIDTH - prompt.get_width()) // 2, 540))

    def _render_calibration_overlay(self):
        """Shows 3-second countdown for neutral face baseline calibration."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 2, 12, 160))
        self.screen.blit(overlay, (0, 0))

        title = self.font_title.render("CALIBRATING NEUTRAL FACE", True, (0, 230, 255))
        sub = self.font_sub.render("Look straight at the camera with a relaxed neutral face...", True, (255, 255, 255))
        countdown = self.font_title.render(f"{max(1, int(self.calibration_timer) + 1)}", True, (255, 230, 0))

        self.screen.blit(title, ((SCREEN_WIDTH - title.get_width()) // 2, 220))
        self.screen.blit(sub, ((SCREEN_WIDTH - sub.get_width()) // 2, 290))
        self.screen.blit(countdown, ((SCREEN_WIDTH - countdown.get_width()) // 2, 360))

    def _render_game_over_overlay(self):
        """Game Over summary screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 5, 15, 200))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_title.render("GAME OVER", True, (255, 40, 60))
        t2 = self.font_sub.render(f"FINAL SCORE: {self.score}", True, (255, 230, 0))
        t3 = self.font_sub.render(f"HIGH SCORE: {self.high_score}", True, (0, 230, 255))
        t4 = self.font_body.render(f"Distance Traveled: {int(self.distance)} meters", True, (220, 240, 255))

        self.screen.blit(t1, ((SCREEN_WIDTH - t1.get_width()) // 2, 200))
        self.screen.blit(t2, ((SCREEN_WIDTH - t2.get_width()) // 2, 275))
        self.screen.blit(t3, ((SCREEN_WIDTH - t3.get_width()) // 2, 320))
        self.screen.blit(t4, ((SCREEN_WIDTH - t4.get_width()) // 2, 365))

        if int(time.time() * 2) % 2 == 0:
            prompt = self.font_sub.render("PRESS SPACE TO PLAY AGAIN", True, (100, 255, 140))
            self.screen.blit(prompt, ((SCREEN_WIDTH - prompt.get_width()) // 2, 440))


if __name__ == "__main__":
    game = ShapeShiftRacerGame()
    game.run()
