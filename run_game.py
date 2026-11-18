"""
Shape-Shift Racer: Real-Time Face Expression & Pose Controlled Morphing Car Game.

Controls:
  - Face Smile            -> Morph FLAT (slide under low beams)
  - Face Open Mouth ("O") -> Morph NARROW (squeeze through twin pillars)
  - Face Head Turn / Tilt -> Steer Left / Right around roadblocks & shifted gates
  - Keyboard Fallback     -> A/D or Arrows to Steer, Down to Flatten, Up to Narrow
  - P                     -> Pause & Live Sensitivity Menu
  - M                     -> Toggle Synthwave Background Music
  - C                     -> Recalibrate Neutral Face Baseline
  - Space                 -> Start / Skip Calibration / Resume / Play Again
  - Esc or Q              -> Quit
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
    OBSTACLE_BASE_POINTS, COIN_BASE_POINTS, NEAR_MISS_BONUS,
    STREAK_TIERS, CAR_SHAPES
)
from game.road import RoadRenderer
from game.car import Car
from game.obstacles import ObstacleManager
from game.expression_worker import ExpressionWorker
from game.sound_manager import SoundManager
from game.storage import (
    load_highscores, save_highscore,
    load_settings, save_settings
)


class SparkParticle:
    """Debris or golden sparkle particle for collisions and coin pickups."""
    def __init__(self, x, y, palette=None, upward=False):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-5.5, 5.5)
        self.vy = random.uniform(-7.5, -1.5) if upward else random.uniform(-8.0, 3.0)
        if palette is None:
            palette = [(255, 230, 0), (255, 60, 40), (255, 255, 255)]
        self.color = random.choice(palette)
        self.life = 1.0
        self.size = random.uniform(3.0, 6.5)

    def update(self, dt):
        self.x += self.vx * 60.0 * dt
        self.y += self.vy * 60.0 * dt
        self.vy += 12.0 * dt
        self.life -= 1.8 * dt
        return self.life > 0

    def draw(self, screen):
        alpha = int(255 * max(0.0, self.life))
        r = max(1, int(self.size))
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (r, r), r)
        screen.blit(s, (self.x - r, self.y - r))


class FloatingPopup:
    """Animated rising text notification (points, coin pickup, streak, near-miss, biome)."""
    def __init__(self, text, x, y, color=(255, 230, 0), size=26, duration=1.2):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size = size
        self.life = duration
        self.max_life = duration

    def update(self, dt):
        self.y -= 48.0 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, screen):
        alpha = int(255 * max(0.0, min(1.0, self.life / max(self.max_life * 0.5, 0.01))))
        font = pygame.font.SysFont("Impact", self.size)
        txt = font.render(self.text, True, self.color)
        shadow = font.render(self.text, True, (10, 8, 20))
        txt.set_alpha(alpha)
        shadow.set_alpha(alpha)
        pos_x = int(self.x - txt.get_width() // 2)
        pos_y = int(self.y)
        screen.blit(shadow, (pos_x + 2, pos_y + 2))
        screen.blit(txt, (pos_x, pos_y))


def get_multiplier_for_streak(streak):
    for min_streak, mult in STREAK_TIERS:
        if streak >= min_streak:
            return mult
    return 1


class ShapeShiftRacerGame:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self._bring_window_to_front()
        self.clock = pygame.time.Clock()

        # Load persistent user settings & high scores
        self.settings = load_settings()
        self.highscores = load_highscores()
        self.high_score = self.highscores[0]["score"] if self.highscores else 0
        self.is_new_record = False

        # Audio Manager (with procedural synthwave music)
        self.sound_mgr = SoundManager(music_enabled=self.settings.get("music_enabled", True))

        # Game Engines
        self.road = RoadRenderer()
        self.car = Car()
        self.obstacles = ObstacleManager()

        # Game State: "START", "CALIBRATING", "PLAYING", "PAUSED", "GAMEOVER"
        self.state = "START"
        self.score = 0
        self.coins_collected = 0
        self.streak = 0
        self.max_streak = 0
        self.multiplier = 1
        self.lives = 3
        self.speed = BASE_SPEED
        self.distance = 0.0
        self.invulnerable_timer = 0.0
        self.screen_flash_timer = 0.0
        self.screen_shake_timer = 0.0
        self.sparks = []
        self.popups = []

        # Pause & Sensitivity Menu state
        self.pause_menu_idx = 0
        self.pause_items = [
            ("smile_sensitivity", "Smile Sensitivity"),
            ("open_sensitivity", "O-Mouth Sensitivity"),
            ("steer_sensitivity", "Steer Sensitivity"),
            ("music_enabled", "Synthwave Music"),
        ]

        # 3-Step Calibration Wizard state: ("NEUTRAL", "SMILE", "OPEN")
        self.calibration_steps = [
            ("NEUTRAL", "STEP 1/3: RELAXED NEUTRAL FACE", "Look straight ahead with mouth closed & eyes open", (0, 230, 255)),
            ("SMILE",   "STEP 2/3: SMILE BIG!", "Hold a big smile to personalize FLAT morph", (255, 60, 170)),
            ("OPEN",    "STEP 3/3: OPEN MOUTH ('O' / 'AH')", "Open your mouth to personalize NARROW morph", (80, 255, 140)),
        ]
        self.calibration_step_idx = 0
        self.calibration_step_duration = 2.2
        self.calibration_timer = self.calibration_step_duration

        # Fonts
        self.font_title = pygame.font.SysFont("Impact", 54)
        self.font_sub = pygame.font.SysFont("Impact", 28)
        self.font_hud = pygame.font.SysFont("Impact", 24)
        self.font_small = pygame.font.SysFont("Arial", 15, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 18, bold=True)

        # Fallback PiP surface created safely on main thread
        self.fallback_pip_surface = self._create_fallback_pip_surface()

        # Vision background thread (apply loaded sensitivities)
        self.vision = ExpressionWorker(camera_id=0)
        self.vision.set_sensitivities(
            smile=self.settings["smile_sensitivity"],
            open_m=self.settings["open_sensitivity"],
            steer=self.settings["steer_sensitivity"],
        )
        self.vision.start()

    def _bring_window_to_front(self):
        if platform.system() == "Windows":
            try:
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 5)
                    ctypes.windll.user32.ShowWindow(hwnd, 9)
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    def _create_fallback_pip_surface(self):
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

    def _adjust_pause_setting(self, delta):
        """Adjusts the currently selected Pause Menu item and persists settings."""
        key, _ = self.pause_items[self.pause_menu_idx]
        if key == "music_enabled":
            new_val = self.sound_mgr.toggle_music()
            self.settings["music_enabled"] = new_val
        else:
            val = round(self.settings[key] + delta * 0.1, 2)
            val = max(0.5, min(2.0, val))
            self.settings[key] = val
            self.vision.set_sensitivities(
                smile=self.settings["smile_sensitivity"],
                open_m=self.settings["open_sensitivity"],
                steer=self.settings["steer_sensitivity"],
            )
        save_settings(self.settings)

    def run(self):
        """Main game loop running at 60 FPS."""
        running = True

        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.08)

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or (event.key == pygame.K_ESCAPE and self.state != "PAUSED"):
                        running = False
                    elif event.key == pygame.K_ESCAPE and self.state == "PAUSED":
                        self.state = "PLAYING"
                    elif event.key == pygame.K_m:
                        enabled = self.sound_mgr.toggle_music()
                        self.settings["music_enabled"] = enabled
                        save_settings(self.settings)
                        self.popups.append(FloatingPopup(
                            "MUSIC: ON" if enabled else "MUSIC: MUTED",
                            SCREEN_WIDTH // 2, 130, (0, 230, 255), size=24, duration=1.0
                        ))
                    elif event.key == pygame.K_p:
                        if self.state == "PLAYING":
                            self.state = "PAUSED"
                        elif self.state == "PAUSED":
                            self.state = "PLAYING"
                    elif event.key == pygame.K_c:
                        self.vision.trigger_calibration("NEUTRAL")
                        self.sound_mgr.play("calibrated")
                        self.popups.append(FloatingPopup(
                            "RECALIBRATING NEUTRAL FACE...",
                            SCREEN_WIDTH // 2, 170, (100, 255, 160), size=24, duration=1.2
                        ))
                    elif event.key == pygame.K_j and self.state == "PLAYING":
                        if self.car.trigger_jump():
                            self.sound_mgr.play("jump")
                            self.popups.append(FloatingPopup("JUMP!", self.car.screen_x, self.car.screen_y - 90, (0, 240, 255), size=24, duration=0.7))
                    elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT) and self.state == "PLAYING":
                        if self.car.activate_shield():
                            self.sound_mgr.play("shield")
                            self.popups.append(FloatingPopup("SHIELD + NITRO!", SCREEN_WIDTH // 2, 230, (0, 245, 255), size=34, duration=1.2))
                    elif self.state == "PAUSED":
                        if event.key in (pygame.K_UP, pygame.K_w):
                            self.pause_menu_idx = (self.pause_menu_idx - 1) % len(self.pause_items)
                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            self.pause_menu_idx = (self.pause_menu_idx + 1) % len(self.pause_items)
                        elif event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_MINUS):
                            self._adjust_pause_setting(-1)
                        elif event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_EQUALS, pygame.K_PLUS):
                            self._adjust_pause_setting(1)
                        elif event.key == pygame.K_SPACE:
                            self.state = "PLAYING"
                        elif event.key == pygame.K_r:
                            self._reset_game()
                            self.state = "PLAYING"
                    elif event.key == pygame.K_SPACE:
                        if self.state == "START":
                            self.state = "CALIBRATING"
                            self.calibration_step_idx = 0
                            self.calibration_timer = self.calibration_step_duration
                            self.vision.trigger_calibration("NEUTRAL")
                        elif self.state == "CALIBRATING":
                            # Skip remaining calibration steps cleanly
                            stage_key = self.calibration_steps[self.calibration_step_idx][0]
                            self.vision.finalize_calibration_stage(stage_key)
                            self.vision.trigger_calibration(None)
                            self._reset_game()
                            self.state = "PLAYING"
                        elif self.state == "PLAYING":
                            if self.car.trigger_jump():
                                self.sound_mgr.play("jump")
                                self.popups.append(FloatingPopup("JUMP!", self.car.screen_x, self.car.screen_y - 90, (0, 240, 255), size=24, duration=0.7))
                        elif self.state == "GAMEOVER":
                            self._reset_game()
                            self.state = "PLAYING"

            # 2. State-Specific Update
            if self.state == "START":
                self._update_start_screen(dt)
            elif self.state == "CALIBRATING":
                self._update_calibration_screen(dt)
            elif self.state == "PLAYING":
                self._update_playing(dt)
            elif self.state == "PAUSED":
                self._update_paused(dt)
            elif self.state == "GAMEOVER":
                self._update_game_over(dt)

            # 3. Render Everything
            self._render()
            pygame.display.flip()

        self.sound_mgr.stop_music()
        self.vision.stop()
        pygame.quit()
        sys.exit(0)

    def _reset_game(self):
        self.score = 0
        self.coins_collected = 0
        self.streak = 0
        self.max_streak = 0
        self.multiplier = 1
        self.lives = 3
        self.speed = BASE_SPEED
        self.distance = 0.0
        self.is_new_record = False
        self.invulnerable_timer = 0.0
        self.screen_flash_timer = 0.0
        self.screen_shake_timer = 0.0
        self.sparks.clear()
        self.popups.clear()
        self.road.reset()
        self.obstacles.reset()
        self.car.norm_x = 0.0
        self.car.set_target_steer(0.0)
        self.car.set_shape("NORMAL")
        self.car.reset_abilities()

    def _update_start_screen(self, dt):
        self.road.update(120.0, dt, distance_meters=0.0)
        self.car.update(self.road, dt)
        self.popups = [p for p in self.popups if p.update(dt)]

    def _update_calibration_screen(self, dt):
        self.road.update(150.0, dt, distance_meters=0.0)
        self.car.update(self.road, dt)
        self.calibration_timer -= dt
        if self.calibration_timer <= 0:
            # Finalize current step and advance wizard
            stage_key = self.calibration_steps[self.calibration_step_idx][0]
            self.vision.finalize_calibration_stage(stage_key)
            self.sound_mgr.play("calibrated")

            self.calibration_step_idx += 1
            if self.calibration_step_idx < len(self.calibration_steps):
                next_stage = self.calibration_steps[self.calibration_step_idx][0]
                self.calibration_timer = self.calibration_step_duration
                self.vision.trigger_calibration(next_stage)
            else:
                self.vision.trigger_calibration(None)
                self._reset_game()
                self.state = "PLAYING"

    def _update_paused(self, dt):
        # Keep popups animating while paused (live PiP keeps running in background!)
        self.popups = [p for p in self.popups if p.update(dt)]

    def _update_playing(self, dt):
        vision_state = self.vision.get_state(consume_triggers=True)

        # Face-triggered Jump (Eyebrow Raise) & Shield+Nitro (Asymmetric Wink)
        if vision_state["jump_triggered"]:
            if self.car.trigger_jump():
                self.sound_mgr.play("jump")
                self.popups.append(FloatingPopup("BROW JUMP!", self.car.screen_x, self.car.screen_y - 90, (0, 240, 255), size=24, duration=0.8))

        if vision_state["wink_triggered"]:
            if self.car.activate_shield():
                self.sound_mgr.play("shield")
                self.popups.append(FloatingPopup("WINK SHIELD + NITRO!", SCREEN_WIDTH // 2, 230, (0, 245, 255), size=34, duration=1.3))

        # Keyboard fallback overrides
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
        if self.car.set_shape(shape_input):
            self.sound_mgr.play("morph")

        # Effective speed gets +30% Nitro boost while Shield is active!
        effective_speed = self.speed * (1.30 if self.car.is_shielded else 1.0)

        # Update Distance, Road, Dynamic Biome & Curves
        self.distance += (effective_speed * dt) / 10.0
        biome_name, biome_changed = self.road.update(effective_speed, dt, distance_meters=self.distance)
        if biome_changed:
            self.sound_mgr.play("biome")
            self.popups.append(FloatingPopup(
                f"ZONE: {biome_name}",
                SCREEN_WIDTH // 2, 160,
                self.road.get_current_accent_color(), size=36, duration=1.8
            ))

        self.car.update(self.road, dt)

        # Timers
        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= dt
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= dt
        if self.screen_shake_timer > 0:
            self.screen_shake_timer -= dt

        self.sparks = [d for d in self.sparks if d.update(dt)]
        self.popups = [p for p in self.popups if p.update(dt)]

        # Update Obstacles, Coins, Gauntlets, Laser Strips & Collisions
        ev = self.obstacles.update(effective_speed, dt, self.car, self.sound_mgr, self.road)

        if ev["gauntlet_started"]:
            self.popups.append(FloatingPopup(
                "GAUNTLET WAVE!", SCREEN_WIDTH // 2, 210, (255, 80, 180), size=34, duration=1.5
            ))

        if ev["shield_smashes"] > 0:
            smash_pts = 150 * ev["shield_smashes"] * self.multiplier
            self.score += smash_pts
            self.screen_shake_timer = 0.14
            self.popups.append(FloatingPopup(
                f"SHIELD SMASH +{smash_pts}!",
                self.car.screen_x, self.car.screen_y - self.car.h - 45,
                (0, 245, 255), size=28, duration=1.1
            ))
            for _ in range(18):
                self.sparks.append(SparkParticle(
                    self.car.screen_x, self.car.screen_y - self.car.h / 2.0,
                    palette=[(0, 240, 255), (255, 230, 80), (255, 255, 255)]
                ))

        if ev["obstacles_passed"] > 0:
            prev_mult = self.multiplier
            self.streak += ev["obstacles_passed"]
            self.max_streak = max(self.max_streak, self.streak)
            self.multiplier = get_multiplier_for_streak(self.streak)

            gained = OBSTACLE_BASE_POINTS * ev["obstacles_passed"] * self.multiplier
            self.score += gained
            self.speed = min(MAX_SPEED, self.speed + SPEED_INCREMENT * ev["obstacles_passed"])

            self.popups.append(FloatingPopup(
                f"+{gained}", self.car.screen_x, self.car.screen_y - self.car.h - 20,
                (100, 255, 160), size=24, duration=0.9
            ))

            if self.multiplier > prev_mult:
                self.sound_mgr.play("streak")
                self.popups.append(FloatingPopup(
                    f"STREAK {self.multiplier}x!",
                    SCREEN_WIDTH // 2, 260, (255, 220, 40), size=36, duration=1.3
                ))

        if ev["near_misses"] > 0:
            bonus = NEAR_MISS_BONUS * ev["near_misses"] * self.multiplier
            self.score += bonus
            self.popups.append(FloatingPopup(
                f"NEAR MISS +{bonus}!",
                self.car.screen_x, self.car.screen_y - self.car.h - 52,
                (0, 240, 255), size=24, duration=1.1
            ))

        if ev["coins_collected"] > 0:
            self.coins_collected += ev["coins_collected"]
            coin_pts = COIN_BASE_POINTS * ev["coins_collected"] * self.multiplier
            self.score += coin_pts
            for cx, cy in ev["coin_screen_positions"]:
                self.popups.append(FloatingPopup(f"COIN +{coin_pts}", cx, cy, (255, 215, 0), size=22, duration=0.9))
                for _ in range(10):
                    self.sparks.append(SparkParticle(cx, cy, palette=[(255, 230, 0), (255, 255, 200), (255, 170, 0)], upward=True))

        if ev["collision"] and self.invulnerable_timer <= 0:
            self.lives -= 1
            self.streak = 0
            self.multiplier = 1
            self.screen_flash_timer = 0.20
            self.screen_shake_timer = 0.25
            self.invulnerable_timer = 1.4

            for _ in range(28):
                self.sparks.append(SparkParticle(self.car.screen_x, self.car.screen_y - self.car.h / 2.0))

            if self.lives <= 0:
                self.highscores, self.is_new_record = save_highscore(
                    self.score, self.distance, self.max_streak, self.coins_collected
                )
                self.high_score = self.highscores[0]["score"] if self.highscores else self.score
                self.state = "GAMEOVER"

    def _update_game_over(self, dt):
        self.road.update(60.0, dt, distance_meters=self.distance)
        self.sparks = [d for d in self.sparks if d.update(dt)]
        self.popups = [p for p in self.popups if p.update(dt)]

    def _render(self):
        """Master draw call with subtle collision screen shake."""
        render_target = self.screen
        shake_x, shake_y = 0, 0
        if self.screen_shake_timer > 0:
            mag = int(10 * (self.screen_shake_timer / 0.25))
            shake_x = random.randint(-mag, mag)
            shake_y = random.randint(-mag, mag)
            render_target = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # 1. 2.5D Dynamic Biome Curved Road
        self.road.render(render_target)

        # 2. Obstacles & Coins
        self.obstacles.render(render_target, self.road)

        # 3. Morphing GT Sports Car
        if self.invulnerable_timer <= 0 or int(self.invulnerable_timer * 10) % 2 == 0:
            self.car.render(render_target)

        # 4. Sparks & Floating Popups
        for d in self.sparks:
            d.draw(render_target)
        for p in self.popups:
            p.draw(render_target)

        # Blit shaken world back to main screen if shaking
        if render_target is not self.screen:
            self.screen.fill((10, 5, 15))
            self.screen.blit(render_target, (shake_x, shake_y))

        # 5. Red Collision Screen Flash
        if self.screen_flash_timer > 0:
            flash_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(140 * (self.screen_flash_timer / 0.20))
            flash_surf.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surf, (0, 0))

        # 6. HUD & Overlays
        self._render_hud()

        if self.state == "START":
            self._render_start_overlay()
        elif self.state == "CALIBRATING":
            self._render_calibration_overlay()
        elif self.state == "PAUSED":
            self._render_pause_overlay()
        elif self.state == "GAMEOVER":
            self._render_game_over_overlay()

    def _render_hud(self):
        """Displays scoreboard, streak multiplier, biome badge, hearts, morph badge, and webcam PiP."""
        accent = self.road.get_current_accent_color()

        # Top-Left Telemetry Box
        tele_rect = pygame.Rect(20, 20, 295, 142)
        tele_surf = pygame.Surface((tele_rect.width, tele_rect.height), pygame.SRCALPHA)
        tele_surf.fill((15, 12, 28, 210))
        pygame.draw.rect(tele_surf, accent, (0, 0, tele_rect.width, tele_rect.height), width=1, border_radius=8)
        self.screen.blit(tele_surf, tele_rect.topleft)

        # Score, Distance, Speed, Coins
        self.screen.blit(self.font_hud.render(f"SCORE: {self.score}", True, (255, 230, 0)), (34, 28))
        self.screen.blit(self.font_small.render(f"BEST: {max(self.high_score, self.score)}   COINS: {self.coins_collected}", True, (255, 210, 100)), (34, 62))
        self.screen.blit(self.font_small.render(f"DIST: {int(self.distance)} m   SPEED: {int(self.speed)} km/h", True, (200, 240, 255)), (34, 86))
        self.screen.blit(self.font_small.render(f"ZONE: {self.road.get_current_biome_name()}", True, accent), (34, 110))

        # Shortcut Hint
        music_lbl = "ON" if self.sound_mgr.music_enabled else "OFF"
        self.screen.blit(self.font_small.render(f"[P] Pause   [M] Music: {music_lbl}   [C] Calibrate", True, (155, 165, 190)), (34, 134))

        # Vector Hearts (top-right inside telemetry box)
        for i in range(3):
            hx = 235 + i * 24
            hy = 40
            col = (255, 50, 80) if i < self.lives else (70, 50, 65)
            pygame.draw.circle(self.screen, col, (hx - 4, hy - 2), 5)
            pygame.draw.circle(self.screen, col, (hx + 4, hy - 2), 5)
            pygame.draw.polygon(self.screen, col, [(hx - 9, hy), (hx + 9, hy), (hx, hy + 10)])

        # Streak & Multiplier Pill (below telemetry box when active)
        pill_y = 170
        if self.streak > 0:
            mult_col = (255, 220, 40) if self.multiplier >= 3 else ((0, 255, 180) if self.multiplier >= 2 else (180, 210, 255))
            pill_surf = pygame.Surface((220, 34), pygame.SRCALPHA)
            pill_surf.fill((18, 14, 34, 215))
            pygame.draw.rect(pill_surf, mult_col, (0, 0, 220, 34), width=2, border_radius=6)
            self.screen.blit(pill_surf, (20, pill_y))
            st_txt = self.font_hud.render(f"STREAK {self.streak}   {self.multiplier}x", True, mult_col)
            self.screen.blit(st_txt, (32, pill_y + 3))
            pill_y += 42

        # Shield / Nitro Status Pill (Wink or Shift)
        if self.car.is_shielded:
            sh_text = f"NITRO SHIELD: {self.car.shield_timer:.1f}s"
            sh_col = (0, 245, 255)
        elif self.car.shield_cooldown > 0:
            sh_text = f"SHIELD COOLDOWN: {self.car.shield_cooldown:.1f}s"
            sh_col = (150, 155, 175)
        else:
            sh_text = "SHIELD READY (WINK / SHIFT)"
            sh_col = (80, 255, 150)

        sh_surf = pygame.Surface((235, 30), pygame.SRCALPHA)
        sh_surf.fill((18, 14, 34, 210))
        pygame.draw.rect(sh_surf, sh_col, (0, 0, 235, 30), width=1, border_radius=6)
        self.screen.blit(sh_surf, (20, pill_y))
        self.screen.blit(self.font_small.render(sh_text, True, sh_col), (30, pill_y + 6))

        # Top-Center Morph Shape Indicator (keeps the car completely unobstructed!)
        shape_cfg = CAR_SHAPES[self.car.current_shape_key]
        shape_name = shape_cfg["name"]
        shape_col = shape_cfg["color_primary"]

        badge_w, badge_h = 310, 46
        badge_x = (SCREEN_WIDTH - badge_w) // 2
        badge_y = 20

        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        badge_surf.fill((18, 14, 30, 220))
        pygame.draw.rect(badge_surf, shape_col, (0, 0, badge_w, badge_h), width=2, border_radius=8)
        self.screen.blit(badge_surf, (badge_x, badge_y))

        morph_title = self.font_sub.render(f"MORPH: {shape_name}", True, shape_col)
        self.screen.blit(morph_title, (badge_x + (badge_w - morph_title.get_width()) // 2, badge_y + 7))

        # Top-Right Webcam Picture-in-Picture (converted safely on main thread)
        vision_state = self.vision.get_state()
        rgb_bytes = vision_state["pip_rgb_bytes"]
        if rgb_bytes is not None:
            pip_surf = pygame.image.frombuffer(rgb_bytes, (PIP_WIDTH, PIP_HEIGHT), "RGB")
        else:
            pip_surf = self.fallback_pip_surface

        pip_border = pygame.Rect(PIP_X - 2, PIP_Y - 2, PIP_WIDTH + 4, PIP_HEIGHT + 4)
        pygame.draw.rect(self.screen, accent, pip_border, width=2, border_radius=4)
        self.screen.blit(pip_surf, (PIP_X, PIP_Y))

    def _render_start_overlay(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 5, 20, 180))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_title.render("SHAPE-SHIFT RACER", True, (0, 230, 255))
        t2 = self.font_sub.render("468-POINT FACE EXPRESSION & POSE CONTROLLED MORPHING CAR", True, (255, 230, 0))
        self.screen.blit(t1, ((SCREEN_WIDTH - t1.get_width()) // 2, 120))
        self.screen.blit(t2, ((SCREEN_WIDTH - t2.get_width()) // 2, 190))

        card_w, card_h = 740, 310
        card_x = (SCREEN_WIDTH - card_w) // 2
        card_y = 232
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((20, 16, 38, 230))
        pygame.draw.rect(card_surf, (255, 0, 128), (0, 0, card_w, card_h), width=2, border_radius=10)
        self.screen.blit(card_surf, (card_x, card_y))

        rules = [
            ("SMILE BIG", "--> Morphs car FLAT (Slides under low horizontal beams)"),
            ("OPEN MOUTH ('O'/'AH')", "--> Morphs car NARROW (Squeezes through twin pillars)"),
            ("RAISE EYEBROWS", "--> JUMP over red ground laser strips (Key: J / SPACE)"),
            ("WINK ONE EYE", "--> SHIELD + NITRO Boost to smash blocks (Key: SHIFT)"),
            ("HEAD TURN / TILT", "--> 3D solvePnP Steering for shifted slits & roadblocks"),
            ("PAUSE / AUDIO", "--> [P] Live sensitivity sliders, [M] Synthwave music"),
        ]

        for i, (action, desc) in enumerate(rules):
            a_txt = self.font_body.render(action, True, (255, 230, 0))
            d_txt = self.font_body.render(desc, True, (220, 240, 255))
            y_pos = card_y + 18 + i * 47
            self.screen.blit(a_txt, (card_x + 26, y_pos))
            self.screen.blit(d_txt, (card_x + 240, y_pos))

        if int(time.time() * 2) % 2 == 0:
            prompt = self.font_sub.render("PRESS SPACE FOR 3-STEP CALIBRATION & START", True, (100, 255, 140))
            self.screen.blit(prompt, ((SCREEN_WIDTH - prompt.get_width()) // 2, 570))

    def _render_calibration_overlay(self):
        """3-Step Interactive Calibration Wizard (NEUTRAL -> SMILE -> OPEN)."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((6, 3, 14, 175))
        self.screen.blit(overlay, (0, 0))

        step_idx = min(self.calibration_step_idx, len(self.calibration_steps) - 1)
        _, headline, instruction, accent = self.calibration_steps[step_idx]

        panel_w, panel_h = 700, 290
        px = (SCREEN_WIDTH - panel_w) // 2
        py = 195
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((18, 14, 34, 235))
        pygame.draw.rect(panel, accent, (0, 0, panel_w, panel_h), width=2, border_radius=12)
        self.screen.blit(panel, (px, py))

        # Step indicator pills (1, 2, 3)
        step_labels = ["1. NEUTRAL", "2. SMILE", "3. OPEN"]
        for i, lbl in enumerate(step_labels):
            pill_x = px + 95 + i * 185
            pill_y = py + 22
            done = i < step_idx
            active = i == step_idx
            col = (80, 255, 140) if done else (accent if active else (95, 100, 125))
            pygame.draw.rect(self.screen, col, (pill_x, pill_y, 140, 28), width=2 if not active else 0, border_radius=14)
            txt_col = (15, 12, 24) if active else col
            st = self.font_small.render(lbl, True, txt_col)
            self.screen.blit(st, (pill_x + (140 - st.get_width()) // 2, pill_y + 5))

        title = self.font_sub.render(headline, True, accent)
        sub = self.font_body.render(instruction, True, (235, 245, 255))
        self.screen.blit(title, (px + (panel_w - title.get_width()) // 2, py + 78))
        self.screen.blit(sub, (px + (panel_w - sub.get_width()) // 2, py + 122))

        # Progress bar for current step
        progress = 1.0 - max(0.0, min(1.0, self.calibration_timer / self.calibration_step_duration))
        bar_x, bar_y, bar_w, bar_h = px + 90, py + 172, panel_w - 180, 22
        pygame.draw.rect(self.screen, (45, 40, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(self.screen, accent, (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=6)

        skip_txt = self.font_small.render("Look at the top-right camera preview   |   Press SPACE to skip calibration", True, (175, 185, 210))
        self.screen.blit(skip_txt, (px + (panel_w - skip_txt.get_width()) // 2, py + 232))

    def _render_pause_overlay(self):
        """Interactive Pause & Live Sensitivity Tuning Menu."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 6, 18, 195))
        self.screen.blit(overlay, (0, 0))

        panel_w, panel_h = 620, 380
        px = (SCREEN_WIDTH - panel_w) // 2
        py = (SCREEN_HEIGHT - panel_h) // 2
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((22, 18, 42, 240))
        pygame.draw.rect(panel, (0, 230, 255), (0, 0, panel_w, panel_h), width=2, border_radius=12)
        self.screen.blit(panel, (px, py))

        title = self.font_title.render("PAUSED", True, (0, 230, 255))
        self.screen.blit(title, (px + (panel_w - title.get_width()) // 2, py + 18))

        hint = self.font_small.render("Use UP/DOWN to select   |   LEFT/RIGHT (-/+) to adjust live", True, (180, 200, 230))
        self.screen.blit(hint, (px + (panel_w - hint.get_width()) // 2, py + 86))

        for idx, (key, label) in enumerate(self.pause_items):
            selected = idx == self.pause_menu_idx
            row_y = py + 125 + idx * 50
            row_rect = pygame.Rect(px + 34, row_y - 6, panel_w - 68, 40)

            if selected:
                pygame.draw.rect(self.screen, (45, 36, 82), row_rect, border_radius=6)
                pygame.draw.rect(self.screen, (255, 230, 0), row_rect, width=2, border_radius=6)

            lbl_col = (255, 230, 0) if selected else (220, 230, 245)
            self.screen.blit(self.font_body.render(label, True, lbl_col), (px + 50, row_y + 4))

            if key == "music_enabled":
                val_str = "ON" if self.settings["music_enabled"] else "OFF"
                val_col = (80, 255, 140) if self.settings["music_enabled"] else (255, 90, 90)
                val_txt = self.font_hud.render(f"<  {val_str}  >", True, val_col)
                self.screen.blit(val_txt, (px + panel_w - 190, row_y))
            else:
                val = self.settings[key]
                # Draw visual slider bar (0.5 -> 2.0)
                bar_x, bar_y, bar_w, bar_h = px + 290, row_y + 8, 175, 14
                pygame.draw.rect(self.screen, (50, 45, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
                fill_ratio = (val - 0.5) / 1.5
                pygame.draw.rect(self.screen, (0, 230, 255), (bar_x, bar_y, int(bar_w * fill_ratio), bar_h), border_radius=4)
                val_txt = self.font_body.render(f"{val:.1f}x", True, (255, 255, 255))
                self.screen.blit(val_txt, (bar_x + bar_w + 18, row_y + 4))

        footer = self.font_small.render("[P / SPACE] Resume     [C] Recalibrate     [R] Restart     [Q] Quit", True, (100, 255, 150))
        self.screen.blit(footer, (px + (panel_w - footer.get_width()) // 2, py + panel_h - 36))

    def _render_game_over_overlay(self):
        """Game Over summary with persistent Top-5 High Score Leaderboard."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 5, 18, 210))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_title.render("GAME OVER", True, (255, 45, 65))
        self.screen.blit(t1, ((SCREEN_WIDTH - t1.get_width()) // 2, 70))

        if self.is_new_record:
            rec = self.font_sub.render("★ NEW HIGH SCORE! ★", True, (255, 225, 40))
            self.screen.blit(rec, ((SCREEN_WIDTH - rec.get_width()) // 2, 136))

        summary = self.font_sub.render(
            f"SCORE: {self.score}   |   DIST: {int(self.distance)} m   |   COINS: {self.coins_collected}   |   MAX STREAK: {self.max_streak}",
            True, (220, 240, 255)
        )
        self.screen.blit(summary, ((SCREEN_WIDTH - summary.get_width()) // 2, 182))

        # Persistent Top-5 Leaderboard Panel
        lb_w, lb_h = 640, 260
        lb_x = (SCREEN_WIDTH - lb_w) // 2
        lb_y = 235
        lb_surf = pygame.Surface((lb_w, lb_h), pygame.SRCALPHA)
        lb_surf.fill((22, 16, 40, 235))
        pygame.draw.rect(lb_surf, (255, 215, 0), (0, 0, lb_w, lb_h), width=2, border_radius=10)
        self.screen.blit(lb_surf, (lb_x, lb_y))

        lb_title = self.font_hud.render("TOP 5 HIGH SCORES (SAVED)", True, (255, 215, 0))
        self.screen.blit(lb_title, (lb_x + (lb_w - lb_title.get_width()) // 2, lb_y + 14))

        cols = [
            ("RANK", 32),
            ("SCORE", 105),
            ("DIST", 215),
            ("COINS", 305),
            ("STREAK", 390),
            ("DATE", 480),
        ]
        for heading, cx in cols:
            self.screen.blit(self.font_small.render(heading, True, (150, 175, 210)), (lb_x + cx, lb_y + 52))

        if not self.highscores:
            empty = self.font_body.render("No scores recorded yet.", True, (180, 180, 200))
            self.screen.blit(empty, (lb_x + (lb_w - empty.get_width()) // 2, lb_y + 120))
        else:
            for i, entry in enumerate(self.highscores[:5]):
                col = (255, 230, 60) if i == 0 else (225, 235, 250)
                ry = lb_y + 82 + i * 32
                vals = [
                    f"#{i + 1}",
                    str(entry.get("score", 0)),
                    f"{entry.get('distance', 0)}m",
                    str(entry.get("coins", 0)),
                    str(entry.get("max_streak", 0)),
                    str(entry.get("date", "")),
                ]
                for val, (_, cx) in zip(vals, cols):
                    self.screen.blit(self.font_body.render(val, True, col), (lb_x + cx, ry))

        if int(time.time() * 2) % 2 == 0:
            prompt = self.font_sub.render("PRESS SPACE TO PLAY AGAIN", True, (100, 255, 140))
            self.screen.blit(prompt, ((SCREEN_WIDTH - prompt.get_width()) // 2, 530))


if __name__ == "__main__":
    game = ShapeShiftRacerGame()
    game.run()
