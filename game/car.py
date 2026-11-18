"""
Detailed Rear-View Morphing Sports Car Renderer.
Draws a recognizable 3D-style GT sports car (tires, rear bumper, diffuser,
twin exhaust pipes, side mirrors, rear windshield, LED taillights, spoiler)
that smoothly morphs between GT Coupe (NORMAL), Slammed Supercar (FLAT),
and Slim Concept Car (NARROW).
"""

import random
import pygame
from game.config import (
    CAR_SHAPES, SCREEN_WIDTH, SCREEN_HEIGHT
)


class ExhaustParticle:
    """Glowing exhaust spark/flame particle emitted from twin tailpipes."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.8, 0.8)
        self.vy = random.uniform(3.0, 6.5)
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.06, 0.11)
        self.size = random.uniform(2.5, 5.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        self.size = max(0.5, self.size - 0.15)
        return self.life > 0

    def draw(self, surface):
        alpha = int(220 * max(0.0, self.life))
        r = max(1, int(self.size))
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), (r, r), r)
        surface.blit(s, (self.x - r, self.y - r))


class Car:
    def __init__(self, base_y=None):
        self.norm_x = 0.0
        self.target_norm_x = 0.0
        self.steer_velocity = 0.0

        self.current_shape_key = "NORMAL"
        cfg = CAR_SHAPES[self.current_shape_key]
        self.w = float(cfg["width"])
        self.h = float(cfg["height"])
        self.target_w = self.w
        self.target_h = self.h

        self.screen_x = SCREEN_WIDTH / 2.0
        self.screen_y = base_y if base_y else (SCREEN_HEIGHT - 65)

        self.color_primary = list(cfg["color_primary"])
        self.particles = []
        self.wheel_phase = 0.0

        # Jump physics (triggered by Eyebrow Raise or J/Space)
        self.jump_y = 0.0
        self.jump_vy = 0.0
        self.jump_gravity = 1350.0

        # Energy Shield + Nitro Boost (triggered by Wink or Shift)
        self.shield_timer = 0.0
        self.shield_cooldown = 0.0
        self.shield_pulse = 0.0

    @property
    def is_airborne(self):
        return self.jump_y > 18.0

    @property
    def is_shielded(self):
        return self.shield_timer > 0.0

    def trigger_jump(self):
        """Launches the car into the air if grounded. Returns True if jumped."""
        if self.jump_y <= 1.0:
            self.jump_vy = 540.0
            return True
        return False

    def activate_shield(self, duration=3.5, cooldown=9.0):
        """Activates glowing Energy Shield + Nitro Boost if off cooldown."""
        if self.shield_timer <= 0.0 and self.shield_cooldown <= 0.0:
            self.shield_timer = duration
            self.shield_cooldown = cooldown
            return True
        return False

    def reset_abilities(self):
        self.jump_y = 0.0
        self.jump_vy = 0.0
        self.shield_timer = 0.0
        self.shield_cooldown = 0.0

    def set_target_steer(self, target_norm_x):
        """Sets target steering offset from vision or keyboard."""
        self.target_norm_x = max(-0.82, min(0.82, target_norm_x))

    def set_shape(self, shape_key):
        """Switches vehicle target morph shape."""
        if shape_key in CAR_SHAPES and shape_key != self.current_shape_key:
            self.current_shape_key = shape_key
            cfg = CAR_SHAPES[shape_key]
            self.target_w = float(cfg["width"])
            self.target_h = float(cfg["height"])
            return True
        return False

    def update(self, road_renderer, dt):
        """Interpolates vehicle position, morph dimensions, body roll, and exhaust particles."""
        prev_x = self.norm_x
        self.norm_x += (self.target_norm_x - self.norm_x) * min(1.0, 14.0 * dt)
        self.steer_velocity = (self.norm_x - prev_x) / max(dt, 1e-4)

        sx, sy, _ = road_renderer.get_screen_coords(self.norm_x, 40.0)
        self.screen_x = sx
        self.screen_y = sy

        # Smooth shape tweening
        morph_speed = 18.0 * dt
        self.w += (self.target_w - self.w) * min(1.0, morph_speed)
        self.h += (self.target_h - self.h) * min(1.0, morph_speed)

        # Smooth paint color transition
        target_col = CAR_SHAPES[self.current_shape_key]["color_primary"]
        for c in range(3):
            self.color_primary[c] += (target_col[c] - self.color_primary[c]) * min(1.0, 10.0 * dt)

        # Wheel tread scroll
        self.wheel_phase = (self.wheel_phase + 25.0 * dt) % 8.0

        # Jump physics integration
        if self.jump_y > 0.0 or self.jump_vy > 0.0:
            self.jump_y += self.jump_vy * dt
            self.jump_vy -= self.jump_gravity * dt
            if self.jump_y <= 0.0:
                self.jump_y = 0.0
                self.jump_vy = 0.0

        # Shield & Nitro Boost timers
        if self.shield_timer > 0.0:
            self.shield_timer = max(0.0, self.shield_timer - dt)
            self.shield_pulse += 10.0 * dt
        elif self.shield_cooldown > 0.0:
            self.shield_cooldown = max(0.0, self.shield_cooldown - dt)

        # Twin exhaust flame particles (intense cyan/gold nitro flames when shielded!)
        spawn_prob = 0.95 if self.is_shielded else 0.70
        if random.random() < spawn_prob:
            pipe_offset = self.w * 0.24
            rear_y = (self.screen_y - self.jump_y) - max(4, self.h * 0.12)
            if self.is_shielded:
                flame_col = random.choice([(0, 240, 255), (120, 255, 255), (255, 230, 80)])
            else:
                flame_col = random.choice([(255, 160, 30), (80, 200, 255), (255, 90, 20)])
            self.particles.append(ExhaustParticle(self.screen_x - pipe_offset, rear_y, flame_col))
            self.particles.append(ExhaustParticle(self.screen_x + pipe_offset, rear_y, flame_col))

        self.particles = [p for p in self.particles if p.update()]

    def _shade(self, color, delta):
        return tuple(max(0, min(255, int(c + delta))) for c in color)

    def render(self, screen):
        """Draws a detailed rear-view sports car that smoothly morphs, jumps, and activates shield."""
        cx = self.screen_x
        ground_y = self.screen_y
        cy = ground_y - self.jump_y
        w = self.w
        h = self.h
        half_w = w / 2.0

        base_col = tuple(int(c) for c in self.color_primary)
        dark_col = self._shade(base_col, -65)
        mid_col = self._shade(base_col, -25)
        light_col = self._shade(base_col, 45)

        # Subtle body lean when steering left/right
        lean = max(-6.0, min(6.0, self.steer_velocity * 1.8))

        # 1. Ground Contact Shadow (stays on asphalt and shrinks as car jumps higher!)
        air_factor = max(0.35, 1.0 - self.jump_y / 180.0)
        shadow_w = max(12, int(w * 1.18 * air_factor))
        shadow_h = max(6, int(h * 0.22 * air_factor))
        shadow_surf = pygame.Surface((shadow_w + 20, shadow_h + 10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(145 * air_factor)), (10, 5, shadow_w, shadow_h))
        screen.blit(shadow_surf, (cx - shadow_w / 2 - 10, ground_y - shadow_h / 2 - 2))

        # 2. Rear Rubber Tires (Left & Right)
        tire_w = max(8, int(w * 0.20))
        tire_h = max(10, int(min(h * 0.42, 26)))
        tire_top = int(cy - tire_h + 2)
        left_tire_rect = pygame.Rect(int(cx - half_w - 2), tire_top, tire_w, tire_h)
        right_tire_rect = pygame.Rect(int(cx + half_w - tire_w + 2), tire_top, tire_w, tire_h)

        for trect in (left_tire_rect, right_tire_rect):
            pygame.draw.rect(screen, (20, 20, 26), trect, border_radius=3)
            # Subtle animated tire tread lines
            for ty in range(int(self.wheel_phase) % 5, trect.height - 2, 5):
                pygame.draw.line(screen, (45, 45, 55),
                                 (trect.left + 2, trect.top + ty),
                                 (trect.right - 2, trect.top + ty), 1)
            pygame.draw.rect(screen, (60, 60, 72), trect, width=1, border_radius=3)

        # 3. Exhaust Particles behind bumper
        for p in self.particles:
            p.draw(screen)

        # Vertical landmarks of car silhouette
        chassis_bottom = cy - max(4.0, tire_h * 0.30)
        bumper_top = cy - h * 0.34
        trunk_top = cy - h * 0.56
        roof_top = cy - h

        # 4. Main Lower Car Body / Sculpted Rear Fenders
        body_pts = [
            (cx - half_w, chassis_bottom - 2),
            (cx - half_w * 1.02, bumper_top),
            (cx - half_w * 0.90 + lean * 0.4, trunk_top),
            (cx + half_w * 0.90 + lean * 0.4, trunk_top),
            (cx + half_w * 1.02, bumper_top),
            (cx + half_w, chassis_bottom - 2),
        ]
        pygame.draw.polygon(screen, base_col, body_pts)

        # Lower bumper shadow band for 3D depth
        bumper_band = [
            (cx - half_w, chassis_bottom - 2),
            (cx - half_w * 1.01, bumper_top + 3),
            (cx + half_w * 1.01, bumper_top + 3),
            (cx + half_w, chassis_bottom - 2),
        ]
        pygame.draw.polygon(screen, mid_col, bumper_band)
        pygame.draw.polygon(screen, light_col, body_pts, width=1)

        # 5. Lower Rear Diffuser & Twin Chrome Exhaust Pipes (drawn on top of lower bumper)
        diffuser_h = max(5, int(min(h * 0.16, 12)))
        diffuser_rect = pygame.Rect(
            int(cx - half_w * 0.72),
            int(chassis_bottom - diffuser_h - 1),
            int(w * 0.72),
            diffuser_h
        )
        pygame.draw.rect(screen, (26, 28, 38), diffuser_rect, border_radius=2)
        for fin_ratio in (-0.10, 0.0, 0.10):
            fx = int(cx + w * fin_ratio)
            pygame.draw.line(screen, (70, 76, 95), (fx, diffuser_rect.top + 1), (fx, diffuser_rect.bottom - 1), 1)

        pipe_r = max(2, int(min(w * 0.048, diffuser_h * 0.45, 5)))
        pipe_y = int(diffuser_rect.centery)
        for sign in (-1, 1):
            px = int(cx + sign * w * 0.24)
            pygame.draw.circle(screen, (200, 210, 225), (px, pipe_y), pipe_r)
            pygame.draw.circle(screen, (30, 20, 25), (px, pipe_y), max(1, pipe_r - 1))
            if pipe_r >= 3:
                pygame.draw.circle(screen, (255, 130, 35), (px, pipe_y), max(1, pipe_r - 2))

        # 6. Side Mirrors (visible when car isn't super narrow)
        if w > 48:
            mirror_w = max(5, int(w * 0.08))
            mirror_h = max(3, int(min(h * 0.10, 6)))
            mirror_y = int(trunk_top - mirror_h * 0.6)
            pygame.draw.ellipse(screen, base_col, (int(cx - half_w * 0.90 - mirror_w + lean * 0.5), mirror_y, mirror_w, mirror_h))
            pygame.draw.ellipse(screen, base_col, (int(cx + half_w * 0.90 + lean * 0.5), mirror_y, mirror_w, mirror_h))

        # 7. Upper Greenhouse Cabin & Curved Roofline
        roof_half_w = half_w * 0.62
        cabin_pts = [
            (cx - half_w * 0.84 + lean * 0.5, trunk_top + 1),
            (cx - roof_half_w + lean, roof_top),
            (cx + roof_half_w + lean, roof_top),
            (cx + half_w * 0.84 + lean * 0.5, trunk_top + 1),
        ]
        pygame.draw.polygon(screen, dark_col, cabin_pts)
        # Roof metallic top highlight
        pygame.draw.line(screen, light_col,
                         (cx - roof_half_w + lean + 2, roof_top + 1),
                         (cx + roof_half_w + lean - 2, roof_top + 1), 2)

        # Rear Windshield Glass
        glass_margin_x = max(3.0, w * 0.08)
        glass_top = roof_top + max(2.5, h * 0.07)
        glass_bot = trunk_top - max(1.5, h * 0.04)
        if glass_bot > glass_top + 2:
            glass_pts = [
                (cx - half_w * 0.72 + lean * 0.55, glass_bot),
                (cx - roof_half_w + glass_margin_x * 0.4 + lean, glass_top),
                (cx + roof_half_w - glass_margin_x * 0.4 + lean, glass_top),
                (cx + half_w * 0.72 + lean * 0.55, glass_bot),
            ]
            pygame.draw.polygon(screen, (25, 40, 65), glass_pts)
            # Diagonal glossy reflection streak on rear glass
            ref_x = cx - roof_half_w * 0.35 + lean * 0.8
            pygame.draw.line(screen, (140, 210, 255),
                             (ref_x, glass_top + 2),
                             (ref_x - w * 0.10, glass_bot - 1), 2)
            pygame.draw.line(screen, (90, 150, 200),
                             (ref_x + w * 0.12, glass_top + 2),
                             (ref_x + w * 0.02, glass_bot - 1), 1)
            pygame.draw.polygon(screen, (100, 150, 200), glass_pts, width=1)

        # 8. Rear Taillight Clusters (Left & Right LED Brake Lights) & License Plate
        tl_y = int(trunk_top + (bumper_top - trunk_top) * 0.22)
        tl_h = max(4, int((bumper_top - trunk_top) * 0.52))
        tl_w = max(7, int(w * 0.25))

        # Left & Right Taillight housings
        left_tl = pygame.Rect(int(cx - half_w * 0.88), tl_y, tl_w, tl_h)
        right_tl = pygame.Rect(int(cx + half_w * 0.88 - tl_w), tl_y, tl_w, tl_h)

        for tl_rect in (left_tl, right_tl):
            # Outer dark housing
            pygame.draw.rect(screen, (60, 10, 15), tl_rect, border_radius=2)
            # Bright glowing red brake LED bar
            inner = tl_rect.inflate(-2, -2)
            if inner.width > 1 and inner.height > 1:
                pygame.draw.rect(screen, (255, 25, 45), inner, border_radius=2)
                # Bright hot-center glow strip
                core_h = max(1, inner.height // 3)
                pygame.draw.rect(screen, (255, 160, 160),
                                 (inner.left + 1, inner.centery - core_h // 2, max(1, inner.width - 2), core_h))

        # Center License Plate
        plate_w = max(10, int(w * 0.22))
        plate_h = max(5, int(min(tl_h * 1.05, 12)))
        plate_rect = pygame.Rect(int(cx - plate_w / 2), int(tl_y + (tl_h - plate_h) / 2), plate_w, plate_h)
        pygame.draw.rect(screen, (240, 240, 245), plate_rect, border_radius=1)
        pygame.draw.rect(screen, (40, 60, 160), (plate_rect.left, plate_rect.top, plate_rect.width, max(1, plate_h // 4)))

        # 9. Sleek Rear Spoiler Wing (Horizontal racing wing on twin struts)
        spoiler_w = w * 0.86
        strut_h = max(3, int(min(h * 0.12, 8)))
        wing_y = int(trunk_top - strut_h)
        wing_h = max(3, int(min(h * 0.08, 5)))

        # Twin spoiler struts
        for sign in (-1, 1):
            st_x = int(cx + sign * (spoiler_w * 0.28) + lean * 0.4)
            pygame.draw.line(screen, (25, 25, 35), (st_x, int(trunk_top + 1)), (st_x, wing_y + wing_h), 2)

        # Horizontal wing blade
        wing_rect = pygame.Rect(int(cx - spoiler_w / 2 + lean * 0.4), wing_y, int(spoiler_w), wing_h)
        pygame.draw.rect(screen, dark_col, wing_rect, border_radius=2)
        pygame.draw.rect(screen, light_col, wing_rect, width=1, border_radius=2)

        # 10. Glowing Energy Shield Dome (when activated via Wink / Shift)
        if self.is_shielded:
            import math
            pulse = 0.5 + 0.5 * math.sin(self.shield_pulse)
            dome_w = int(w * 1.38)
            dome_h = int(h * 1.48) + 18
            dome_surf = pygame.Surface((dome_w + 16, dome_h + 16), pygame.SRCALPHA)
            fill_alpha = int(42 + 28 * pulse)
            ring_alpha = int(170 + 80 * pulse)
            pygame.draw.ellipse(dome_surf, (0, 235, 255, fill_alpha), (8, 8, dome_w, dome_h))
            pygame.draw.ellipse(dome_surf, (120, 255, 255, ring_alpha), (8, 8, dome_w, dome_h), width=3)
            pygame.draw.ellipse(dome_surf, (255, 230, 80, int(120 * pulse)), (12, 12, dome_w - 8, dome_h - 8), width=1)
            screen.blit(dome_surf, (int(cx - dome_w / 2 - 8), int(cy - dome_h + 10)))

    def get_hitbox(self):
        """Returns the current screen-space AABB bounding box (left, top, width, height)."""
        return pygame.Rect(
            int(self.screen_x - self.w / 2.0),
            int((self.screen_y - self.jump_y) - self.h),
            int(self.w),
            int(self.h)
        )
