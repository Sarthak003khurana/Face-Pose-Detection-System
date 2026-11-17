"""
Morphing Player Car with dynamic vector body rendering, smooth shape tweening,
and particle exhaust trails.
"""

import math
import random
import pygame
from game.config import (
    CAR_SHAPES, SCREEN_WIDTH, SCREEN_HEIGHT,
    COLOR_NEON_CYAN, COLOR_NEON_YELLOW
)


class Particle:
    """Exhaust jet particle emitted by the car."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.2, 1.2)
        self.vy = random.uniform(2.5, 6.0)
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.08)
        self.size = random.uniform(3.0, 6.0)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        self.size = max(0.5, self.size - 0.15)
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * self.life)
        s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), (int(self.size), int(self.size)), int(self.size))
        surface.blit(s, (self.x - self.size, self.y - self.size))


class Car:
    def __init__(self, base_y=None):
        self.norm_x = 0.0       # Normalized lane coordinate [-0.85, 0.85]
        self.target_norm_x = 0.0
        
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
        self.glow_angle = 0.0

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
        """Interpolates vehicle position, size, and updates particles."""
        # 1. Smooth lateral steering (ease-out)
        self.norm_x += (self.target_norm_x - self.norm_x) * min(1.0, 14.0 * dt)
        
        # Calculate screen position based on road geometry
        sx, sy, _ = road_renderer.get_screen_coords(self.norm_x, 40.0)
        self.screen_x = sx
        self.screen_y = sy

        # 2. Smooth shape tweening
        morph_speed = 18.0 * dt
        self.w += (self.target_w - self.w) * min(1.0, morph_speed)
        self.h += (self.target_h - self.h) * min(1.0, morph_speed)

        # 3. Smooth color transition
        target_col = CAR_SHAPES[self.current_shape_key]["color_primary"]
        for c in range(3):
            self.color_primary[c] += (target_col[c] - self.color_primary[c]) * min(1.0, 10.0 * dt)

        # 4. Particles (Exhaust jets)
        self.glow_angle += 10.0 * dt
        if random.random() < 0.75:
            # Emit from rear left and right
            left_rear_x = self.screen_x - self.w * 0.35
            right_rear_x = self.screen_x + self.w * 0.35
            rear_y = self.screen_y + 4
            col = (int(self.color_primary[0]), int(self.color_primary[1]), int(self.color_primary[2]))
            self.particles.append(Particle(left_rear_x, rear_y, col))
            self.particles.append(Particle(right_rear_x, rear_y, col))

        # Update active particles
        self.particles = [p for p in self.particles if p.update()]

    def render(self, screen):
        """Draws the dynamic morphing vector car and exhaust glow."""
        # Draw exhaust particles behind the car
        for p in self.particles:
            p.draw(screen)

        cx = self.screen_x
        cy = self.screen_y
        w = self.w
        h = self.h
        col = tuple(int(c) for c in self.color_primary)

        # 1. Ground Shadow (ambient occlusion under the car)
        shadow_w = int(w * 1.15)
        shadow_h = max(int(h * 0.28), 6)
        shadow_surf = pygame.Surface((shadow_w * 2, shadow_h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 110),
                            (shadow_w // 2, shadow_h // 2, shadow_w, shadow_h))
        screen.blit(shadow_surf, (cx - shadow_w, cy - shadow_h // 2))

        # 2. Main Car Body Polygons (morphs depending on shape)
        half_w = w / 2.0
        roof_w = half_w * 0.55
        body_y_bottom = cy
        body_y_top = cy - h
        roof_y = cy - h * 0.88
        hood_y = cy - h * 0.40

        # Body lower chassis
        chassis_pts = [
            (cx - half_w, body_y_bottom),
            (cx + half_w, body_y_bottom),
            (cx + half_w * 0.88, hood_y),
            (cx - half_w * 0.88, hood_y)
        ]
        pygame.draw.polygon(screen, col, chassis_pts)

        # Cabin / Roof
        cabin_pts = [
            (cx - half_w * 0.88, hood_y),
            (cx + half_w * 0.88, hood_y),
            (cx + roof_w, roof_y),
            (cx - roof_w, roof_y)
        ]
        # Darker tint for cabin
        cabin_col = (max(0, col[0] - 60), max(0, col[1] - 60), max(0, col[2] - 60))
        pygame.draw.polygon(screen, cabin_col, cabin_pts)

        # Windshield glass (tinted cyan/white)
        windshield_pts = [
            (cx - half_w * 0.70, hood_y - 2),
            (cx + half_w * 0.70, hood_y - 2),
            (cx + roof_w * 0.80, roof_y + 3),
            (cx - roof_w * 0.80, roof_y + 3)
        ]
        pygame.draw.polygon(screen, (220, 245, 255), windshield_pts)

        # Neon Accent Outline
        outline_pts = [
            (cx - half_w, body_y_bottom),
            (cx - half_w * 0.88, hood_y),
            (cx - roof_w, roof_y),
            (cx + roof_w, roof_y),
            (cx + half_w * 0.88, hood_y),
            (cx + half_w, body_y_bottom)
        ]
        pygame.draw.lines(screen, (255, 255, 255), False, outline_pts, 2)

        # 3. Headlights / Taillights
        # Glowing red rear taillights strip
        taillight_y = body_y_bottom - max(int(h * 0.15), 3)
        tail_h = max(int(h * 0.12), 3)
        pygame.draw.rect(screen, (255, 30, 40),
                         (cx - half_w * 0.80, taillight_y, half_w * 1.60, tail_h),
                         border_radius=2)

        # Twin license plate / exhaust core in center
        pygame.draw.rect(screen, (255, 220, 50),
                         (cx - 8, body_y_bottom - 5, 16, 4))

        # 4. Shape-Specific Visual Accents
        if self.current_shape_key == "FLAT":
            # Extra aerodynamic winglets on left and right
            wing_w = w * 0.18
            pygame.draw.polygon(screen, (255, 240, 100), [
                (cx - half_w, body_y_bottom),
                (cx - half_w - wing_w, body_y_bottom - 6),
                (cx - half_w, hood_y)
            ])
            pygame.draw.polygon(screen, (255, 240, 100), [
                (cx + half_w, body_y_bottom),
                (cx + half_w + wing_w, body_y_bottom - 6),
                (cx + half_w, hood_y)
            ])
        elif self.current_shape_key == "NARROW":
            # Vertical twin tailfins protruding upward
            fin_h = h * 0.25
            pygame.draw.polygon(screen, (100, 255, 255), [
                (cx - roof_w, roof_y),
                (cx - roof_w - 4, roof_y - fin_h),
                (cx - roof_w + 3, roof_y)
            ])
            pygame.draw.polygon(screen, (100, 255, 255), [
                (cx + roof_w, roof_y),
                (cx + roof_w + 4, roof_y - fin_h),
                (cx + roof_w - 3, roof_y)
            ])

    def get_hitbox(self):
        """Returns the current screen-space AABB bounding box (left, top, width, height)."""
        return pygame.Rect(
            int(self.screen_x - self.w / 2.0),
            int(self.screen_y - self.h),
            int(self.w),
            int(self.h)
        )
