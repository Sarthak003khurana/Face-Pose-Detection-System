"""
2.5D Retro Arcade Road & Environment Renderer.
Features smooth dynamic biome transitions (Sunset Drive -> Cyber Night ->
Neon Tokyo -> Plasma Storm) and gentle curved pseudo-3D perspective road segments.
"""

import math
import pygame
from game.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, HORIZON_Y,
    ROAD_BOTTOM_WIDTH, ROAD_TOP_WIDTH,
    COLOR_LANE_MARK, NUM_LANES,
    BIOMES, BIOME_DISTANCE, MAX_Z
)


def _lerp_color(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )


def _to_rgb(c):
    return (int(max(0, min(255, c[0]))), int(max(0, min(255, c[1]))), int(max(0, min(255, c[2]))))


class RoadRenderer:
    def __init__(self):
        self.distance = 0.0
        self.distance_meters = 0.0
        self.segment_length = 40.0
        self.num_segments = 28

        # Dynamic biome state
        self.biome_idx = 0
        self.palette = {k: tuple(float(v) for v in val) for k, val in BIOMES[0].items() if isinstance(val, tuple)}

        # Gentle dynamic road curve state
        self.current_curve = 0.0
        self.target_curve = 0.0
        self._curve_time = 0.0

        # Pre-rendered sky surface (rebuilt cleanly when biome colors shift)
        self.sky_surf = pygame.Surface((SCREEN_WIDTH, HORIZON_Y))
        self._last_sky_key = None
        self._rebuild_sky_surface()

    def _rebuild_sky_surface(self):
        top = _to_rgb(self.palette["sky_top"])
        bot = _to_rgb(self.palette["sky_bottom"])
        key = (top, bot)
        if key == self._last_sky_key:
            return
        self._last_sky_key = key

        for y in range(HORIZON_Y):
            ratio = y / max(HORIZON_Y, 1)
            r = int(top[0] * (1 - ratio) + bot[0] * ratio)
            g = int(top[1] * (1 - ratio) + bot[1] * ratio)
            b = int(top[2] * (1 - ratio) + bot[2] * ratio)
            pygame.draw.line(self.sky_surf, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def reset(self):
        self.distance = 0.0
        self.distance_meters = 0.0
        self.biome_idx = 0
        self.current_curve = 0.0
        self.target_curve = 0.0
        self._curve_time = 0.0

    def get_current_biome_name(self):
        return BIOMES[self.biome_idx]["name"]

    def get_current_accent_color(self):
        return _to_rgb(self.palette["accent"])

    def update(self, speed, dt, distance_meters=None):
        """
        Updates road scroll distance, dynamic biome palette interpolation,
        and gentle road curvature.
        Returns: (biome_name, biome_changed)
        """
        self.distance += speed * dt
        if distance_meters is not None:
            self.distance_meters = distance_meters
        else:
            self.distance_meters += (speed * dt) / 10.0

        # Check biome transition every BIOME_DISTANCE meters
        new_biome_idx = int(self.distance_meters // BIOME_DISTANCE) % len(BIOMES)
        biome_changed = new_biome_idx != self.biome_idx
        self.biome_idx = new_biome_idx

        # Smoothly interpolate palette colors toward active biome
        target_biome = BIOMES[self.biome_idx]
        blend = min(1.0, 2.2 * dt)
        for k in self.palette:
            self.palette[k] = _lerp_color(self.palette[k], target_biome[k], blend)
        self._rebuild_sky_surface()

        # Subtle, gentle road curves (with long straightaway plateaus)
        self._curve_time += dt * (speed / 420.0)
        raw_wave = math.sin(self._curve_time * 0.22) * 0.65 + math.sin(self._curve_time * 0.11) * 0.35
        if abs(raw_wave) < 0.35:
            self.target_curve = 0.0
        else:
            self.target_curve = max(-0.32, min(0.32, raw_wave * 0.45))
        self.current_curve += (self.target_curve - self.current_curve) * min(1.0, 1.2 * dt)

        return BIOMES[self.biome_idx]["name"], biome_changed

    def _curve_dx(self, p):
        """Horizontal pixel shift for perspective curve at depth ratio p in [0, 1]."""
        return self.current_curve * ((1.0 - p) ** 2) * 130.0

    def render(self, screen):
        """Draws the complete 2.5D curved environment with dynamic biome styling."""
        # 1. Sky
        screen.blit(self.sky_surf, (0, 0))

        sky_bot = _to_rgb(self.palette["sky_bottom"])
        sun_col = _to_rgb(self.palette["sun"])
        sun_glow = _to_rgb(self.palette["sun_glow"])
        mountain_col = _to_rgb(self.palette["mountain"])
        ground_col = _to_rgb(self.palette["ground"])
        asphalt_a = _to_rgb(self.palette["asphalt_a"])
        asphalt_b = _to_rgb(self.palette["asphalt_b"])
        curb_light = _to_rgb(self.palette["curb_light"])
        curb_dark = _to_rgb(self.palette["curb_dark"])

        # Subtle horizon parallax from current curve
        horizon_parallax = int(self.current_curve * 55)

        # 2. Synthwave Sun on horizon
        sun_center = (SCREEN_WIDTH // 2 + horizon_parallax, HORIZON_Y - 30)
        sun_radius = 85
        for r_offset in range(12, 0, -3):
            alpha = int(45 * (1.0 - r_offset / 12.0))
            glow_surf = pygame.Surface((sun_radius * 2 + 50, sun_radius * 2 + 50), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*sun_glow, alpha),
                               (sun_radius + 25, sun_radius + 25), sun_radius + r_offset)
            screen.blit(glow_surf, (sun_center[0] - sun_radius - 25, sun_center[1] - sun_radius - 25))

        pygame.draw.circle(screen, sun_col, sun_center, sun_radius)
        for stripe_y in range(sun_center[1] - 10, sun_center[1] + sun_radius, 14):
            thickness = max(2, int((stripe_y - (sun_center[1] - 10)) / 10))
            pygame.draw.line(screen, sky_bot,
                             (sun_center[0] - sun_radius - 5, stripe_y),
                             (sun_center[0] + sun_radius + 5, stripe_y), thickness)

        # 3. Parallax Mountain Silhouettes
        m_shift = int(self.current_curve * 35)
        mountain_pts = [
            (-80, HORIZON_Y),
            (120 + m_shift, HORIZON_Y - 48),
            (260 + m_shift, HORIZON_Y - 22),
            (380 + m_shift, HORIZON_Y - 64),
            (520 + m_shift, HORIZON_Y - 18),
            (680 + m_shift, HORIZON_Y - 74),
            (820 + m_shift, HORIZON_Y - 32),
            (980 + m_shift, HORIZON_Y - 68),
            (1140 + m_shift, HORIZON_Y - 28),
            (SCREEN_WIDTH + 80, HORIZON_Y)
        ]
        pygame.draw.polygon(screen, mountain_col, mountain_pts)

        # 4. Ground Background below horizon
        road_height = SCREEN_HEIGHT - HORIZON_Y
        pygame.draw.rect(screen, ground_col, (0, HORIZON_Y, SCREEN_WIDTH, road_height))

        # 5. Curved Perspective Road Segments
        for i in range(self.num_segments, -1, -1):
            p1 = (i / self.num_segments) ** 2.2
            p2 = ((i + 1) / self.num_segments) ** 2.2

            y1 = int(HORIZON_Y + p1 * road_height)
            y2 = int(HORIZON_Y + p2 * road_height)

            if y2 <= y1 or y1 >= SCREEN_HEIGHT:
                continue

            w1 = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p1
            w2 = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p2

            cx1 = SCREEN_WIDTH / 2.0 + self._curve_dx(p1)
            cx2 = SCREEN_WIDTH / 2.0 + self._curve_dx(p2)

            x1_l = int(cx1 - w1 / 2.0)
            x1_r = int(cx1 + w1 / 2.0)
            x2_l = int(cx2 - w2 / 2.0)
            x2_r = int(cx2 + w2 / 2.0)

            world_z = i * self.segment_length + (self.distance % self.segment_length)
            is_odd = int(world_z / self.segment_length) % 2 == 1

            # Road asphalt
            asphalt_color = asphalt_a if is_odd else asphalt_b
            pygame.draw.polygon(screen, asphalt_color, [
                (x1_l, y1), (x1_r, y1), (x2_r, y2), (x2_l, y2)
            ])

            # Curbs
            curb_w1 = max(w1 * 0.05, 4)
            curb_w2 = max(w2 * 0.05, 4)
            curb_col = curb_light if is_odd else curb_dark

            pygame.draw.polygon(screen, curb_col, [
                (x1_l - curb_w1, y1), (x1_l, y1), (x2_l, y2), (x2_l - curb_w2, y2)
            ])
            pygame.draw.polygon(screen, curb_col, [
                (x1_r, y1), (x1_r + curb_w1, y1), (x2_r + curb_w2, y2), (x2_r, y2)
            ])

            # Dashed Lane Dividers
            if is_odd:
                lane_mark_w = max(w1 * 0.012, 2)
                for lane_idx in range(1, NUM_LANES):
                    ratio = lane_idx / NUM_LANES
                    lx1 = x1_l + w1 * ratio
                    lx2 = x2_l + w2 * ratio
                    pygame.draw.polygon(screen, COLOR_LANE_MARK, [
                        (lx1 - lane_mark_w, y1), (lx1 + lane_mark_w, y1),
                        (lx2 + lane_mark_w, y2), (lx2 - lane_mark_w, y2)
                    ])

    def get_screen_coords(self, normalized_x, world_z):
        """
        Converts normalized road coordinates (X in [-1.0, 1.0], Z in [0, MAX_Z])
        to 2D curved perspective screen coordinates (screen_x, screen_y, scale).
        """
        depth = max(world_z + 100.0, 1.0)
        scale = 140.0 / depth
        raw_p = max(0.0, min(1.0, 1.0 - (world_z / MAX_Z)))
        p = raw_p ** 2.2

        road_h = SCREEN_HEIGHT - HORIZON_Y
        screen_y = HORIZON_Y + p * road_h

        road_w = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p
        road_center_x = SCREEN_WIDTH / 2.0 + self._curve_dx(p)
        screen_x = road_center_x + (normalized_x * (road_w / 2.0))

        return screen_x, screen_y, scale
