"""
2.5D Retro Arcade Road & Environment Renderer.
Simulates a classic pseudo-3D perspective road with moving road segments,
side curbs, lane dividers, and a synthwave horizon.
"""

import math
import pygame
from game.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, HORIZON_Y,
    ROAD_BOTTOM_WIDTH, ROAD_TOP_WIDTH,
    COLOR_SKY_TOP, COLOR_SKY_BOTTOM,
    COLOR_SUN, COLOR_SUN_GLOW,
    COLOR_MOUNTAIN, COLOR_ROAD,
    COLOR_CURB_LIGHT, COLOR_CURB_DARK,
    COLOR_LANE_MARK, NUM_LANES
)


class RoadRenderer:
    def __init__(self):
        self.distance = 0.0
        self.segment_length = 40.0
        self.num_segments = 28
        self._create_sky_surface()

    def _create_sky_surface(self):
        """Pre-renders a vertical gradient sky background for maximum FPS."""
        self.sky_surf = pygame.Surface((SCREEN_WIDTH, HORIZON_Y))
        for y in range(HORIZON_Y):
            ratio = y / max(HORIZON_Y, 1)
            r = int(COLOR_SKY_TOP[0] * (1 - ratio) + COLOR_SKY_BOTTOM[0] * ratio)
            g = int(COLOR_SKY_TOP[1] * (1 - ratio) + COLOR_SKY_BOTTOM[1] * ratio)
            b = int(COLOR_SKY_TOP[2] * (1 - ratio) + COLOR_SKY_BOTTOM[2] * ratio)
            pygame.draw.line(self.sky_surf, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def update(self, speed, dt):
        """Updates road scroll distance based on current speed."""
        self.distance += speed * dt

    def render(self, screen):
        """Draws the complete 2.5D environment: sky, sun, mountains, and road."""
        # 1. Sky
        screen.blit(self.sky_surf, (0, 0))

        # 2. Synthwave Sun on horizon
        sun_center = (SCREEN_WIDTH // 2, HORIZON_Y - 30)
        sun_radius = 85
        # Sun outer glow
        for r_offset in range(12, 0, -3):
            alpha = int(45 * (1.0 - r_offset / 12.0))
            glow_surf = pygame.Surface((sun_radius * 2 + 50, sun_radius * 2 + 50), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*COLOR_SUN_GLOW, alpha),
                               (sun_radius + 25, sun_radius + 25), sun_radius + r_offset)
            screen.blit(glow_surf, (sun_center[0] - sun_radius - 25, sun_center[1] - sun_radius - 25))

        # Main sun disk
        pygame.draw.circle(screen, COLOR_SUN, sun_center, sun_radius)
        # Synthwave horizontal stripes across the sun
        for stripe_y in range(sun_center[1] - 10, sun_center[1] + sun_radius, 14):
            thickness = max(2, int((stripe_y - (sun_center[1] - 10)) / 10))
            pygame.draw.line(screen, COLOR_SKY_BOTTOM,
                             (sun_center[0] - sun_radius - 5, stripe_y),
                             (sun_center[0] + sun_radius + 5, stripe_y), thickness)

        # 3. Mountain Silhouettes
        mountain_pts = [
            (0, HORIZON_Y),
            (120, HORIZON_Y - 45),
            (260, HORIZON_Y - 20),
            (380, HORIZON_Y - 60),
            (520, HORIZON_Y - 15),
            (680, HORIZON_Y - 70),
            (820, HORIZON_Y - 30),
            (980, HORIZON_Y - 65),
            (1140, HORIZON_Y - 25),
            (SCREEN_WIDTH, HORIZON_Y)
        ]
        pygame.draw.polygon(screen, COLOR_MOUNTAIN, mountain_pts)

        # 4. Perspective Grid Sides (outside the road)
        grid_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT - HORIZON_Y))
        grid_surf.fill((12, 8, 20))
        # Draw perspective road segments
        road_height = SCREEN_HEIGHT - HORIZON_Y

        # Render road segments from horizon downward
        for i in range(self.num_segments, -1, -1):
            p1 = (i / self.num_segments) ** 2.2
            p2 = ((i + 1) / self.num_segments) ** 2.2

            y1 = int(HORIZON_Y + p1 * road_height)
            y2 = int(HORIZON_Y + p2 * road_height)

            if y2 <= y1 or y1 >= SCREEN_HEIGHT:
                continue

            w1 = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p1
            w2 = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p2

            x1_l = int((SCREEN_WIDTH - w1) / 2)
            x1_r = int((SCREEN_WIDTH + w1) / 2)
            x2_l = int((SCREEN_WIDTH - w2) / 2)
            x2_r = int((SCREEN_WIDTH + w2) / 2)

            # Segment alternating color based on world distance
            world_z = i * self.segment_length + (self.distance % self.segment_length)
            is_odd = int(world_z / self.segment_length) % 2 == 1

            # Road asphalt
            asphalt_color = (28, 22, 40) if is_odd else (22, 16, 32)
            pygame.draw.polygon(screen, asphalt_color, [
                (x1_l, y1), (x1_r, y1), (x2_r, y2), (x2_l, y2)
            ])

            # Rumble strips / Curbs
            curb_w1 = max(w1 * 0.05, 4)
            curb_w2 = max(w2 * 0.05, 4)
            curb_col = COLOR_CURB_LIGHT if is_odd else COLOR_CURB_DARK

            # Left curb
            pygame.draw.polygon(screen, curb_col, [
                (x1_l - curb_w1, y1), (x1_l, y1), (x2_l, y2), (x2_l - curb_w2, y2)
            ])
            # Right curb
            pygame.draw.polygon(screen, curb_col, [
                (x1_r, y1), (x1_r + curb_w1, y1), (x2_r + curb_w2, y2), (x2_r, y2)
            ])

            # Dashed Lane Dividers (for 3 lanes: 2 dividing lines)
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
        to 2D screen coordinates (screen_x, screen_y, scale).
        """
        depth = max(world_z + 100.0, 1.0)
        scale = 140.0 / depth
        p = (1.0 - (world_z / 1200.0)) ** 2.2
        p = max(0.0, min(1.0, p))

        road_h = SCREEN_HEIGHT - HORIZON_Y
        screen_y = HORIZON_Y + p * road_h

        road_w = ROAD_TOP_WIDTH + (ROAD_BOTTOM_WIDTH - ROAD_TOP_WIDTH) * p
        road_center_x = SCREEN_WIDTH / 2.0
        screen_x = road_center_x + (normalized_x * (road_w / 2.0))

        return screen_x, screen_y, scale
