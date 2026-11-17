"""
Obstacle classes and spawning manager for Shape-Shift Racer.
Supports Low Clearance Beams, Narrow Pillar Gates, and Lane Roadblocks.
"""

import random
import pygame
from game.config import (
    MAX_Z, CAR_Z, LOW_BEAM_GAP_HEIGHT, PILLAR_GAP_WIDTH,
    COLOR_NEON_YELLOW, COLOR_NEON_RED, COLOR_NEON_CYAN,
    SCREEN_WIDTH, SCREEN_HEIGHT
)


class Obstacle:
    """Base class for pseudo-3D approaching road obstacles."""
    def __init__(self, z=MAX_Z, norm_x=0.0):
        self.z = float(z)
        self.norm_x = float(norm_x)
        self.passed = False
        self.collided = False
        self.type_name = "BASE"

    def update(self, speed, dt):
        """Moves obstacle toward player camera along Z axis."""
        self.z -= speed * dt
        return self.z > -20.0

    def check_collision(self, car):
        """Overridden by subclasses to compute accurate AABB collision."""
        return False

    def draw(self, screen, road_renderer):
        """Overridden by subclasses for 2.5D rendering."""
        pass


class LowBeam(Obstacle):
    """
    Overhanging horizontal barrier.
    Requires player to be in FLAT shape (Smile) to slide underneath.
    """
    def __init__(self, z=MAX_Z):
        super().__init__(z=z, norm_x=0.0)
        self.type_name = "LOW_BEAM"
        self.prompt_text = "▼ SLIDE FLAT (SMILE) ▼"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False
        
        # Check Z-depth window near car
        if abs(self.z - CAR_Z) < 22.0:
            # If car height exceeds clearance threshold, it crashes!
            if car.h > (LOW_BEAM_GAP_HEIGHT + 3.0):
                self.collided = True
                return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return

        # Get screen coordinates for left and right edges of road at this Z
        lx, ly, scale = road_renderer.get_screen_coords(-0.95, self.z)
        rx, ry, _ = road_renderer.get_screen_coords(0.95, self.z)
        cx, cy, _ = road_renderer.get_screen_coords(0.0, self.z)

        beam_w = rx - lx
        beam_h = max(int(65 * scale), 8)
        gap_h = max(int(LOW_BEAM_GAP_HEIGHT * 1.5 * scale), 4)

        beam_top = int(ly - gap_h - beam_h)
        beam_bottom = int(ly - gap_h)

        # 1. Solid Horizontal Beam Bar
        beam_rect = pygame.Rect(int(lx), beam_top, int(beam_w), beam_h)
        pygame.draw.rect(screen, (220, 20, 60), beam_rect, border_radius=3)
        pygame.draw.rect(screen, (255, 230, 0), beam_rect, width=max(1, int(2 * scale)))

        # 2. Hazard Stripes
        stripe_step = max(int(24 * scale), 6)
        for sx in range(int(lx), int(rx), stripe_step):
            pygame.draw.line(screen, (40, 20, 30),
                             (sx, beam_bottom),
                             (min(sx + stripe_step // 2, int(rx)), beam_top),
                             max(1, int(3 * scale)))

        # 3. Side Support Pillars (holding the beam)
        pillar_w = max(int(16 * scale), 3)
        pygame.draw.rect(screen, (100, 100, 120),
                         (int(lx) - pillar_w, beam_top, pillar_w, int(ly - beam_top)))
        pygame.draw.rect(screen, (100, 100, 120),
                         (int(rx), beam_top, pillar_w, int(ry - beam_top)))

        # 4. Neon Prompt Text
        if scale > 0.45:
            font_size = max(12, int(18 * scale))
            font = pygame.font.SysFont("Impact", font_size)
            txt = font.render(self.prompt_text, True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, beam_top + (beam_h - txt.get_height()) // 2))


class PillarGate(Obstacle):
    """
    Twin vertical concrete pillars leaving only a narrow vertical slit.
    Requires player to be in NARROW shape (Mouth Open / O) to squeeze through.
    """
    def __init__(self, z=MAX_Z, norm_x=0.0):
        super().__init__(z=z, norm_x=norm_x)
        self.type_name = "PILLAR_GATE"
        self.prompt_text = "► SQUEEZE (O-MOUTH) ◄"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False

        if abs(self.z - CAR_Z) < 22.0:
            # Check if car is aligned with the gap
            # If car width exceeds gap width, it hits the pillars!
            if car.w > (PILLAR_GAP_WIDTH + 4.0):
                self.collided = True
                return True
            # Also ensure car is laterally centered with the slit
            if abs(car.norm_x - self.norm_x) > 0.32:
                self.collided = True
                return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return

        cx, cy, scale = road_renderer.get_screen_coords(self.norm_x, self.z)
        lx, ly, _ = road_renderer.get_screen_coords(-0.95, self.z)
        rx, ry, _ = road_renderer.get_screen_coords(0.95, self.z)

        pillar_h = max(int(140 * scale), 16)
        gap_w = max(int(PILLAR_GAP_WIDTH * 1.5 * scale), 8)
        half_gap = gap_w / 2.0

        # Left Pillar: from left road edge to (cx - half_gap)
        left_pillar_w = max(1, (cx - half_gap) - lx)
        left_rect = pygame.Rect(int(lx), int(cy - pillar_h), int(left_pillar_w), pillar_h)
        pygame.draw.rect(screen, (30, 160, 220), left_rect, border_radius=2)
        pygame.draw.rect(screen, (0, 255, 255), left_rect, width=max(1, int(2 * scale)))

        # Right Pillar: from (cx + half_gap) to right road edge
        right_pillar_w = max(1, rx - (cx + half_gap))
        right_rect = pygame.Rect(int(cx + half_gap), int(cy - pillar_h), int(right_pillar_w), pillar_h)
        pygame.draw.rect(screen, (30, 160, 220), right_rect, border_radius=2)
        pygame.draw.rect(screen, (0, 255, 255), right_rect, width=max(1, int(2 * scale)))

        # Top Crossbar connecting pillars
        bar_h = max(int(14 * scale), 3)
        pygame.draw.rect(screen, (20, 80, 140),
                         (int(lx), int(cy - pillar_h), int(rx - lx), bar_h))

        # Neon Arrows pointing into the narrow slit
        if scale > 0.45:
            font_size = max(11, int(16 * scale))
            font = pygame.font.SysFont("Impact", font_size)
            txt = font.render(self.prompt_text, True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, int(cy - pillar_h - txt.get_height() - 4)))


class RoadBlock(Obstacle):
    """
    Solid blockade occupying a specific lane.
    Requires player to steer (tilt/turn head) to an open lane.
    """
    def __init__(self, z=MAX_Z, lane=0):
        # 3 Lanes: lane 0 = Left (-0.55), lane 1 = Center (0.0), lane 2 = Right (0.55)
        lane_map = {0: -0.55, 1: 0.0, 2: 0.55}
        norm_x = lane_map.get(lane, 0.0)
        super().__init__(z=z, norm_x=norm_x)
        self.lane = lane
        self.type_name = "ROAD_BLOCK"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False

        if abs(self.z - CAR_Z) < 22.0:
            # If car is within the blocked lane
            if abs(car.norm_x - self.norm_x) < 0.36:
                self.collided = True
                return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return

        cx, cy, scale = road_renderer.get_screen_coords(self.norm_x, self.z)
        block_w = max(int(110 * scale), 12)
        block_h = max(int(70 * scale), 8)

        rect = pygame.Rect(int(cx - block_w / 2.0), int(cy - block_h), block_w, block_h)
        pygame.draw.rect(screen, (240, 120, 0), rect, border_radius=3)
        pygame.draw.rect(screen, (255, 255, 255), rect, width=max(1, int(2 * scale)))

        # Hazard warning cross
        pygame.draw.line(screen, (40, 20, 10), rect.topleft, rect.bottomright, max(1, int(3 * scale)))
        pygame.draw.line(screen, (40, 20, 10), rect.topright, rect.bottomleft, max(1, int(3 * scale)))

        if scale > 0.45:
            font = pygame.font.SysFont("Impact", max(10, int(15 * scale)))
            txt = font.render("! STEER !", True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, rect.top - txt.get_height() - 2))


class ObstacleManager:
    """Manages spawning, updating, and collision checking for all obstacles."""
    def __init__(self):
        self.obstacles = []
        self.spawn_timer = 0.0
        self.min_spawn_interval = 2.0  # seconds between obstacles
        self.last_spawn_type = None

    def update(self, speed, dt, car, sound_mgr):
        """
        Updates obstacles, spawns new ones, and checks for collisions.
        Returns: (points_earned, collision_detected)
        """
        points = 0
        collision = False

        # 1. Update and check collisions
        for obs in self.obstacles:
            obs.update(speed, dt)

            # Check collision
            if obs.check_collision(car):
                collision = True
                sound_mgr.play("crash")

            # Check if successfully passed
            if not obs.passed and obs.z < CAR_Z - 15.0 and not obs.collided:
                obs.passed = True
                points += 100
                sound_mgr.play("clear")

        # 2. Prune old obstacles behind camera
        self.obstacles = [obs for obs in self.obstacles if obs.z > -25.0]

        # 3. Dynamic Spawning
        self.spawn_timer += dt
        spawn_threshold = max(1.4, 2.5 - (speed / 1000.0))
        if self.spawn_timer >= spawn_threshold:
            self.spawn_timer = 0.0
            self._spawn_random_obstacle()

        return points, collision

    def _spawn_random_obstacle(self):
        """Selects a fair and varied sequence of obstacles."""
        choices = ["LOW_BEAM", "PILLAR_GATE", "ROAD_BLOCK"]
        # Avoid 3 identical obstacles in a row
        choice = random.choice(choices)

        if choice == "LOW_BEAM":
            self.obstacles.append(LowBeam(z=MAX_Z))
        elif choice == "PILLAR_GATE":
            self.obstacles.append(PillarGate(z=MAX_Z, norm_x=0.0))
        elif choice == "ROAD_BLOCK":
            # Pick a lane (0: Left, 1: Center, 2: Right)
            lane = random.choice([0, 1, 2])
            self.obstacles.append(RoadBlock(z=MAX_Z, lane=lane))

    def render(self, screen, road_renderer):
        """Draws obstacles sorted by Z depth (far to near)."""
        sorted_obstacles = sorted(self.obstacles, key=lambda o: o.z, reverse=True)
        for obs in sorted_obstacles:
            obs.draw(screen, road_renderer)

    def reset(self):
        self.obstacles.clear()
        self.spawn_timer = 0.0
