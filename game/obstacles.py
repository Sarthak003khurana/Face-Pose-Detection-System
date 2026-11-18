"""
Obstacle classes, collectible coins, combo gauntlets, and collision manager
for Shape-Shift Racer.

Supports:
  - LowBeam: Overhanging barrier (requires FLAT / Smile)
  - PillarGate: Twin vertical pillars with center OR shifted lanes (requires NARROW / O-Mouth + Steer)
  - RoadBlock: Single-lane or Double-lane solid blockades (requires Steering)
  - Morph Gauntlets: Chained 3-obstacle sequences for escalating arcade action
  - Coins: Spinning energy coins placed along optimal passing paths
"""

import math
import random
import pygame
from game.config import (
    MAX_Z, CAR_Z, LOW_BEAM_GAP_HEIGHT, PILLAR_GAP_WIDTH,
    COLOR_NEON_YELLOW, COLOR_NEON_RED, COLOR_NEON_CYAN
)

LANE_X = {0: -0.55, 1: 0.0, 2: 0.55}


class Coin:
    """Spinning golden collectible energy coin placed along optimal passing paths."""
    def __init__(self, z, norm_x=0.0, elevation=14.0):
        self.z = float(z)
        self.norm_x = float(norm_x)
        self.elevation = float(elevation)
        self.collected = False
        self.spin_phase = random.uniform(0.0, math.pi * 2.0)

    def update(self, speed, dt):
        self.z -= speed * dt
        self.spin_phase += 8.0 * dt
        return self.z > -20.0 and not self.collected

    def check_collect(self, car):
        if self.collected:
            return False
        if abs(self.z - CAR_Z) < 28.0 and abs(car.norm_x - self.norm_x) < 0.34:
            self.collected = True
            return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0 or self.collected:
            return
        cx, cy, scale = road_renderer.get_screen_coords(self.norm_x, self.z)
        radius = max(4, int(18 * scale))
        spin_w = max(2, int(abs(math.cos(self.spin_phase)) * radius))
        bob = math.sin(self.spin_phase * 1.3) * 4.0 * scale
        coin_y = int(cy - self.elevation * scale - radius + bob)

        # Soft outer glow
        glow_r = radius + max(2, int(6 * scale))
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 220, 40, 75), (glow_r + 1, glow_r + 1), glow_r)
        screen.blit(glow_surf, (int(cx) - glow_r - 1, coin_y - glow_r - 1))

        # Spinning coin ellipse
        rect = pygame.Rect(int(cx - spin_w), coin_y - radius, spin_w * 2, radius * 2)
        pygame.draw.ellipse(screen, (255, 215, 0), rect)
        pygame.draw.ellipse(screen, (255, 250, 180), rect, width=max(1, int(2 * scale)))
        if spin_w > 4:
            inner = rect.inflate(-max(2, spin_w // 2), -max(3, radius // 2))
            if inner.width > 1 and inner.height > 1:
                pygame.draw.ellipse(screen, (255, 165, 0), inner)


class Obstacle:
    """Base class for pseudo-3D approaching road obstacles."""
    def __init__(self, z=MAX_Z, norm_x=0.0):
        self.z = float(z)
        self.norm_x = float(norm_x)
        self.passed = False
        self.collided = False
        self.near_miss_awarded = False
        self.type_name = "BASE"

    def update(self, speed, dt):
        self.z -= speed * dt
        return self.z > -25.0

    def check_collision(self, car):
        return False

    def check_near_miss(self, car):
        return False

    def draw(self, screen, road_renderer):
        pass


class LowBeam(Obstacle):
    """
    Overhanging horizontal barrier.
    Requires player to be in FLAT shape (Smile) to slide underneath.
    """
    def __init__(self, z=MAX_Z):
        super().__init__(z=z, norm_x=0.0)
        self.type_name = "LOW_BEAM"
        self.prompt_text = "SLIDE FLAT (SMILE)"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False
        if abs(self.z - CAR_Z) < 22.0:
            if car.h > (LOW_BEAM_GAP_HEIGHT + 3.0):
                self.collided = True
                return True
        return False

    def check_near_miss(self, car):
        if self.collided or self.near_miss_awarded:
            return False
        if self.z < CAR_Z - 14.0 and (LOW_BEAM_GAP_HEIGHT - 4.0) <= car.h <= (LOW_BEAM_GAP_HEIGHT + 3.0):
            self.near_miss_awarded = True
            return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return

        lx, ly, scale = road_renderer.get_screen_coords(-0.95, self.z)
        rx, ry, _ = road_renderer.get_screen_coords(0.95, self.z)
        cx, cy, _ = road_renderer.get_screen_coords(0.0, self.z)

        beam_w = rx - lx
        beam_h = max(int(65 * scale), 8)
        gap_h = max(int(LOW_BEAM_GAP_HEIGHT * 1.5 * scale), 4)

        beam_top = int(ly - gap_h - beam_h)
        beam_bottom = int(ly - gap_h)

        # Solid Horizontal Beam Bar
        beam_rect = pygame.Rect(int(lx), beam_top, int(beam_w), beam_h)
        pygame.draw.rect(screen, (220, 20, 60), beam_rect, border_radius=3)
        pygame.draw.rect(screen, (255, 230, 0), beam_rect, width=max(1, int(2 * scale)))

        # Hazard Stripes
        stripe_step = max(int(24 * scale), 6)
        for sx in range(int(lx), int(rx), stripe_step):
            pygame.draw.line(screen, (40, 20, 30),
                             (sx, beam_bottom),
                             (min(sx + stripe_step // 2, int(rx)), beam_top),
                             max(1, int(3 * scale)))

        # Side Support Pillars
        pillar_w = max(int(16 * scale), 3)
        pygame.draw.rect(screen, (100, 100, 120),
                         (int(lx) - pillar_w, beam_top, pillar_w, int(ly - beam_top)))
        pygame.draw.rect(screen, (100, 100, 120),
                         (int(rx), beam_top, pillar_w, int(ry - beam_top)))

        if scale > 0.45:
            font = pygame.font.SysFont("Impact", max(12, int(18 * scale)))
            txt = font.render(self.prompt_text, True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, beam_top + (beam_h - txt.get_height()) // 2))


class PillarGate(Obstacle):
    """
    Twin vertical pillars leaving a narrow vertical slit at `norm_x` (center, left, or right).
    Requires NARROW shape (Open Mouth) + aligning laterally with the slit.
    """
    def __init__(self, z=MAX_Z, norm_x=0.0):
        super().__init__(z=z, norm_x=norm_x)
        self.type_name = "PILLAR_GATE"
        if norm_x < -0.2:
            self.prompt_text = "LEFT SLIT (O-MOUTH)"
        elif norm_x > 0.2:
            self.prompt_text = "RIGHT SLIT (O-MOUTH)"
        else:
            self.prompt_text = "SQUEEZE (O-MOUTH)"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False
        if abs(self.z - CAR_Z) < 22.0:
            if car.w > (PILLAR_GAP_WIDTH + 4.0):
                self.collided = True
                return True
            if abs(car.norm_x - self.norm_x) > 0.30:
                self.collided = True
                return True
        return False

    def check_near_miss(self, car):
        if self.collided or self.near_miss_awarded:
            return False
        if self.z < CAR_Z - 14.0 and 0.20 <= abs(car.norm_x - self.norm_x) <= 0.30:
            self.near_miss_awarded = True
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

        # Left Pillar
        left_pillar_w = max(1, (cx - half_gap) - lx)
        left_rect = pygame.Rect(int(lx), int(cy - pillar_h), int(left_pillar_w), pillar_h)
        pygame.draw.rect(screen, (30, 150, 215), left_rect, border_radius=2)
        pygame.draw.rect(screen, (0, 255, 255), left_rect, width=max(1, int(2 * scale)))

        # Right Pillar
        right_pillar_w = max(1, rx - (cx + half_gap))
        right_rect = pygame.Rect(int(cx + half_gap), int(cy - pillar_h), int(right_pillar_w), pillar_h)
        pygame.draw.rect(screen, (30, 150, 215), right_rect, border_radius=2)
        pygame.draw.rect(screen, (0, 255, 255), right_rect, width=max(1, int(2 * scale)))

        # Top Crossbar
        bar_h = max(int(14 * scale), 3)
        pygame.draw.rect(screen, (20, 75, 140), (int(lx), int(cy - pillar_h), int(rx - lx), bar_h))

        # Highlight gap floor edges with bright neon markers
        pygame.draw.line(screen, (80, 255, 160), (int(cx - half_gap), int(cy)), (int(cx - half_gap), int(cy - pillar_h)), max(1, int(2 * scale)))
        pygame.draw.line(screen, (80, 255, 160), (int(cx + half_gap), int(cy)), (int(cx + half_gap), int(cy - pillar_h)), max(1, int(2 * scale)))

        if scale > 0.45:
            font = pygame.font.SysFont("Impact", max(11, int(16 * scale)))
            txt = font.render(self.prompt_text, True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, int(cy - pillar_h - txt.get_height() - 4)))


class RoadBlock(Obstacle):
    """
    Solid concrete/neon blockade occupying 1 or 2 lanes out of 3.
    Requires steering into an open lane.
    """
    def __init__(self, z=MAX_Z, lanes=None):
        if lanes is None:
            lanes = [random.choice([0, 1, 2])]
        elif isinstance(lanes, int):
            lanes = [lanes]
        self.lanes = list(lanes)
        primary_x = LANE_X.get(self.lanes[0], 0.0)
        super().__init__(z=z, norm_x=primary_x)
        self.type_name = "ROAD_BLOCK"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False
        if abs(self.z - CAR_Z) < 22.0:
            for lane in self.lanes:
                lx = LANE_X[lane]
                if abs(car.norm_x - lx) < 0.34:
                    self.collided = True
                    return True
        return False

    def check_near_miss(self, car):
        if self.collided or self.near_miss_awarded:
            return False
        if self.z < CAR_Z - 14.0:
            min_dist = min(abs(car.norm_x - LANE_X[lane]) for lane in self.lanes)
            if 0.34 <= min_dist <= 0.48:
                self.near_miss_awarded = True
                return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return

        for lane in self.lanes:
            lx = LANE_X[lane]
            cx, cy, scale = road_renderer.get_screen_coords(lx, self.z)
            block_w = max(int(110 * scale), 12)
            block_h = max(int(70 * scale), 8)

            rect = pygame.Rect(int(cx - block_w / 2.0), int(cy - block_h), block_w, block_h)
            pygame.draw.rect(screen, (240, 115, 0), rect, border_radius=3)
            pygame.draw.rect(screen, (255, 255, 255), rect, width=max(1, int(2 * scale)))

            pygame.draw.line(screen, (40, 20, 10), rect.topleft, rect.bottomright, max(1, int(3 * scale)))
            pygame.draw.line(screen, (40, 20, 10), rect.topright, rect.bottomleft, max(1, int(3 * scale)))

            if scale > 0.45:
                font = pygame.font.SysFont("Impact", max(10, int(15 * scale)))
                txt = font.render("! STEER !", True, (255, 255, 255))
                screen.blit(txt, (cx - txt.get_width() // 2, rect.top - txt.get_height() - 2))


class LaserStrip(Obstacle):
    """
    Glowing neon ground laser/spike strip spanning the road.
    Requires jumping (Eyebrow Raise or J/Space) to leap cleanly over it!
    """
    def __init__(self, z=MAX_Z):
        super().__init__(z=z, norm_x=0.0)
        self.type_name = "LASER_STRIP"
        self.prompt_text = "JUMP (RAISE BROWS)"

    def check_collision(self, car):
        if self.collided or self.passed:
            return False
        if abs(self.z - CAR_Z) < 22.0:
            if not car.is_airborne:
                self.collided = True
                return True
        return False

    def check_near_miss(self, car):
        if self.collided or self.near_miss_awarded:
            return False
        if self.z < CAR_Z - 14.0 and 18.0 < car.jump_y <= 38.0:
            self.near_miss_awarded = True
            return True
        return False

    def draw(self, screen, road_renderer):
        if self.z <= 0:
            return
        lx, ly, scale = road_renderer.get_screen_coords(-0.95, self.z)
        rx, ry, _ = road_renderer.get_screen_coords(0.95, self.z)
        cx, cy, _ = road_renderer.get_screen_coords(0.0, self.z)

        strip_h = max(int(18 * scale), 4)
        strip_w = max(4, int(rx - lx))
        top_y = int(ly - strip_h)

        # Glowing plasma laser beam along the asphalt
        glow_surf = pygame.Surface((strip_w + 10, strip_h + 14), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (255, 40, 20, 110), (5, 4, strip_w, strip_h + 6), border_radius=4)
        screen.blit(glow_surf, (int(lx) - 5, top_y - 4))

        pygame.draw.rect(screen, (255, 40, 40), (int(lx), top_y, strip_w, strip_h), border_radius=3)
        core_h = max(2, strip_h // 3)
        pygame.draw.rect(screen, (255, 240, 140), (int(lx) + 2, top_y + (strip_h - core_h) // 2, max(2, strip_w - 4), core_h))

        # Side emitter pylons
        pylon_w = max(int(14 * scale), 3)
        pylon_h = max(int(26 * scale), 6)
        pygame.draw.rect(screen, (90, 95, 115), (int(lx) - pylon_w, int(ly - pylon_h), pylon_w, pylon_h))
        pygame.draw.rect(screen, (90, 95, 115), (int(rx), int(ry - pylon_h), pylon_w, pylon_h))

        if scale > 0.45:
            font = pygame.font.SysFont("Impact", max(11, int(16 * scale)))
            txt = font.render(self.prompt_text, True, (255, 245, 130))
            screen.blit(txt, (cx - txt.get_width() // 2, top_y - txt.get_height() - 4))


class ObstacleManager:
    """Manages spawning obstacles, combo gauntlets, coins, and collision/shield-smash events."""
    def __init__(self):
        self.obstacles = []
        self.coins = []
        self.spawn_timer = 0.0
        self.obstacles_spawned = 0
        self.gauntlet_queue = []  # Queued (delay_sec, obstacle_spec) for rapid gauntlet waves

    def update(self, speed, dt, car, sound_mgr, road_renderer=None):
        """
        Updates obstacles & coins, spawns waves, and checks collisions/collectibles.
        Returns an event summary dictionary.
        """
        events = {
            "obstacles_passed": 0,
            "coins_collected": 0,
            "near_misses": 0,
            "shield_smashes": 0,
            "collision": False,
            "gauntlet_started": False,
            "coin_screen_positions": [],
        }

        # 1. Update Obstacles
        for obs in self.obstacles:
            obs.update(speed, dt)

            if obs.check_collision(car):
                if car.is_shielded:
                    # Shield shatters obstacle for bonus points instead of taking damage!
                    obs.passed = True
                    events["shield_smashes"] += 1
                    sound_mgr.play("smash")
                else:
                    events["collision"] = True
                    sound_mgr.play("crash")

            if not obs.passed and obs.z < CAR_Z - 15.0 and not obs.collided:
                obs.passed = True
                events["obstacles_passed"] += 1
                sound_mgr.play("clear")
                if obs.check_near_miss(car):
                    events["near_misses"] += 1
                    sound_mgr.play("near_miss")

        self.obstacles = [obs for obs in self.obstacles if obs.z > -25.0]

        # 2. Update Coins
        for coin in self.coins:
            coin.update(speed, dt)
            if coin.check_collect(car):
                events["coins_collected"] += 1
                sound_mgr.play("coin")
                if road_renderer is not None:
                    sx, sy, _ = road_renderer.get_screen_coords(coin.norm_x, CAR_Z)
                    events["coin_screen_positions"].append((sx, sy - 35))

        self.coins = [c for c in self.coins if c.z > -20.0 and not c.collected]

        # 3. Spawning (drain active gauntlet queue first, else regular wave)
        self.spawn_timer += dt
        if self.gauntlet_queue:
            next_delay, spec = self.gauntlet_queue[0]
            if self.spawn_timer >= next_delay:
                self.spawn_timer = 0.0
                self.gauntlet_queue.pop(0)
                self._spawn_from_spec(spec)
        else:
            spawn_threshold = max(1.35, 2.45 - (speed / 1050.0))
            if self.spawn_timer >= spawn_threshold:
                self.spawn_timer = 0.0
                if self.obstacles_spawned >= 5 and random.random() < 0.24:
                    self._queue_gauntlet_wave(speed)
                    events["gauntlet_started"] = True
                else:
                    self._spawn_random_obstacle()

        return events

    def _queue_gauntlet_wave(self, speed):
        """Queues a fun 3-obstacle combo Morph Gauntlet."""
        gap_time = max(1.15, 1.65 - (speed / 1600.0))
        patterns = [
            [("LOW_BEAM", 0.0), ("PILLAR_GATE", 0.0), ("LASER_STRIP", 0.0)],
            [("ROAD_BLOCK_DOUBLE", None), ("LOW_BEAM", 0.0), ("PILLAR_GATE", random.choice([-0.42, 0.42]))],
            [("LASER_STRIP", 0.0), ("LOW_BEAM", 0.0), ("ROAD_BLOCK_DOUBLE", None)],
        ]
        chosen = random.choice(patterns)
        self._spawn_from_spec(chosen[0])
        for spec in chosen[1:]:
            self.gauntlet_queue.append((gap_time, spec))

    def _spawn_from_spec(self, spec):
        kind, param = spec
        self.obstacles_spawned += 1

        if kind == "LOW_BEAM":
            self.obstacles.append(LowBeam(z=MAX_Z))
            coin_x = random.choice([-0.35, 0.0, 0.35])
            self.coins.append(Coin(z=MAX_Z - 35.0, norm_x=coin_x, elevation=8.0))

        elif kind == "PILLAR_GATE":
            gate_x = param if param is not None else 0.0
            self.obstacles.append(PillarGate(z=MAX_Z, norm_x=gate_x))
            self.coins.append(Coin(z=MAX_Z - 35.0, norm_x=gate_x, elevation=22.0))

        elif kind == "LASER_STRIP":
            self.obstacles.append(LaserStrip(z=MAX_Z))
            # High floating coin above the laser strip for mid-air grabs!
            self.coins.append(Coin(z=MAX_Z - 20.0, norm_x=0.0, elevation=64.0))

        elif kind == "ROAD_BLOCK_SINGLE":
            blocked = random.choice([0, 1, 2])
            self.obstacles.append(RoadBlock(z=MAX_Z, lanes=[blocked]))
            open_lanes = [l for l in (0, 1, 2) if l != blocked]
            self.coins.append(Coin(z=MAX_Z - 30.0, norm_x=LANE_X[random.choice(open_lanes)], elevation=18.0))

        elif kind == "ROAD_BLOCK_DOUBLE":
            open_lane = random.choice([0, 1, 2])
            blocked = [l for l in (0, 1, 2) if l != open_lane]
            self.obstacles.append(RoadBlock(z=MAX_Z, lanes=blocked))
            self.coins.append(Coin(z=MAX_Z - 30.0, norm_x=LANE_X[open_lane], elevation=18.0))

    def _spawn_random_obstacle(self):
        """Selects a varied obstacle, unlocking shifted gates, laser strips, and double blocks."""
        if self.obstacles_spawned < 4:
            choices = [("LOW_BEAM", 0.0), ("PILLAR_GATE", 0.0), ("ROAD_BLOCK_SINGLE", None), ("LASER_STRIP", 0.0)]
        else:
            choices = [
                ("LOW_BEAM", 0.0),
                ("PILLAR_GATE", 0.0),
                ("PILLAR_GATE", random.choice([-0.42, 0.42])),
                ("LASER_STRIP", 0.0),
                ("ROAD_BLOCK_SINGLE", None),
                ("ROAD_BLOCK_DOUBLE", None),
            ]
        self._spawn_from_spec(random.choice(choices))

    def render(self, screen, road_renderer):
        """Draws all obstacles and coins sorted far-to-near along Z."""
        renderables = [(o.z, o) for o in self.obstacles] + [(c.z, c) for c in self.coins]
        renderables.sort(key=lambda item: item[0], reverse=True)
        for _, obj in renderables:
            obj.draw(screen, road_renderer)

    def reset(self):
        self.obstacles.clear()
        self.coins.clear()
        self.gauntlet_queue.clear()
        self.spawn_timer = 0.0
        self.obstacles_spawned = 0
