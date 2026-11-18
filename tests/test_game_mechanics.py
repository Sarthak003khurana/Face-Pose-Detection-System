"""Unit & integration tests for car morphing, jump/shield physics, obstacles, biomes, audio, and persistence."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from game.config import CAR_Z, CAR_SHAPES
from game.road import RoadRenderer
from game.car import Car
from game.obstacles import (
    ObstacleManager, LowBeam, PillarGate, RoadBlock, LaserStrip, Coin
)
from game.sound_manager import SoundManager
from game.storage import load_highscores, save_highscore, load_settings, save_settings


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


def test_car_morph_jump_and_shield_physics():
    road = RoadRenderer()
    car = Car()

    # Morph to FLAT
    assert car.set_shape("FLAT") is True
    for _ in range(30):
        car.update(road, 0.03)
    assert car.w > 125.0 and car.h < 26.0

    # Jump arc rises airborne and lands back at 0
    assert car.trigger_jump() is True
    car.update(road, 0.15)
    assert car.is_airborne is True and car.jump_y > 40.0
    for _ in range(40):
        car.update(road, 0.03)
    assert car.jump_y == 0.0 and car.is_airborne is False

    # Shield activates and blocks immediate reactivation on cooldown
    assert car.activate_shield(duration=1.0, cooldown=4.0) is True
    assert car.is_shielded is True
    assert car.activate_shield() is False


def test_obstacle_collisions_for_all_types():
    car = Car()

    # 1. LowBeam: NORMAL collides, FLAT clears
    beam = LowBeam(z=CAR_Z)
    car.h = CAR_SHAPES["NORMAL"]["height"]
    assert beam.check_collision(car) is True
    beam2 = LowBeam(z=CAR_Z)
    car.h = CAR_SHAPES["FLAT"]["height"]
    assert beam2.check_collision(car) is False

    # 2. Shifted PillarGate: NARROW at matching lane clears; wrong lane or wide car crashes
    gate = PillarGate(z=CAR_Z, norm_x=-0.42)
    car.w = CAR_SHAPES["NARROW"]["width"]
    car.norm_x = -0.42
    assert gate.check_collision(car) is False

    gate_wrong_lane = PillarGate(z=CAR_Z, norm_x=-0.42)
    car.norm_x = 0.42
    assert gate_wrong_lane.check_collision(car) is True

    # 3. LaserStrip: Grounded collides, Airborne (Jump) clears!
    laser = LaserStrip(z=CAR_Z)
    car.jump_y = 0.0
    assert laser.check_collision(car) is True
    laser2 = LaserStrip(z=CAR_Z)
    car.jump_y = 55.0
    assert laser2.check_collision(car) is False

    # 4. Double-Lane RoadBlock: Blocked lanes collide, open lane clears
    rb = RoadBlock(z=CAR_Z, lanes=[0, 2])
    car.norm_x = 0.0  # Center lane open
    assert rb.check_collision(car) is False
    rb2 = RoadBlock(z=CAR_Z, lanes=[0, 2])
    car.norm_x = -0.55  # Left lane blocked
    assert rb2.check_collision(car) is True


def test_shield_smash_and_coin_collection():
    road = RoadRenderer()
    car = Car()
    sm = SoundManager(music_enabled=False)
    om = ObstacleManager()

    car.activate_shield(duration=3.5)
    om.obstacles.append(RoadBlock(z=CAR_Z + 5.0, lanes=[1]))
    om.coins.append(Coin(z=CAR_Z + 5.0, norm_x=0.0))

    ev = om.update(420.0, 0.016, car, sm, road)
    assert ev["collision"] is False
    assert ev["shield_smashes"] == 1
    assert ev["coins_collected"] == 1


def test_dynamic_biomes_and_curved_road_render():
    screen = pygame.Surface((1280, 720))
    road = RoadRenderer()
    seen = set()
    for d in (0, 500, 1000, 1500):
        name, _ = road.update(450.0, 0.25, distance_meters=d)
        seen.add(name)
        road.render(screen)
    assert seen == {"SUNSET DRIVE", "CYBER NIGHT", "NEON TOKYO", "PLASMA STORM"}


def test_sound_manager_all_sfx_and_music():
    sm = SoundManager(music_enabled=False)
    expected = {"morph", "clear", "coin", "streak", "near_miss", "biome", "crash", "calibrated", "jump", "shield", "smash"}
    assert expected.issubset(set(sm.sounds.keys()))
    assert sm.music_sound is not None


def test_persistent_storage_roundtrip(tmp_path):
    hs_file = str(tmp_path / "hs.json")
    st_file = str(tmp_path / "st.json")

    scores, is_rec = save_highscore(2000, 900, 9, coins=15, filepath=hs_file)
    assert is_rec is True and scores[0]["score"] == 2000
    scores2, is_rec2 = save_highscore(1200, 400, 4, coins=5, filepath=hs_file)
    assert is_rec2 is False and len(scores2) == 2 and scores2[0]["score"] == 2000
    assert load_highscores(hs_file)[0]["coins"] == 15

    save_settings({"smile_sensitivity": 1.4, "open_sensitivity": 1.1, "steer_sensitivity": 0.8, "music_enabled": False}, st_file)
    loaded = load_settings(st_file)
    assert loaded["smile_sensitivity"] == pytest.approx(1.4)
    assert loaded["music_enabled"] is False
