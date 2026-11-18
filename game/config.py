"""
Game configuration, visual constants, dynamic biomes, and expression thresholds.
"""

# Screen & Display Settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
TITLE = "Shape-Shift Racer: Morphing Car (Face Controlled)"

# 2.5D Road Geometry
HORIZON_Y = int(SCREEN_HEIGHT * 0.44)
ROAD_BOTTOM_WIDTH = int(SCREEN_WIDTH * 0.78)
ROAD_TOP_WIDTH = int(SCREEN_WIDTH * 0.12)
ROAD_LEFT_BOTTOM = (SCREEN_WIDTH - ROAD_BOTTOM_WIDTH) // 2
ROAD_RIGHT_BOTTOM = ROAD_LEFT_BOTTOM + ROAD_BOTTOM_WIDTH
ROAD_LEFT_TOP = (SCREEN_WIDTH - ROAD_TOP_WIDTH) // 2
ROAD_RIGHT_TOP = ROAD_LEFT_TOP + ROAD_TOP_WIDTH
NUM_LANES = 3

# Camera & Perspective
CAMERA_DEPTH = 300.0  # Controls perspective scaling
MAX_Z = 1200.0        # Spawn distance of obstacles
CAR_Z = 40.0          # Player's Z distance along the road

# Vehicle Morphing Dimensions (Width, Height in pixels at CAR_Z)
CAR_SHAPES = {
    "NORMAL": {
        "width": 84,
        "height": 50,
        "color_primary": (0, 210, 255),    # Neon Cyan
        "color_accent": (255, 255, 255),
        "name": "CRUISER",
        "description": "Standard Balanced"
    },
    "FLAT": {
        "width": 136,
        "height": 22,
        "color_primary": (255, 0, 128),    # Neon Hot Pink / Magenta
        "color_accent": (255, 240, 100),
        "name": "LOW-RIDER",
        "description": "Ultra-Flat (Under Low Beams)"
    },
    "NARROW": {
        "width": 32,
        "height": 100,
        "color_primary": (0, 255, 140),    # Neon Lime Green
        "color_accent": (100, 255, 255),
        "name": "SLIM-BLADE",
        "description": "Ultra-Narrow (Through Pillar Slits)"
    }
}

# Clearance Dimensions for Obstacle Gaps (at CAR_Z)
LOW_BEAM_GAP_HEIGHT = 30   # Only FLAT (h=22) fits; NORMAL (h=50) & NARROW (h=100) crash
PILLAR_GAP_WIDTH = 48      # Only NARROW (w=32) fits; NORMAL (w=84) & FLAT (w=136) crash

# Movement, Speed & Scoring
BASE_SPEED = 420.0         # Units per second along Z-axis
SPEED_INCREMENT = 10.0     # Incremental speed increase per obstacle passed
MAX_SPEED = 920.0
STEER_SPEED = 6.0          # Responsiveness of lateral movement

OBSTACLE_BASE_POINTS = 100
COIN_BASE_POINTS = 50
NEAR_MISS_BONUS = 50

# Streak Multiplier Tiers: (min_streak, multiplier)
STREAK_TIERS = [
    (10, 5),
    (6, 3),
    (3, 2),
    (0, 1),
]

# Dynamic Biomes (cycles every BIOME_DISTANCE meters)
BIOME_DISTANCE = 500.0
BIOMES = [
    {
        "name": "SUNSET DRIVE",
        "sky_top": (15, 5, 32),
        "sky_bottom": (62, 16, 72),
        "sun": (255, 215, 0),
        "sun_glow": (255, 90, 40),
        "mountain": (36, 12, 52),
        "ground": (12, 8, 22),
        "asphalt_a": (28, 22, 42),
        "asphalt_b": (22, 16, 34),
        "curb_light": (255, 40, 120),
        "curb_dark": (42, 20, 72),
        "accent": (0, 230, 255),
    },
    {
        "name": "CYBER NIGHT",
        "sky_top": (4, 10, 28),
        "sky_bottom": (14, 38, 80),
        "sun": (0, 235, 255),
        "sun_glow": (0, 120, 255),
        "mountain": (10, 22, 48),
        "ground": (5, 10, 20),
        "asphalt_a": (18, 26, 44),
        "asphalt_b": (13, 20, 35),
        "curb_light": (0, 220, 255),
        "curb_dark": (18, 42, 85),
        "accent": (0, 255, 200),
    },
    {
        "name": "NEON TOKYO",
        "sky_top": (8, 22, 28),
        "sky_bottom": (22, 62, 58),
        "sun": (120, 255, 170),
        "sun_glow": (255, 40, 160),
        "mountain": (14, 36, 42),
        "ground": (8, 16, 18),
        "asphalt_a": (22, 34, 38),
        "asphalt_b": (16, 26, 30),
        "curb_light": (50, 255, 140),
        "curb_dark": (70, 18, 68),
        "accent": (255, 60, 180),
    },
    {
        "name": "PLASMA STORM",
        "sky_top": (28, 4, 14),
        "sky_bottom": (85, 18, 28),
        "sun": (255, 120, 20),
        "sun_glow": (255, 30, 50),
        "mountain": (48, 10, 20),
        "ground": (20, 6, 10),
        "asphalt_a": (38, 18, 26),
        "asphalt_b": (28, 12, 18),
        "curb_light": (255, 200, 0),
        "curb_dark": (95, 16, 24),
        "accent": (255, 210, 40),
    },
]

# Vision & Expression Thresholds / Defaults
DEFAULT_SMILE_MULTIPLIER = 1.14
DEFAULT_OPEN_MOUTH_MULTIPLIER = 1.28
STEER_SENSITIVITY_YAW = 0.045
STEER_SENSITIVITY_ROLL = 0.055

DEFAULT_SETTINGS = {
    "smile_sensitivity": 1.0,
    "open_sensitivity": 1.0,
    "steer_sensitivity": 1.0,
    "music_enabled": True,
}

# Persistence Files
HIGHSCORE_FILE = "highscores.json"
SETTINGS_FILE = "settings.json"
MAX_HIGHSCORES = 5

# Base Palette Shortcuts
COLOR_SKY_TOP = BIOMES[0]["sky_top"]
COLOR_SKY_BOTTOM = BIOMES[0]["sky_bottom"]
COLOR_SUN = BIOMES[0]["sun"]
COLOR_SUN_GLOW = BIOMES[0]["sun_glow"]
COLOR_MOUNTAIN = BIOMES[0]["mountain"]
COLOR_GRID = (160, 40, 180)
COLOR_ROAD = (25, 20, 35)
COLOR_CURB_LIGHT = BIOMES[0]["curb_light"]
COLOR_CURB_DARK = BIOMES[0]["curb_dark"]
COLOR_LANE_MARK = (255, 255, 255)
COLOR_HUD_BG = (20, 20, 30, 200)
COLOR_NEON_YELLOW = (255, 230, 0)
COLOR_NEON_RED = (255, 50, 50)
COLOR_NEON_GREEN = (50, 255, 120)
COLOR_NEON_CYAN = (0, 230, 255)

# Webcam Picture-in-Picture (PiP) Window
PIP_WIDTH = 260
PIP_HEIGHT = 195
PIP_X = SCREEN_WIDTH - PIP_WIDTH - 20
PIP_Y = 20
