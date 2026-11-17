"""
Game configuration, visual constants, and expression thresholds.
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

# Movement & Speed
BASE_SPEED = 420.0         # Units per second along Z-axis
SPEED_INCREMENT = 12.0     # Incremental speed increase per obstacle passed
MAX_SPEED = 900.0
STEER_SPEED = 6.0          # Responsiveness of lateral movement

# Vision & Expression Thresholds
DEFAULT_SMILE_MULTIPLIER = 1.14      # Smile ratio must exceed baseline * 1.14
DEFAULT_OPEN_MOUTH_MULTIPLIER = 1.28 # Mouth vertical ratio must exceed baseline * 1.28
STEER_SENSITIVITY_YAW = 0.045
STEER_SENSITIVITY_ROLL = 0.055

# Palette (Cyberpunk / Retro Synthwave)
COLOR_SKY_TOP = (15, 5, 30)
COLOR_SKY_BOTTOM = (55, 15, 65)
COLOR_SUN = (255, 215, 0)
COLOR_SUN_GLOW = (255, 90, 40)
COLOR_MOUNTAIN = (35, 12, 50)
COLOR_GRID = (160, 40, 180)
COLOR_ROAD = (25, 20, 35)
COLOR_CURB_LIGHT = (255, 40, 120)
COLOR_CURB_DARK = (40, 20, 70)
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
