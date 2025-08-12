import numpy as np

# ===== Joint-safe ranges (degrees) =====
# 2~4번 상한을 살짝 완화(실효 상한이 ~ -5° 되게 조정)
SAFE_DEG_LO = np.array([-20.0, -120.0, -120.0, -180.0], dtype=np.float32)
SAFE_DEG_HI = np.array([ +20.0,   -4.0,   -4.0,   -8.0], dtype=np.float32)

# (env에서 1° 안전마진을 빼므로 실효 상한 ≈ -5°, -5°, -9°)

# Home pose
HOME_DEG = np.array([0.0, -25.0, -35.0, -120.0], dtype=np.float32)

# ===== Control =====
# base가 폭주해서 align을 망가뜨리는 걸 줄이기 위해 base Δ↓, 2~4는 약간 ↑
MAX_DELTA = np.array([0.020, 0.025, 0.025, 0.025], dtype=np.float32)
CONTROL_REPEAT = 4
DRIVE_VELOCITY = 0.6
RANDOM_WARMUP_STEPS = 0

# ===== Curriculum =====
CURRICULUM = [
    (0,   0.08),
    (200, 0.10),
    (400, 0.12),
    (700, 0.16),
    (1000, 0.20),
]

HAND_Z_RANGE = (0.02, 0.08)
MIN_SPAWN_DIST = 0.12
MAX_XY_FROM_HAND = 0.18

# ===== Episode / success =====
PATIENCE = 160
MIN_STEPS_BEFORE_SUCCESS = 12
SUCCESS_DIST = 0.11
SUCCESS_ALIGN = 0.90

SMALL_ACTION_EPS = 0.15  # (env에서 쓰지 않으면 지워도 됩니다)

# ===== Prim paths =====
PLANE_PATH = "/World/respone_area"
HAND_PATH = "/World/fake_hand"
TRACKING_LIGHT_PATH = "/World/another_dof"
ROBOT_LIGHT_PATH = "/World/dofbot/link4/Xform/RectLight"
DOFBOT_PATH = "/World/dofbot"

# ===== Joint names =====
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
