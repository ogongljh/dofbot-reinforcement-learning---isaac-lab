import numpy as np

# ==== 안전각 / 제어 ====
SAFE_DEG_LO = np.array([-20.0, -120.0, -120.0, -180.0], dtype=np.float32)
SAFE_DEG_HI = np.array([ +20.0,   -4.0,   -4.0,   -8.0], dtype=np.float32)
HOME_DEG    = np.array([0.0, -25.0, -35.0, -120.0], dtype=np.float32)

# 제어/탐색
MAX_DELTA = np.array([0.035, 0.050, 0.050, 0.050], dtype=np.float32)
CONTROL_REPEAT = 6
DRIVE_VELOCITY = 0.8

# ✅ 에피소드=400 기준 워밍업
RANDOM_WARMUP_STEPS = 100

# 커리큘럼
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

# 에피소드/성공 판정
PATIENCE = 160
MIN_STEPS_BEFORE_SUCCESS = 12
SUCCESS_DIST = 0.11
SUCCESS_ALIGN = 0.90

# ✅ 실제 Stage 경로 (사용자 환경 기준) - 오타 유지 의도
PLANE_PATH = "/World/respone_area"   # 사용자 환경에서 'respone_area'가 맞음
HAND_PATH = "/World/fake_hand"
TRACKING_LIGHT_PATH = "/World/another_dof"
ROBOT_LIGHT_PATH = "/World/dofbot/link4/Xform/RectLight"
DOFBOT_PATH = "/World/dofbot"

# Joint names
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# ==== 그림자/ROI 파라미터 ====
USE_SHADOW_REWARD = True
HAND_RADIUS = 0.025
ROI_RADIUS  = 0.050  # 샘플링 원판 반경

# 최종 그림자(final) 기준 성공/plateau
SUCCESS_SHADOW = 0.12
MIN_IMPROVE_DELTA = 0.002
NO_IMPROVE_PATIENCE = 90

# (호환용)
K_SHADOW_ABS   = 0.6
K_SHADOW_DELTA = 0.6
K_SHADOW_BEST  = 0.6

# ==== 리워드 스케일 ====
REWARD_REMOVED_K = 0.5   # (이번 스텝에서 추가로 지운 양)
REWARD_FINAL_K   = 0.2   # (남아 있는 최종 그림자)
