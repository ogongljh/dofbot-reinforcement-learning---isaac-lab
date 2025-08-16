import numpy as np

# ===== Joint-safe ranges (degrees) =====
# 각 조인트의 안전한 최소/최대 각도 범위
SAFE_DEG_LO = np.array([-20.0, -120.0, -120.0, -180.0], dtype=np.float32)  # 최소 각도
SAFE_DEG_HI = np.array([ +20.0,   -4.0,   -4.0,   -8.0], dtype=np.float32)  # 최대 각도

# ===== Home pose =====
# 로봇팔 초기 홈 포지션(°)
HOME_DEG = np.array([0.0, -25.0, -35.0, -120.0], dtype=np.float32)

# ===== Control parameters =====
# 조인트별 한 스텝에서 허용되는 최대 각도 변화량(라디안으로 환산되어 사용됨)
# → 1번(base)은 덜, 2~4번은 약간 더 움직이게 설정
MAX_DELTA = np.array([0.020, 0.025, 0.025, 0.025], dtype=np.float32)
CONTROL_REPEAT = 4        # 같은 제어 명령을 반복 적용하는 횟수
DRIVE_VELOCITY = 0.6      # 조인트 속도 비율 (1.0 = 최대 속도)
RANDOM_WARMUP_STEPS = 0   # 에피소드 시작 시 랜덤 워밍업 스텝 수

# ===== Curriculum learning =====
# 학습이 진행됨에 따라 목표 거리 범위를 점점 늘려가는 설정 (step 수, 목표 거리)
CURRICULUM = [
    (0,   0.08),
    (200, 0.10),
    (400, 0.12),
    (700, 0.16),
    (1000, 0.20),
]

# ===== Hand / spawn constraints =====
HAND_Z_RANGE = (0.02, 0.08)   # 손(Z축) 이동 가능 범위 [m]
MIN_SPAWN_DIST = 0.12         # 손과 로봇 라이트 최소 스폰 거리 [m]
MAX_XY_FROM_HAND = 0.18       # 손 기준 XY축 최대 이동 거리 [m]

# ===== Episode / success conditions =====
PATIENCE = 160                 # 개선 없음을 허용하는 최대 스텝 수
MIN_STEPS_BEFORE_SUCCESS = 12  # 성공 판정 전 최소 수행 스텝 수
SUCCESS_DIST = 0.11            # 성공 판정 거리 임계값 [m]
SUCCESS_ALIGN = 0.90           # 성공 판정 정렬(align) 임계값 (0~1)

SMALL_ACTION_EPS = 0.15        # 작은 행동 변화 허용 범위 (미사용 시 제거 가능)

# ===== Prim paths (Isaac Sim USD 경로) =====
PLANE_PATH = "/World/respone_area"                 # 작업 영역 plane
HAND_PATH = "/World/fake_hand"                     # 손끝 오브젝트
TRACKING_LIGHT_PATH = "/World/another_dof"         # 추적용 광원
ROBOT_LIGHT_PATH = "/World/dofbot/link4/Xform/RectLight"  # 로봇팔 부착 광원
DOFBOT_PATH = "/World/dofbot"                      # DOFBOT 로봇 전체 경로

# ===== Joint names =====
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]  # 제어할 DOFBOT 조인트 목록

# ===== Shadow reward params (Sim 전용 보상, 관측엔 포함하지 않음) =====
# 손(가리는 물체)을 반지름 r의 구/원기둥으로 근사하여 점광원 투영으로 그림자 면적을 평가
HAND_RADIUS = 0.035        # 손 유효 반지름 [m] (실측에 맞춰 조정)
ROI_RADIUS  = 0.10         # 손 주변 ROI 반경 [m] (SUCCESS_DIST와 유사/약간 크게)
K_SHADOW    = 1.2          # 그림자 페널티 계수 (0.6~1.8 튜닝 권장)
USE_SHADOW_REWARD = True   # 그림자 보상 on/off (실험 비교에 유용)
