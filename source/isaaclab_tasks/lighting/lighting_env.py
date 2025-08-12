import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 설정/유틸/씬 구성 모듈
from .lighting_config import (
    SAFE_DEG_LO, SAFE_DEG_HI, HOME_DEG, MAX_DELTA, CONTROL_REPEAT, DRIVE_VELOCITY,
    RANDOM_WARMUP_STEPS, CURRICULUM, HAND_Z_RANGE, MIN_SPAWN_DIST, MAX_XY_FROM_HAND,
    PATIENCE, MIN_STEPS_BEFORE_SUCCESS, SUCCESS_DIST, SUCCESS_ALIGN,
    PLANE_PATH, HAND_PATH, TRACKING_LIGHT_PATH, ROBOT_LIGHT_PATH, DOFBOT_PATH, JOINT_NAMES,
    HAND_RADIUS, ROI_RADIUS, K_SHADOW, USE_SHADOW_REWARD
)
from .lighting_utils import deg2rad, polar_sample, aim_pitch_deg
from .lighting_scene import LightingScene


class LightingEnv(gym.Env):
    """DOFBOT + 조명 추적 강화학습 환경 (Gym API 준수)"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, world_path: str, sim_app, episode_steps=400, use_curriculum=True):
        super().__init__()
        # ─ 기본 인자 보관
        self._sim_app = sim_app
        self._world_path = world_path
        self.episode_steps = int(episode_steps)
        self.use_curriculum = bool(use_curriculum)

        # ── Isaac Sim 의존 모듈은 SimulationApp 이후 import (지연 임포트)
        try:
            from isaacsim.core import World
        except ImportError:
            from omni.isaac.core import World  # 구버전 fallback
        try:
            from isaacsim import dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        except ImportError:
            from omni.isaac.dynamic_control import _dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        from pxr import Usd, PhysxSchema  # Stage 검색용

        # ── 액션/관측 공간 정의
        # action: 4개 조인트의 연속 제어 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # observation: [조인트4 + 손xyz + 로봇라이트xyz + 로봇→손 벡터xyz + 거리]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 + 3 + 3 + 3 + 1,), dtype=np.float32
        )

        # ✅ Stage 열기 → World 초기화 → 시뮬레이션 진행 순서 맞춤
        self.scene = LightingScene(
            sim_app=self._sim_app,
            world_path=self._world_path,
            plane_path=PLANE_PATH,
            hand_path=HAND_PATH,
            tracking_light_path=TRACKING_LIGHT_PATH,
            robot_light_path=ROBOT_LIGHT_PATH
        )
        self.world = World()
        self.world.reset();  self._sim_app.update()
        self.world.play();   self._sim_app.update()
        for _ in range(2):
            self.world.step(False); self._sim_app.update()

        # Dynamic Control 핸들
        self.dc = _acquire_dc()

        # ── DOFBOT articulation prim 경로 자동 탐색 (fallback 포함)
        self.dof_path = self._find_dofbot_path(stage=self.scene.robot_light_prim.GetStage(),
                                               default=DOFBOT_PATH, Usd=Usd, PhysxSchema=PhysxSchema)
        self.art = self.dc.get_articulation(self.dof_path)
        if self.art is None:
            raise RuntimeError(f"❌ Articulation not found at {self.dof_path}. 경로/시뮬 상태 확인.")

        # ── 조인트 핸들 확보
        self.joint_names = JOINT_NAMES
        self.joints = [self.dc.find_articulation_dof(self.art, n) for n in self.joint_names]
        if any(h == 0 for h in self.joints):
            raise RuntimeError("❌ One or more DOF handles invalid. Check joint names / USD.")

        # ── 시뮬레이터가 가진 조인트 제한(lower/upper) 읽기
        self.joint_limits = []
        for h in self.joints:
            props = self.dc.get_dof_properties(h)
            if getattr(props, "hasLimits", False):
                lo = float(np.array(getattr(props, "lower")).reshape(-1)[0])
                hi = float(np.array(getattr(props, "upper")).reshape(-1)[0])
            else:
                lo, hi = -math.pi, math.pi
            self.joint_limits.append((lo, hi))
        self.joint_limits = np.array(self.joint_limits, dtype=np.float32)

        # ── config 안전각 + 시뮬레이터 제한을 동시에 만족하도록 안전 범위 계산
        self.sim_safe_lo = np.maximum(deg2rad(SAFE_DEG_LO), self.joint_limits[:, 0])
        self.sim_safe_hi = np.minimum(deg2rad(SAFE_DEG_HI), self.joint_limits[:, 1])

        # 추가 하드가드: joint2~4는 충분히 숙일 수 있도록 하한을 -120°까지 보장
        for i in (1, 2, 3):
            self.sim_safe_lo[i] = max(self.sim_safe_lo[i], np.radians(-120.0))
            # 상한은 lighting_config에서 이미 -5/-10° 부근으로 제한됨

        # 홈 포즈(라디안)
        self.home_pose = deg2rad(HOME_DEG)

        # ── 제어 파라미터
        self.max_delta = np.asarray(MAX_DELTA, dtype=np.float32)   # 스텝당 각도 변화량 스케일
        self.control_repeat = int(CONTROL_REPEAT)                  # 명령 반복 스텝수
        self.drive_velocity = float(DRIVE_VELOCITY)                # 조인트 속도 타깃
        self.random_warmup_steps = int(RANDOM_WARMUP_STEPS)        # 초기 랜덤 워밍업

        # 액션 부호 → 실제 조인트 증감 방향 매핑 (시뮬/실기 정합을 위한 부호 보정)
        self.joint_sign = np.array([+1, -1, -1, -1], dtype=np.float32)

        # 로봇 라이트의 "앞방향" 부호 (reset 때 도트프로덕트로 확정)
        self._fwd_sign = +1.0

        # ── 에피소드 상태 변수들
        self.episode_idx = 0
        self._step_count = 0
        self._no_improve_count = 0
        self.best_reward = -1e9
        self.prev_q = None  # 직전 조인트 각

        # 커리큘럼 & 스폰 파라미터
        self.spawn_radius = float(CURRICULUM[0][1])
        self.hand_z_min, self.hand_z_max = HAND_Z_RANGE
        self.min_spawn_dist = float(MIN_SPAWN_DIST)
        self.max_xy_from_hand = float(MAX_XY_FROM_HAND)

        # 성공/조기종료 조건 파라미터
        self.patience = int(PATIENCE)
        self.min_steps_before_success = int(MIN_STEPS_BEFORE_SUCCESS)
        self.success_dist = float(SUCCESS_DIST)
        self.success_align = float(SUCCESS_ALIGN)

        # ── 안전 마진(경계 근접 반전/클리핑용)
        self._EPS_BOUND = np.radians(1.0)  # 경계 바깥쪽 마진
        self._EPS_NEAR  = np.radians(1.0)  # 경계 근접 판단 마진

        # 보상/종료 보조 상태
        self._prev_align = 0.0
        self._bad_align_count = 0

        # shaping(Δ거리/Δ정렬) 상태
        self._prev_dist = None
        self._prev_align_for_r = None

    # ---------- 내부 유틸 ----------
    def _find_dofbot_path(self, stage, default, Usd, PhysxSchema):
        """Stage에서 DOFBOT articulation prim 경로를 탐색."""
        # 기본 경로 우선
        prim = stage.GetPrimAtPath(default)
        if prim and prim.IsValid():
            return default
        # 흔한 대체 경로들
        for c in ["/World/dofbot", "/World/DOFBOT", "/World/dofbot/dofbot"]:
            p = stage.GetPrimAtPath(c)
            if p and p.IsValid():
                return c
        # 최후: articulation root API를 가진 prim 순회탐색
        for p in stage.Traverse():
            try:
                if p.HasAPI(PhysxSchema.PhysxArticulationRootAPI):
                    return p.GetPath().pathString
            except Exception:
                continue
        raise RuntimeError("❌ DOFBOT articulation prim을 Stage에서 찾지 못했습니다. lighting_config.DOFBOT_PATH 확인.")

    def _apply_curriculum(self):
        """에피소드 번호에 따라 spawn 반경을 점진적으로 확대."""
        if not self.use_curriculum:
            return
        ep = self.episode_idx
        for th, r in CURRICULUM:
            if ep >= th:
                self.spawn_radius = float(r)

    def _move_to_home(self, max_steps=400, tol=1e-3):
        """로봇을 홈 포즈로 이동 (약간의 지터 추가로 지역최소 회피)."""
        jitter = (np.random.uniform(-1.0, 1.0, size=4).astype(np.float32)) * np.radians(4.0)
        q_home = (self.home_pose + jitter).astype(np.float32)
        # 타깃/속도 지정
        for h, q in zip(self.joints, q_home):
            self.dc.set_dof_position_target(h, float(q))
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        # 도달까지 스텝
        for _ in range(int(max_steps)):
            self.world.step(False); self._sim_app.update()
            cur = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
            if float(np.max(np.abs(cur - q_home))) < tol:
                break
        # 속도 0으로 정지
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

    # ── 그림자 보상 유틸: 원–원 교집합 면적
    def _circle_intersection_area(self, c1, r1, c2, r2):
        (x1, y1), (x2, y2) = c1, c2
        d = math.hypot(x2 - x1, y2 - y1)
        if d >= r1 + r2:
            return 0.0  # 분리
        if d <= abs(r1 - r2):
            return math.pi * (min(r1, r2) ** 2)  # 완전 포함
        r1_2, r2_2 = r1*r1, r2*r2
        alpha = math.acos((d*d + r1_2 - r2_2) / (2*d*r1))
        beta  = math.acos((d*d + r2_2 - r1_2) / (2*d*r2))
        return r1_2*alpha + r2_2*beta - 0.5*math.sqrt(
            max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        )

    def _shadow_ratio_point_light(self):
        """점광원 근사로 plane 위 ROI와 그림자 원의 겹침 비율(0~1)을 반환."""
        # 좌표
        Lx, Ly, Lz = self.scene.get_robot_light_world_pos()
        fx, fy, fz = self.scene.get_hand_pos()
        plane_z = self.scene.get_plane_center()[2]

        # 투영 스케일
        s_den = (fz - Lz)
        if abs(s_den) < 1e-6:
            return 0.0
        s = (plane_z - Lz) / s_den
        if s <= 0.0:
            return 0.0  # 작업면 반대방향이면 투영 없음

        # 그림자 원(중심, 반지름)
        Cx = Lx + s*(fx - Lx)
        Cy = Ly + s*(fy - Ly)
        R  = max(0.0, s * float(HAND_RADIUS))

        # ROI: 손 중심 주변 원
        A_int = self._circle_intersection_area((Cx, Cy), R, (fx, fy), float(ROI_RADIUS))
        A_roi = math.pi * (float(ROI_RADIUS) ** 2)
        return float(A_int / A_roi) if A_roi > 0 else 0.0

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        """에피소드 초기화: 랜덤 초기자세, 목표(hand) 스폰, 추적광 정렬."""
        super().reset(seed=None)

        # 상태 초기화
        self._step_count = 0
        self._no_improve_count = 0
        self.best_reward = -1e9
        self._apply_curriculum()
        self._prev_dist = None
        self._prev_align_for_r = None

        # 안전 범위 내 랜덤 초기자세(경계 마진 적용)
        margin = np.radians(3.0)
        q_lo = self.sim_safe_lo + margin
        q_hi = self.sim_safe_hi - margin
        # joint2~4는 상한을 -5° 이하로 더 낮춰 초기자세를 전방 쪽으로 치우치게 샘플
        for i in (1, 2, 3):
            q_hi[i] = min(q_hi[i], np.radians(-5.0))
        q0 = np.random.uniform(q_lo, q_hi).astype(np.float32)
        # 초기 타깃 지정/속도 부여
        for h, q in zip(self.joints, q0):
            self.dc.set_dof_position_target(h, float(q))
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        # 안정화
        for _ in range(30):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # 추적광 초기 위치/자세 준비
        self.scene.cache_tracking_base()
        lx0, ly0, lz0 = self.scene.tracking_base

        # 작업 plane 중심/로봇라이트 현재 위치
        cx, cy, cz = self.scene.get_plane_center()
        rl = self.scene.get_robot_light_world_pos()

        # 목표(fake_hand) 스폰: 일정 반경 내에서 손/라이트와 최소거리 조건을 만족
        fake = None
        for _ in range(40):
            rx, ry = polar_sample(cx, cy, 0.3*self.spawn_radius, self.spawn_radius, self.np_random)
            rz = float(self.np_random.uniform(self.hand_z_min, self.hand_z_max)) + cz
            if math.dist((rx, ry, rz), rl) >= self.min_spawn_dist:
                fake = (rx, ry, rz); break
        if fake is None:
            fake = (rx, ry, rz)
        self.scene.set_hand_pos(*fake)

        # 추적광의 xy 이동 제한(MAX_XY_FROM_HAND) 안에서 목표를 바라보도록 조정
        dx, dy = fake[0] - lx0, fake[1] - ly0
        dxy = math.hypot(dx, dy)
        if dxy > self.max_xy_from_hand:
            nx, ny = dx/dxy, dy/dxy
            nlx, nly = fake[0] - nx*self.max_xy_from_hand, fake[1] - ny*self.max_xy_from_hand
        else:
            nlx, nly = lx0, ly0
        pitch = aim_pitch_deg((nlx, nly, lz0), fake)
        self.scene.set_tracking_light_pose(nlx, nly, lz0, pitch)

        # 몇 스텝 돌려서 상태 반영
        for _ in range(3):
            self.world.step(False); self._sim_app.update()

        # 로봇 라이트 "앞방향" 부호 결정 (라이트 forward와 목표 방향의 내적 부호)
        self.prev_q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        fwd = self.scene.get_robot_light_forward()
        toT, _ = self.scene.get_vec_robot_to_hand()
        dot = float(fwd[0]*toT[0] + fwd[1]*toT[1] + fwd[2]*toT[2])
        self._fwd_sign = +1.0 if dot >= 0.0 else -1.0
        print(f"[CALIB] forward_sign={self._fwd_sign:+.0f} (dot={dot:.3f})")

        # 보조 상태 초기화
        align0, _ = self._forward_alignment()
        self._prev_align = align0
        self._bad_align_count = 0

        self.episode_idx += 1
        print(f"[RESET] ep={self.episode_idx} q0(deg)={np.degrees(self.prev_q)}")

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """액션 적용 → 시뮬 진행 → 관측/보상 계산 → 종료/트렁케이션 판정."""
        # 워밍업 단계면 랜덤액션
        if self._step_count < self.random_warmup_steps:
            action = self.action_space.sample().astype(np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)

        # 액션 스케일링 + 부호 매핑
        d_sim = (np.clip(action, -1.0, 1.0) * self.max_delta) * self.joint_sign

        hit_limit = 0           # 경계 히트 카운터
        tgt_cmd = np.zeros(4, dtype=np.float32)  # 디버깅 로그용 목표값

        # ── 경계 안전마진, 최소 가동폭 보장, 경계 근접 시 반전 가드 ──
        MIN_SPAN = np.radians(5.0)
        for i, h in enumerate(self.joints):
            cur = self.dc.get_dof_position(h)
            lo_hw, hi_hw = self.joint_limits[i]   # 시뮬 하드 제한
            lo_raw = float(self.sim_safe_lo[i])   # 우리가 설정한 안전 제한
            hi_raw = float(self.sim_safe_hi[i])

            # 안전 마진 적용
            lo = max(lo_hw, lo_raw + self._EPS_BOUND)
            hi = min(hi_hw, hi_raw - self._EPS_BOUND)

            # 유효 가동폭 확보 (너무 좁으면 중앙 기준으로 5° 확보)
            if hi - lo < MIN_SPAN:
                mid = 0.5 * (lo + hi)
                lo = mid - 0.5 * MIN_SPAN
                hi = mid + 0.5 * MIN_SPAN

            # 경계 근접 + 경계 쪽으로 나가려는 액션이면 방향 반전
            near_hi = (cur >= hi - self._EPS_NEAR) and (d_sim[i] > 0)
            near_lo = (cur <= lo + self._EPS_NEAR) and (d_sim[i] < 0)
            if near_hi or near_lo:
                d_sim[i] = -abs(d_sim[i]) if near_hi else abs(d_sim[i])

            # 목표값 = 현재 + Δ, 이후 안전구간으로 클리핑
            tgt = float(cur + float(d_sim[i]))
            tgt = float(np.clip(tgt, lo, hi))

            # 경계에 딱 붙었는지 체크(히트 카운트)
            if tgt <= lo + 1e-6 or tgt >= hi - 1e-6:
                hit_limit += 1

            # 타깃 반영
            self.dc.set_dof_position_target(h, tgt)
            tgt_cmd[i] = tgt

        # 속도 타깃 지정 후 control_repeat 만큼 시뮬스텝 수행
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(self.control_repeat):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # ── 추적광이 항상 목표(hand)를 바라보도록 유지 (xy 이동 제한 준수)
        fx, fy, fz = self.scene.get_hand_pos()
        lx0, ly0, lz0 = self.scene.tracking_base
        dx, dy = fx - lx0, fy - ly0
        dxy = math.hypot(dx, dy)
        if dxy > self.max_xy_from_hand:
            nx, ny = dx/dxy, dy/dxy
            nlx, nly = fx - nx*self.max_xy_from_hand, fy - ny*self.max_xy_from_hand
        else:
            nlx, nly = lx0, ly0
        pitch = aim_pitch_deg((nlx, nly, lz0), (fx, fy, fz))
        self.scene.set_tracking_light_pose(nlx, nly, lz0, pitch)

        # 관측/보상 계산
        obs = self._get_obs()
        reward = self._compute_reward()

        # ===== 그림자 보상(시뮬 전용, 정책 관측엔 미포함) =====
        if USE_SHADOW_REWARD:
            shadow_ratio = self._shadow_ratio_point_light()   # 0~1
            r_shadow = -float(K_SHADOW) * shadow_ratio
            reward += r_shadow
        else:
            shadow_ratio = 0.0
            r_shadow = 0.0

        # 성공/종료 판단용 정렬/거리
        align, dist = self._forward_alignment()
        dalign = align - self._prev_align
        self._prev_align = align

        # 나쁜 정렬(뒤를 봄) 누적
        if align < 0.0:
            self._bad_align_count += 1
        else:
            self._bad_align_count = 0

        self._step_count += 1
        success_ready = (self._step_count >= self.min_steps_before_success)
        success = success_ready and (dist <= self.success_dist) and (align >= self.success_align)

        # 베스트 리워드 갱신 기반 인내심 관리
        if reward > self.best_reward + 1e-3:
            self.best_reward = reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        # 스톨(정지) 감지: 직전 조인트와 거의 동일하면 스톨로 판단
        q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        dq = float(np.max(np.abs(q - self.prev_q))) if self.prev_q is not None else 0.0
        self.prev_q = q.copy()
        stalled = dq < 1e-4

        # ✅ 성공 시 보너스(+ 시간 보너스 일부)
        if success:
            success_bonus = 10.0
            time_bonus = 1.0 - (self._step_count / self.episode_steps)
            reward += success_bonus + 0.5 * time_bonus

        terminated = bool(success)

        # ✅ 조기 종료 조건 (히트/정렬/개선 없음/스톨)
        truncated = (
            (self._step_count >= self.episode_steps) or
            ((hit_limit >= 4) and (self._step_count > 60)) or   # 경계 히트 과다
            (self._no_improve_count >= self.patience) or        # 개선 없음
            (stalled and self._no_improve_count >= 200)         # 스톨 지속
        )
        # 나쁜 정렬 허용폭 약간 완화
        bad_align_patience = 25
        truncated = truncated or (self._bad_align_count >= bad_align_patience)

        # ── 디버깅 로그
        cur_now = [self.dc.get_dof_position(h) for h in self.joints]
        print(
            "[STEP] ep={} t={} cur(deg)={} tgt(deg)={} hit={} dist={:.3f} align={:.3f} rew={:.3f} r_shadow={:.3f} sr={:.3f}".format(
                self.episode_idx,
                self._step_count - 1,
                np.degrees(cur_now),
                np.degrees(tgt_cmd),
                hit_limit,
                dist, align, float(reward),
                float(r_shadow), float(shadow_ratio)
            )
        )

        # 정보 dict (로깅/디버깅용)
        edge_cnt = int(np.sum((q <= (self.sim_safe_lo + self._EPS_BOUND)) |
                              (q >= (self.sim_safe_hi - self._EPS_BOUND))))
        info = {
            "dist": dist,
            "align": align,
            "hit_limit": hit_limit,
            "saturation_rate": float(edge_cnt / 4.0),
            "step": self._step_count,
            "reward": float(reward),
            "dalign": float(dalign),
            "shadow_ratio": float(shadow_ratio),
            "r_shadow": float(r_shadow),
        }
        return obs, float(reward), terminated, truncated, info

    # ---------- 관측/보상 ----------
    def _get_obs(self):
        """현재 조인트/목표/로봇라이트/상대벡터/거리로 관측 벡터 생성."""
        jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        hx, hy, hz = self.scene.get_hand_pos()
        rx, ry, rz = self.scene.get_robot_light_world_pos()
        vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
        dist = float(math.sqrt(vx*vx + vy*vy + vz*vz))
        return np.array([*jp, hx, hy, hz, rx, ry, rz, vx, vy, vz, dist], dtype=np.float32)

    def _forward_alignment(self):
        """로봇 라이트의 로컬 -Z(앞)과 로봇→목표 단위벡터의 내적([-1,1])과 거리 반환."""
        fwd = self.scene.get_robot_light_forward()
        fwd = (self._fwd_sign * fwd[0], self._fwd_sign * fwd[1], self._fwd_sign * fwd[2])
        toT, d = self.scene.get_vec_robot_to_hand()
        align = float(fwd[0]*toT[0] + fwd[1]*toT[1] + fwd[2]*toT[2])
        return align, d

    def _posture_prior(self, q, align_raw):
        """정면/후면에 따라 joint2~4 부호 선호도를 약하게 부여(엉뚱한 자세 억제)."""
        ahead = (align_raw >= 0.0)
        idxs = [1, 2, 3]
        pen = 0.0
        for i in idxs:
            qi = float(q[i])
            if ahead:
                if qi > 0:  # 정면 볼 때 양의 각은 페널티
                    pen += np.tanh(abs(qi))
            else:
                if qi < 0:  # 후면 볼 때 음의 각은 페널티
                    pen += np.tanh(abs(qi))
        return pen

    def _compute_reward(self):
        """거리/정렬 중심의 shaping 보상 + 경계/스무딩/자세 prior 페널티."""
        align, d = self._forward_alignment()
        jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)

        # ── Potential-based shaping: Δ거리/Δ정렬
        if self._prev_dist is None:
            self._prev_dist = d
        if self._prev_align_for_r is None:
            self._prev_align_for_r = align
        d_dist   = self._prev_dist - d            # 가까워지면 +
        d_align  = align - self._prev_align_for_r # 정면으로 돌면 +
        self._prev_dist = d
        self._prev_align_for_r = align

        # ── 기본 항목
        r_align = align          # 현재 정렬
        r_dalign = d_align       # 정렬 변화량
        r_close = -max(0.0, 0.03 - d) * 6.0  # 너무 가까우면 페널티

        # ✅ 경계 페널티 완화: 경계 근접 조인트 개수만큼 벌점
        near_low  = jp <= (self.sim_safe_lo + self._EPS_BOUND)
        near_high = jp >= (self.sim_safe_hi - self._EPS_BOUND)
        edge_cnt = int(np.sum(near_low | near_high))
        r_edge = -1.0 * edge_cnt   # (기존 -2.5 → -1.0)

        # 소프트 배리어: 안전 범위 중앙에서 벗어나면 페널티 완화
        qmid = 0.5 * (self.sim_safe_lo + self.sim_safe_hi)
        qspan = (self.sim_safe_hi - self.sim_safe_lo) + 1e-6
        closeness = np.maximum(0.0, 0.8 - np.abs((jp - qmid) / (0.5 * qspan)))
        r_soft_barrier = -1.0 * float(np.sum(closeness))

        # ✅ 움직임 스무딩 완화: 급격한 변화 억제
        if self.prev_q is not None:
            dq_vec = jp - self.prev_q
            r_smooth = -0.5 * float(np.sum(dq_vec * dq_vec))  # (기존 -2.0)
        else:
            r_smooth = 0.0

        # 자세 prior(약하게): joint2~4 부호 선호
        post_pen = self._posture_prior(jp, align)

        # 가중치 밸런스
        w_d_dist  = 2.0
        w_align   = 1.0
        w_d_align = 0.5
        w_post    = 0.03
        back_soft = -0.3 * max(0.0, -align)  # 뒤를 보면 약한 페널티

        reward = (
            w_d_dist * d_dist
            + w_align * r_align
            + w_d_align * r_dalign
            + r_close
            + r_edge
            + r_soft_barrier
            + r_smooth
            + back_soft
            - w_post * post_pen
        )

        # 베이스(1번 조인트) 과회전 억제(아주 약하게)
        base_pen = 0.01 * abs(jp[0])
        reward -= base_pen

        # 스케일 안정화를 위한 클리핑
        reward = np.clip(reward, -5.0, 5.0)
        return float(reward)

    def render(self):  # 시각화는 외부 GUI가 담당, 여기서는 미사용
        pass
