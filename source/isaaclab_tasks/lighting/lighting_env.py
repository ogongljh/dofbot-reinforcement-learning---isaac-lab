import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .lighting_config import (
    SAFE_DEG_LO, SAFE_DEG_HI, HOME_DEG, MAX_DELTA, CONTROL_REPEAT, DRIVE_VELOCITY,
    RANDOM_WARMUP_STEPS, CURRICULUM, HAND_Z_RANGE, MIN_SPAWN_DIST, MAX_XY_FROM_HAND,
    PATIENCE, MIN_STEPS_BEFORE_SUCCESS, SUCCESS_DIST, SUCCESS_ALIGN,
    PLANE_PATH, HAND_PATH, TRACKING_LIGHT_PATH, ROBOT_LIGHT_PATH, DOFBOT_PATH, JOINT_NAMES
)
from .lighting_utils import deg2rad, polar_sample, aim_pitch_deg
from .lighting_scene import LightingScene


class LightingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, world_path: str, sim_app, episode_steps=400, use_curriculum=True):
        super().__init__()
        self._sim_app = sim_app
        self._world_path = world_path
        self.episode_steps = int(episode_steps)
        self.use_curriculum = bool(use_curriculum)

        # ── 지연 임포트 (SimulationApp 이후) ─────────────────────────────
        try:
            from isaacsim.core import World
        except ImportError:
            from omni.isaac.core import World  # fallback
        try:
            from isaacsim import dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        except ImportError:
            from omni.isaac.dynamic_control import _dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        from pxr import Usd, PhysxSchema  # dofbot 자동탐색용

        # ---- action/obs ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 + 3 + 3 + 3 + 1,), dtype=np.float32
        )

        # ✅ 1) Stage (open) → 2) World → 3) DC 순서
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

        self.dc = _acquire_dc()

        # ---- dofbot path 자동 탐색 (fallback 포함)
        self.dof_path = self._find_dofbot_path(stage=self.scene.robot_light_prim.GetStage(),
                                               default=DOFBOT_PATH, Usd=Usd, PhysxSchema=PhysxSchema)
        self.art = self.dc.get_articulation(self.dof_path)
        if self.art is None:
            raise RuntimeError(f"❌ Articulation not found at {self.dof_path}. 경로/시뮬 상태 확인.")

        self.joint_names = JOINT_NAMES
        self.joints = [self.dc.find_articulation_dof(self.art, n) for n in self.joint_names]
        if any(h == 0 for h in self.joints):
            raise RuntimeError("❌ One or more DOF handles invalid. Check joint names / USD.")

        # ---- joint limits & safe ranges ----
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

        self.sim_safe_lo = np.maximum(deg2rad(SAFE_DEG_LO), self.joint_limits[:, 0])
        self.sim_safe_hi = np.minimum(deg2rad(SAFE_DEG_HI), self.joint_limits[:, 1])

        # 🔒 하드가드 (전방만, 충분히 숙일 수 있게 넓게): joint2~4 = [-120°, 상한은 config 반영]
        for i in (1, 2, 3):
            self.sim_safe_lo[i] = max(self.sim_safe_lo[i], np.radians(-120.0))
            # 상한은 lighting_config에서 이미 -5/-10deg로 제한됨

        self.home_pose = deg2rad(HOME_DEG)

        # ---- control params ----
        self.max_delta = np.asarray(MAX_DELTA, dtype=np.float32)
        self.control_repeat = int(CONTROL_REPEAT)
        self.drive_velocity = float(DRIVE_VELOCITY)
        self.random_warmup_steps = int(RANDOM_WARMUP_STEPS)

        # 액션→조인트 부호 매핑
        self.joint_sign = np.array([+1, -1, -1, -1], dtype=np.float32)

        # 로봇 라이트 앞방향 부호 (리셋 때 dot 기준으로 즉시 확정)
        self._fwd_sign = +1.0

        # ---- episode state ----
        self.episode_idx = 0
        self._step_count = 0
        self._no_improve_count = 0
        self.best_reward = -1e9
        self.prev_q = None

        # curriculum & spawn
        self.spawn_radius = float(CURRICULUM[0][1])
        self.hand_z_min, self.hand_z_max = HAND_Z_RANGE
        self.min_spawn_dist = float(MIN_SPAWN_DIST)
        self.max_xy_from_hand = float(MAX_XY_FROM_HAND)

        self.patience = int(PATIENCE)
        self.min_steps_before_success = int(MIN_STEPS_BEFORE_SUCCESS)
        self.success_dist = float(SUCCESS_DIST)
        self.success_align = float(SUCCESS_ALIGN)

        # ---- 경계 여유폭(안전 마진) ── (증가)
        self._EPS_BOUND = np.radians(1.0)  # 1.0°
        self._EPS_NEAR  = np.radians(1.0)  # 1.0°

        # 🔧 보상/종료 보조 상태
        self._prev_align = 0.0
        self._bad_align_count = 0

        # 🔧 보상 shaping 상태 (Δ거리/Δ정렬 계산용)
        self._prev_dist = None
        self._prev_align_for_r = None

    # ---------- 내부 유틸 ----------
    def _find_dofbot_path(self, stage, default, Usd, PhysxSchema):
        prim = stage.GetPrimAtPath(default)
        if prim and prim.IsValid():
            return default
        for c in ["/World/dofbot", "/World/DOFBOT", "/World/dofbot/dofbot"]:
            p = stage.GetPrimAtPath(c)
            if p and p.IsValid():
                return c
        for p in stage.Traverse():
            try:
                if p.HasAPI(PhysxSchema.PhysxArticulationRootAPI):
                    return p.GetPath().pathString
            except Exception:
                continue
        raise RuntimeError("❌ DOFBOT articulation prim을 Stage에서 찾지 못했습니다. lighting_config.DOFBOT_PATH 확인.")

    def _apply_curriculum(self):
        if not self.use_curriculum:
            return
        ep = self.episode_idx
        for th, r in CURRICULUM:
            if ep >= th:
                self.spawn_radius = float(r)

    def _move_to_home(self, max_steps=400, tol=1e-3):
        jitter = (np.random.uniform(-1.0, 1.0, size=4).astype(np.float32)) * np.radians(4.0)
        q_home = (self.home_pose + jitter).astype(np.float32)
        for h, q in zip(self.joints, q_home):
            self.dc.set_dof_position_target(h, float(q))
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(int(max_steps)):
            self.world.step(False); self._sim_app.update()
            cur = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
            if float(np.max(np.abs(cur - q_home))) < tol:
                break
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=None)

        self._step_count = 0
        self._no_improve_count = 0
        self.best_reward = -1e9
        self._apply_curriculum()

        # shaping 상태 초기화
        self._prev_dist = None
        self._prev_align_for_r = None

        # 안전 범위 내 랜덤 초기자세(경계 마진 적용)
        margin = np.radians(3.0)
        q_lo = self.sim_safe_lo + margin
        q_hi = self.sim_safe_hi - margin

        # 🔧 초기자세 전방(음수) 샘플 강화: joint2~4 상한 더 낮춰 샘플링
        for i in (1, 2, 3):
            q_hi[i] = min(q_hi[i], np.radians(-5.0))

        q0 = np.random.uniform(q_lo, q_hi).astype(np.float32)

        for h, q in zip(self.joints, q0):
            self.dc.set_dof_position_target(h, float(q))
            self.dc.set_dof_velocity_target(h, self.drive_velocity)

        for _ in range(30):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # tracking light 초기화
        self.scene.cache_tracking_base()
        lx0, ly0, lz0 = self.scene.tracking_base

        cx, cy, cz = self.scene.get_plane_center()
        rl = self.scene.get_robot_light_world_pos()

        fake = None
        for _ in range(40):
            rx, ry = polar_sample(cx, cy, 0.3*self.spawn_radius, self.spawn_radius, self.np_random)
            rz = float(self.np_random.uniform(self.hand_z_min, self.hand_z_max)) + cz
            if math.dist((rx, ry, rz), rl) >= self.min_spawn_dist:
                fake = (rx, ry, rz); break
        if fake is None:
            fake = (rx, ry, rz)
        self.scene.set_hand_pos(*fake)

        dx, dy = fake[0] - lx0, fake[1] - ly0
        dxy = math.hypot(dx, dy)
        if dxy > self.max_xy_from_hand:
            nx, ny = dx/dxy, dy/dxy
            nlx, nly = fake[0] - nx*self.max_xy_from_hand, fake[1] - ny*self.max_xy_from_hand
        else:
            nlx, nly = lx0, ly0
        pitch = aim_pitch_deg((nlx, nly, lz0), fake)
        self.scene.set_tracking_light_pose(nlx, nly, lz0, pitch)

        for _ in range(3):
            self.world.step(False); self._sim_app.update()

        # 앞방향 부호 즉시 확정
        self.prev_q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        fwd = self.scene.get_robot_light_forward()
        toT, _ = self.scene.get_vec_robot_to_hand()
        dot = float(fwd[0]*toT[0] + fwd[1]*toT[1] + fwd[2]*toT[2])
        self._fwd_sign = +1.0 if dot >= 0.0 else -1.0
        print(f"[CALIB] forward_sign={self._fwd_sign:+.0f} (dot={dot:.3f})")

        # 🔧 보조 상태 초기화
        align0, _ = self._forward_alignment()
        self._prev_align = align0
        self._bad_align_count = 0

        self.episode_idx += 1
        print(f"[RESET] ep={self.episode_idx} q0(deg)={np.degrees(self.prev_q)}")

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if self._step_count < self.random_warmup_steps:
            action = self.action_space.sample().astype(np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)

        # 스케일링 (+ 부호 맵 적용)
        d_sim = (np.clip(action, -1.0, 1.0) * self.max_delta) * self.joint_sign

        hit_limit = 0
        tgt_cmd = np.zeros(4, dtype=np.float32)  # 로그용 타깃 기록

        # ── 경계 안전 마진(EPS) 적용 + 최소 가동폭 보장 + 경계 근접 반전 가드 ──
        MIN_SPAN = np.radians(5.0)
        for i, h in enumerate(self.joints):
            cur = self.dc.get_dof_position(h)
            lo_hw, hi_hw = self.joint_limits[i]

            lo_raw = float(self.sim_safe_lo[i])
            hi_raw = float(self.sim_safe_hi[i])

            lo = max(lo_hw, lo_raw + self._EPS_BOUND)
            hi = min(hi_hw, hi_raw - self._EPS_BOUND)

            if hi - lo < MIN_SPAN:
                mid = 0.5 * (lo + hi)
                lo = mid - 0.5 * MIN_SPAN
                hi = mid + 0.5 * MIN_SPAN

            near_hi = (cur >= hi - self._EPS_NEAR) and (d_sim[i] > 0)
            near_lo = (cur <= lo + self._EPS_NEAR) and (d_sim[i] < 0)
            if near_hi or near_lo:
                d_sim[i] = -abs(d_sim[i]) if near_hi else abs(d_sim[i])

            tgt = float(cur + float(d_sim[i]))
            tgt = float(np.clip(tgt, lo, hi))

            if tgt <= lo + 1e-6 or tgt >= hi - 1e-6:
                hit_limit += 1

            self.dc.set_dof_position_target(h, tgt)
            tgt_cmd[i] = tgt

        # 속도 걸고 시뮬 스텝
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(self.control_repeat):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # tracking light 조준 유지
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

        # 관측/리워드
        obs = self._get_obs()
        reward = self._compute_reward()

        # 성공/종료 판정용 정렬/거리
        align, dist = self._forward_alignment()

        dalign = align - self._prev_align
        self._prev_align = align

        if align < 0.0:
            self._bad_align_count += 1
        else:
            self._bad_align_count = 0

        self._step_count += 1
        success_ready = (self._step_count >= self.min_steps_before_success)
        success = success_ready and (dist <= self.success_dist) and (align >= self.success_align)

        # 베스트 리워드 갱신 기반 인내심
        if reward > self.best_reward + 1e-3:
            self.best_reward = reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        # 정지(스톨) 감지
        q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        dq = float(np.max(np.abs(q - self.prev_q))) if self.prev_q is not None else 0.0
        self.prev_q = q.copy()
        stalled = dq < 1e-4

        # ✅ 성공 보너스 + 시간 보너스
        if success:
            success_bonus = 10.0
            time_bonus = 1.0 - (self._step_count / self.episode_steps)
            reward += success_bonus + 0.5 * time_bonus

        terminated = bool(success)

        # ✅ 조기 종료 조건 완화 (히트/정렬)
        truncated = (
            (self._step_count >= self.episode_steps) or
            ((hit_limit >= 4) and (self._step_count > 60)) or   # was 3 & >30
            (self._no_improve_count >= self.patience) or
            (stalled and self._no_improve_count >= 200)
        )

        # ✅ 나쁜 정렬에 대한 인내심 완화
        bad_align_patience = 25  # was 6
        truncated = truncated or (self._bad_align_count >= bad_align_patience)

        # ── 디버깅: 현재값/타깃/거리/정렬/리워드 출력 ──
        cur_now = [self.dc.get_dof_position(h) for h in self.joints]
        print(
            "[STEP] ep={} t={} cur(deg)={} tgt(deg)={} hit={} dist={:.3f} align={:.3f} rew={:.3f}".format(
                self.episode_idx,
                self._step_count - 1,
                np.degrees(cur_now),
                np.degrees(tgt_cmd),
                hit_limit,
                dist, align, float(reward),
            )
        )

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
        }
        return obs, float(reward), terminated, truncated, info


    # ---------- obs/reward ----------
    def _get_obs(self):
        jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        hx, hy, hz = self.scene.get_hand_pos()
        rx, ry, rz = self.scene.get_robot_light_world_pos()
        vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
        dist = float(math.sqrt(vx*vx + vy*vy + vz*vz))
        return np.array([*jp, hx, hy, hz, rx, ry, rz, vx, vy, vz, dist], dtype=np.float32)

    def _forward_alignment(self):
        """라이트 로컬 -Z(앞)과 로봇→타겟 단위 벡터의 내적([-1,1]) 반환."""
        fwd = self.scene.get_robot_light_forward()
        fwd = (self._fwd_sign * fwd[0], self._fwd_sign * fwd[1], self._fwd_sign * fwd[2])
        toT, d = self.scene.get_vec_robot_to_hand()
        align = float(fwd[0]*toT[0] + fwd[1]*toT[1] + fwd[2]*toT[2])
        return align, d

    def _posture_prior(self, q, align_raw):
        ahead = (align_raw >= 0.0)
        idxs = [1, 2, 3]
        pen = 0.0
        for i in idxs:
            qi = float(q[i])
            if ahead:
                if qi > 0:
                    pen += np.tanh(abs(qi))
            else:
                if qi < 0:
                    pen += np.tanh(abs(qi))
        return pen

    def _compute_reward(self):
        align, d = self._forward_alignment()
        jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)

        # --- Potential-based shaping: Δ거리/Δ정렬 ---
        if self._prev_dist is None:
            self._prev_dist = d
        if self._prev_align_for_r is None:
            self._prev_align_for_r = align

        d_dist   = self._prev_dist - d            # 가까워지면 +
        d_align  = align - self._prev_align_for_r # 정면으로 돌면 +
        self._prev_dist = d
        self._prev_align_for_r = align

        # --- 기본 항목 ---
        r_align = align
        r_dalign = d_align

        # 너무 붙으면 벌점
        r_close = -max(0.0, 0.03 - d) * 6.0

        # ✅ 경계 패널티 완화
        near_low  = jp <= (self.sim_safe_lo + self._EPS_BOUND)
        near_high = jp >= (self.sim_safe_hi - self._EPS_BOUND)
        edge_cnt = int(np.sum(near_low | near_high))
        r_edge = -1.0 * edge_cnt   # was -2.5

        # 소프트 배리어
        qmid = 0.5 * (self.sim_safe_lo + self.sim_safe_hi)
        qspan = (self.sim_safe_hi - self.sim_safe_lo) + 1e-6
        closeness = np.maximum(0.0, 0.8 - np.abs((jp - qmid) / (0.5 * qspan)))
        r_soft_barrier = -1.0 * float(np.sum(closeness))

        # ✅ 행동 스무딩 완화
        if self.prev_q is not None:
            dq_vec = jp - self.prev_q
            r_smooth = -0.5 * float(np.sum(dq_vec * dq_vec))  # was -2.0
        else:
            r_smooth = 0.0

        # 2~4 부호 prior(약하게)
        post_pen = self._posture_prior(jp, align)

        # --- 가중치 밸런스 ---
        w_d_dist  = 2.0
        w_align   = 1.0
        w_d_align = 0.5
        w_post    = 0.03

        back_soft = -0.3 * max(0.0, -align)

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

        # 베이스 과회전 억제(완만)
        base_pen = 0.01 * abs(jp[0])
        reward -= base_pen

        # ✅ 리워드 클리핑(스케일 안정화)
        reward = np.clip(reward, -5.0, 5.0)
        return float(reward)


    def render(self):  # not used
        pass
