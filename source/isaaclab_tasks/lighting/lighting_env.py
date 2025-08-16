# lighting_env.py
import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .lighting_config import (
    SAFE_DEG_LO, SAFE_DEG_HI, HOME_DEG, MAX_DELTA, CONTROL_REPEAT, DRIVE_VELOCITY,
    RANDOM_WARMUP_STEPS, CURRICULUM, HAND_Z_RANGE, MIN_SPAWN_DIST, MAX_XY_FROM_HAND,
    PATIENCE, MIN_STEPS_BEFORE_SUCCESS,
    PLANE_PATH, HAND_PATH, TRACKING_LIGHT_PATH, ROBOT_LIGHT_PATH, DOFBOT_PATH, JOINT_NAMES,
    USE_SHADOW_REWARD, HAND_RADIUS, ROI_RADIUS,
    SUCCESS_SHADOW, MIN_IMPROVE_DELTA, NO_IMPROVE_PATIENCE,
    REWARD_REMOVED_K, REWARD_FINAL_K,
)
from .lighting_utils import deg2rad, polar_sample, aim_pitch_deg
from .lighting_scene import LightingScene

import logging
logging.basicConfig(level=logging.INFO)
class LightingEnv(gym.Env):
    """
    그림자 면적 기반 리워드 + 편법 방지:
      removed 증분 보상 / final 패널티
      + ROI 조준 보상, 입사각 패널티, 라이트 높이 제약
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, world_path: str, sim_app, episode_steps=400, use_curriculum=True):
        super().__init__()
        self._sim_app = sim_app
        self._world_path = world_path
        self.episode_steps = int(episode_steps)
        self.use_curriculum = bool(use_curriculum)

        try:
            from isaacsim.core import World
        except ImportError:
            from omni.isaac.core import World
        try:
            from isaacsim import dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        except ImportError:
            from omni.isaac.dynamic_control import _dynamic_control as dynamic_control
            _acquire_dc = dynamic_control.acquire_dynamic_control_interface
        from pxr import Usd, PhysxSchema

        # Action/Obs
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 + 3 + 3 + 3 + 1,), dtype=np.float32
        )

        # Stage/World
        self.scene = LightingScene(
            sim_app=self._sim_app, world_path=self._world_path,
            plane_path=PLANE_PATH, hand_path=HAND_PATH,
            tracking_light_path=TRACKING_LIGHT_PATH, robot_light_path=ROBOT_LIGHT_PATH
        )
        self.world = World()
        self.world.reset();  self._sim_app.update()
        self.world.play();   self._sim_app.update()
        for _ in range(2):
            self.world.step(False); self._sim_app.update()

        # --- Raycast binder (버전 호환) ---
        self._rc_fn = None
        self._ray_mode = "geom"   # 기본 폴백 모드
        self._bind_raycast_functions()

        # DC
        self.dc = _acquire_dc()

        # DOFBOT handles
        self.dof_path = self._find_dofbot_path(self.scene.robot_light_prim.GetStage(), DOFBOT_PATH, Usd, PhysxSchema)
        self.art = self.dc.get_articulation(self.dof_path)
        if self.art is None:
            raise RuntimeError(f"❌ Articulation not found at {self.dof_path}.")
        self.joint_names = JOINT_NAMES
        self.joints = [self.dc.find_articulation_dof(self.art, n) for n in self.joint_names]
        if any(h == 0 for h in self.joints):
            raise RuntimeError("❌ One or more DOF handles invalid. Check joint names / USD.")

        # Joint limits
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

        # Safe range
        self.sim_safe_lo = np.maximum(deg2rad(SAFE_DEG_LO), self.joint_limits[:, 0])
        self.sim_safe_hi = np.minimum(deg2rad(SAFE_DEG_HI), self.joint_limits[:, 1])
        for i in (1, 2, 3):
            self.sim_safe_lo[i] = max(self.sim_safe_lo[i], np.radians(-120.0))
        self.home_pose = deg2rad(HOME_DEG)

        # Control params
        self.max_delta = np.asarray(MAX_DELTA, dtype=np.float32)
        self.control_repeat = int(CONTROL_REPEAT)
        self.drive_velocity = float(DRIVE_VELOCITY)
        self.random_warmup_steps = int(RANDOM_WARMUP_STEPS)
        self.joint_sign = np.array([+1, -1, -1, -1], dtype=np.float32)

        # State
        self._fwd_sign = +1.0
        self.episode_idx = 0
        self._step_count = 0
        self.prev_q = None

        # Curriculum/Spawn
        self.spawn_radius = float(CURRICULUM[0][1])
        self.hand_z_min, self.hand_z_max = HAND_Z_RANGE
        self.min_spawn_dist = float(MIN_SPAWN_DIST)
        self.max_xy_from_hand = float(MAX_XY_FROM_HAND)

        # Termination thresholds
        self.patience = int(PATIENCE)
        self.min_steps_before_success = int(MIN_STEPS_BEFORE_SUCCESS)

        # EPS
        self._EPS_BOUND = np.radians(1.0)
        self._EPS_NEAR  = np.radians(1.0)

        # Shadow tracker
        self._orig_shadow_ratio = 0.0
        self._prev_removed = 0.0
        self._no_improve_steps = 0
        self._best_final_shadow = 1.0

        # Episode-fixed hand & ROI
        self._hand_fixed = None
        self._roi_center = None

        # Diagnostics
        self._shadow_grid = 64
        self._ray_verbose = bool(int(os.environ.get("RAY_VERBOSE", "0")))
        self._diag = dict(ray_calls=0, ray_used=0, ray_errors=0,
                          ray_hand_hits=0, sphere_hits=0, mismatches=0)

    # ---------- Ray binder ----------
    def _bind_raycast_functions(self):
        """다양한 모듈/시그니처를 포괄하는 레이캐스트 래퍼를 바인딩한다."""
        try:
            import importlib
            cand_modules = [
                "omni.isaac.core.utils.scene_query",
                "omni.physx.scripts.utils.scene_query",
                "omni.physx.scripts.scene_query",
                "omni.physx.scene_query",
            ]
            cand_names = ["raycast_closest_hit", "raycast_any_hit", "raycast"]

            def make_wrapper(raw_fn):
                # 모든 시그니처를 시도하는 범용 래퍼
                def _wrapper(start_pos, end_pos):
                    # 키워드
                    try:
                        return raw_fn(start_position=tuple(start_pos), end_position=tuple(end_pos))
                    except TypeError:
                        pass
                    # 위치
                    try:
                        return raw_fn(tuple(start_pos), tuple(end_pos))
                    except TypeError:
                        pass
                    # stage 요구
                    try:
                        stage = self.scene.robot_light_prim.GetStage()
                        return raw_fn(stage, tuple(start_pos), tuple(end_pos))
                    except TypeError:
                        # 더 복잡한 컨텍스트를 요구하는 버전은 여기선 미지원
                        raise
                return _wrapper

            for mn in cand_modules:
                try:
                    mod = importlib.import_module(mn)
                except Exception:
                    continue
                for fn_name in cand_names:
                    fn = getattr(mod, fn_name, None)
                    if not callable(fn):
                        continue
                    self._rc_fn = make_wrapper(fn)
                    self._ray_mode = "closest" if "closest" in fn_name else ("any" if "any" in fn_name else "ray")
                    return

            # PhysX Scene Query Interface 폴백
            try:
                import omni.physx as _physx
                sq = _physx.get_physx_scene_query_interface()
                if sq is not None:
                    import carb
                    # 어떤 메서드가 있는지 확인
                    has_closest = hasattr(sq, "raycast_closest")
                    has_any     = hasattr(sq, "raycast_any")

                    if not (has_closest or has_any):
                        raise RuntimeError("PhysXSceneQuery: no raycast_closest / raycast_any")

                    def _as_f3(v):
                        # v: (x, y, z) → carb.Float3
                        return carb.Float3(float(v[0]), float(v[1]), float(v[2]))

                    def _wrapper_physx(start_pos, end_pos):
                        import numpy as _np
                        origin = _np.array(start_pos, dtype=float)
                        end    = _np.array(end_pos,   dtype=float)
                        dvec   = end - origin
                        dist   = float(_np.linalg.norm(dvec) + 1e-9)
                        direction = dvec / dist

                        o = _as_f3(origin)
                        d = _as_f3(direction)

                        # 1) raycast_closest(origin, dir, distance, bothSides=False) -> dict
                        if has_closest:
                            try:
                                res = sq.raycast_closest(o, d, dist, False)
                                # 표준화: dict 예상. 없으면 bool로 변환
                                if isinstance(res, dict):
                                    # dict 구조: {"hit": bool, "position": carb.Float3, "normal": carb.Float3, ...} 등
                                    hit = bool(res.get("hit", True))  # 일부 빌드는 항상 dict 반환
                                    # collider/path 정보는 보통 없음 → (hit, "") 형태로 넘겨서 _extract_hit_path에서 처리
                                    return (hit, "")
                                return bool(res)
                            except TypeError:
                                # 혹시 다른 빌드(위치 인자만) → 다시 시도
                                res = sq.raycast_closest(o, d, dist)
                                if isinstance(res, dict):
                                    return (bool(res.get("hit", True)), "")
                                return bool(res)

                        # 2) raycast_any(origin, dir, distance) -> bool
                        if has_any:
                            try:
                                hit = sq.raycast_any(o, d, dist)
                            except TypeError:
                                hit = sq.raycast_any(o, d, dist, False)
                            return bool(hit)

                        return False

                    self._rc_fn = _wrapper_physx
                    self._ray_mode = "physx-closest" if has_closest else "physx-any"
                    return
            except Exception:
                pass

        except Exception:
            pass

        self._rc_fn = None
        self._ray_mode = "geom"

    # ---------- Utils ----------
    def _diag_reset(self):
        self._diag.update(ray_calls=0, ray_used=0, ray_errors=0,
                          ray_hand_hits=0, sphere_hits=0, mismatches=0)

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
        raise RuntimeError("❌ DOFBOT articulation prim not found on Stage.")

    def _apply_curriculum(self):
        if not self.use_curriculum:
            return
        ep = self.episode_idx
        for th, r in CURRICULUM:
            if ep >= th:
                self.spawn_radius = float(r)

    def _get_robot_light_world_pos(self):
        return self.scene.get_robot_light_world_pos()

    def _get_tracking_light_world_pos(self):
        return self.scene.get_tracking_light_pos()

    # --- 폴백: 선분-구 교차 ---
    def _segment_hits_sphere(self, p0, p1, c, r):
        d = np.array(p1, dtype=float) - np.array(p0, dtype=float)
        f = np.array(p0, dtype=float) - np.array(c,  dtype=float)
        a = float(np.dot(d, d)); b = 2.0 * float(np.dot(f, d))
        cterm = float(np.dot(f, f) - r*r)
        disc = b*b - 4*a*cterm
        if disc < 0.0: return False
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a); t2 = (-b + sqrt_disc) / (2*a)
        return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

    def _extract_hit_path(self, res):
        """다양한 Isaac/PhysX 레이캐스트 반환 타입에서 collider path를 추출."""
        try:
            if res is None:
                return False, ""
            # 단순 bool만 오는 폴백/IF 버전
            if isinstance(res, bool):
                return (bool(res), "")
            # dict 패턴
            if isinstance(res, dict):
                path = res.get("collider") or res.get("rigid_body") or res.get("primPath") or res.get("path") or ""
                return (True, str(path)) if path else (True, "")
            # tuple/list 패턴
            if isinstance(res, (list, tuple)):
                if len(res) == 2 and isinstance(res[0], bool):
                    return bool(res[0]), str(res[1])
                for e in res:
                    if isinstance(e, str):
                        return True, e
                    for attr in ("collider", "primPath", "path"):
                        if hasattr(e, attr):
                            return True, str(getattr(e, attr))
                return True, ""
            # 객체 패턴
            for attr in ("collider", "primPath", "path"):
                if hasattr(res, attr):
                    return True, str(getattr(res, attr))
            # dict like
            try:
                path = res.get("collider")  # type: ignore[attr-defined]
                if path:
                    return True, str(path)
            except Exception:
                pass
            return True, ""
        except Exception:
            return False, ""

    # --- 레이캐스트/폴백 통합 ---
    def _blocked_by_hand(self, light_pos, p):
        """True면 'p(ROI)→light' 경로가 fake_hand에 가려진 것."""
        self._diag["ray_calls"] += 1
        p_eps = (float(p[0]), float(p[1]), float(p[2]) + 1e-3)

        # (A) 구 교차 폴백: 항상 가능
        hx, hy, hz = self._hand_fixed if self._hand_fixed is not None else self.scene.get_hand_pos()
        sphere_hit = self._segment_hits_sphere(p_eps, light_pos, (hx, hy, hz), float(HAND_RADIUS))
        if sphere_hit: self._diag["sphere_hits"] += 1

        # (B) 레이캐스트(가능하면)
        ray_hit_hand = None
        hit_any = None  # ← 레이가 뭘 맞았는지 여부(physx 모드에선 경로가 없어도 bool은 얻음)
        if self._rc_fn is not None:
            try:
                res = self._rc_fn(p_eps, tuple(light_pos))
                self._diag["ray_used"] += 1

                hit, collider_path = self._extract_hit_path(res)
                hit_any = bool(hit)

                # 기본: 경로 기반 매칭 시도
                if hit_any:
                    cp = str(collider_path or "")
                    # (A) 경로를 받는 빌드(isaac.core.utils.scene_query 등)
                    if cp:
                        is_hand = (cp.startswith(HAND_PATH) or (HAND_PATH in cp))
                        ray_hit_hand = bool(is_hand)
                        if self._ray_verbose:
                            if sphere_hit:
                                logging.info(f"[RAY] hit path='' (physx) -> use sphere_hit={sphere_hit} as hand")
                    else:
                        # (B) physx 폴백 빌드: 경로가 없다 → hand 여부는 sphere 교차로 판정
                        #    - 'hit_any'는 “무언가”를 맞췄다는 뜻이므로,
                        #    - 실제 hand 차폐 여부는 구-선분 교차 결과(sphere_hit)로 대체
                        ray_hit_hand = bool(sphere_hit)
                        if self._ray_verbose:
                            print(f"[RAY] hit path='' (physx) -> use sphere_hit={sphere_hit} as hand")
                else:
                    ray_hit_hand = False

            except Exception as e:
                self._diag["ray_errors"] += 1
                if self._ray_verbose:
                    print(f"[RAY] error: {e}")

        # (C) 비교/기록
        if ray_hit_hand is not None and (ray_hit_hand != sphere_hit):
            self._diag["mismatches"] += 1

        # (D) 최종판정
        return bool(ray_hit_hand) if ray_hit_hand is not None else bool(sphere_hit)

    # --- ROI 중심 사용 샘플링 ---
    def _shadow_ratio_from_light(self, light_pos, grid=64, roi_center=None):
        if roi_center is None: cx, cy, cz = self.scene.get_plane_center()
        else: cx, cy, cz = roi_center
        xs = np.linspace(-1.0, 1.0, grid); ys = np.linspace(-1.0, 1.0, grid)
        hit = total = 0
        for u in xs:
            for v in ys:
                if u*u + v*v > 1.0: continue
                px = cx + u * float(ROI_RADIUS); py = cy + v * float(ROI_RADIUS)
                p  = (px, py, cz); total += 1
                if self._blocked_by_hand(light_pos, p): hit += 1
        return float(hit / total) if total > 0 else 0.0

    def _shadow_ratio_two_lights(self, lightA, lightB, grid=64, roi_center=None):
        if roi_center is None: cx, cy, cz = self.scene.get_plane_center()
        else: cx, cy, cz = roi_center
        xs = np.linspace(-1.0, 1.0, grid); ys = np.linspace(-1.0, 1.0, grid)
        both = total = 0
        for u in xs:
            for v in ys:
                if u*u + v*v > 1.0: continue
                px = cx + u * float(ROI_RADIUS); py = cy + v * float(ROI_RADIUS)
                p  = (px, py, cz); total += 1
                bA = self._blocked_by_hand(lightA, p); bB = self._blocked_by_hand(lightB, p)
                if bA and bB: both += 1
        return float(both / total) if total > 0 else 0.0

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._no_improve_steps = 0
        self._best_final_shadow = 1.0
        self._prev_removed = 0.0
        self.prev_q = None
        self._apply_curriculum()
        self._diag_reset()

        # 랜덤 전방 포즈
        margin = np.radians(3.0)
        q_lo = self.sim_safe_lo + margin
        q_hi = self.sim_safe_hi - margin
        for i in (1, 2, 3): q_hi[i] = min(q_hi[i], np.radians(-5.0))
        q0 = np.random.uniform(q_lo, q_hi).astype(np.float32)
        for h, q in zip(self.joints, q0):
            self.dc.set_dof_position_target(h, float(q))
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(30):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # 베이스 캐시 & hand 샘플링/고정
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
        if fake is None: fake = (rx, ry, rz)
        self.scene.set_hand_pos(*fake)
        self._hand_fixed = tuple(fake)

        # 트래킹 라이트 포인팅
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

        # 정면/부호 세팅
        self.prev_q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        fwd = self.scene.get_robot_light_forward()
        toT, _ = self.scene.get_vec_robot_to_hand()
        dot = float(fwd[0]*toT[0] + fwd[1]*toT[1] + fwd[2]*toT[2])
        self._fwd_sign = +1.0 if dot >= 0.0 else -1.0

        # ROI 중심(트래킹광 → 손 선과 평면 교점) & orig 그림자
        plane_x, plane_y, plane_z = self.scene.get_plane_center()
        Lt = self._get_tracking_light_world_pos()
        fx, fy, fz = self._hand_fixed
        den = (fz - Lt[2])
        if abs(den) < 1e-6:
            roi_cx, roi_cy = plane_x, plane_y
        else:
            t = (plane_z - Lt[2]) / den
            roi_cx = Lt[0] + t * (fx - Lt[0])
            roi_cy = Lt[1] + t * (fy - Lt[1])
        self._roi_center = (float(roi_cx), float(roi_cy), float(plane_z))

        valid_shadow = (USE_SHADOW_REWARD and (Lt[2] > plane_z + 1e-3) and (fz > plane_z + 1e-3))
        self._orig_shadow_ratio = self._shadow_ratio_from_light(
            Lt, grid=self._shadow_grid, roi_center=self._roi_center
        ) if valid_shadow else 0.0

        # 간단 셀프테스트: 레이 사용 카운터 증가 여부 확인
        try:
            import random as _r
            cxr, cyr, czr = self._roi_center
            for _ in range(16):
                while True:
                    u = _r.uniform(-1.0, 1.0); v = _r.uniform(-1.0, 1.0)
                    if u*u + v*v <= 1.0: break
                px = cxr + u * float(ROI_RADIUS); py = cyr + v * float(ROI_RADIUS)
                _ = self._blocked_by_hand(Lt, (px, py, czr))
        except Exception:
            pass

        self.episode_idx += 1
        print(f"[RESET] ep={self.episode_idx} hand=({self._hand_fixed[0]:.3f},{self._hand_fixed[1]:.3f},{self._hand_fixed[2]:.3f}) "
              f"roi=({self._roi_center[0]:.3f},{self._roi_center[1]:.3f},{self._roi_center[2]:.3f}) "
              f"orig_shadow={self._orig_shadow_ratio:.3f} ray={self._ray_mode} grid={self._shadow_grid} warmup={self.random_warmup_steps}")
        return self._get_obs(), {}

    def step(self, action):
        # episode-fixed hand 유지
        if self._hand_fixed is not None:
            fx, fy, fz = self._hand_fixed
            hx, hy, hz = self.scene.get_hand_pos()
            if (abs(hx - fx) + abs(hy - fy) + abs(hz - fz)) > 1e-7:
                self.scene.set_hand_pos(fx, fy, fz)

        # 워밍업 / 액션 스케일
        if self._step_count < self.random_warmup_steps:
            action = self.action_space.sample().astype(np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)
        d_sim = (np.clip(action, -1.0, 1.0) * self.max_delta) * self.joint_sign

        # 조인트 타겟/스텝
        hit_limit = 0
        MIN_SPAN = np.radians(5.0)
        for i, h in enumerate(self.joints):
            cur = self.dc.get_dof_position(h)
            lo_hw, hi_hw = self.joint_limits[i]
            lo_raw = float(self.sim_safe_lo[i]); hi_raw = float(self.sim_safe_hi[i])
            lo = max(lo_hw, lo_raw + self._EPS_BOUND)
            hi = min(hi_hw, hi_raw - self._EPS_BOUND)
            if hi - lo < MIN_SPAN:
                mid = 0.5 * (lo + hi); lo = mid - 0.5 * MIN_SPAN; hi = mid + 0.5 * MIN_SPAN
            if (cur >= hi - self._EPS_NEAR and d_sim[i] > 0) or (cur <= lo + self._EPS_NEAR and d_sim[i] < 0):
                d_sim[i] = -d_sim[i]
            tgt = float(np.clip(cur + float(d_sim[i]), lo, hi))
            if tgt <= lo + 1e-6 or tgt >= hi - 1e-6: hit_limit += 1
            self.dc.set_dof_position_target(h, tgt)

        for h in self.joints:
            self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(self.control_repeat):
            self.world.step(False); self._sim_app.update()
        for h in self.joints:
            self.dc.set_dof_velocity_target(h, 0.0)

        # 트래킹 라이트 포인팅 유지
        fx, fy, fz = self._hand_fixed
        lx0, ly0, lz0 = self.scene.tracking_base
        dx, dy = fx - lx0, fy - ly0
        dxy = math.hypot(dx, dy)
        if dxy > self.max_xy_from_hand:
            nx, ny = (dx / dxy, dy / dxy)
            nlx = fx - nx * self.max_xy_from_hand
            nly = fy - ny * self.max_xy_from_hand
        else:
            nlx, nly = lx0, ly0
        pitch = aim_pitch_deg((nlx, nly, lz0), (fx, fy, fz))
        self.scene.set_tracking_light_pose(nlx, nly, lz0, pitch)

        # 관절 상태
        jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)

        # === 최종 그림자(final) & removed 계산 ===
        plane_z = self.scene.get_plane_center()[2]
        Lr = self._get_robot_light_world_pos()
        Lt = self._get_tracking_light_world_pos()

        shadow_final = 0.0
        removed = 0.0
        valid_shadow = (
            USE_SHADOW_REWARD and
            (Lr[2] > plane_z + 1e-3) and (Lt[2] > plane_z + 1e-3) and (fz > plane_z + 1e-3)
        )
        if valid_shadow:
            shadow_final = self._shadow_ratio_two_lights(Lt, Lr, grid=self._shadow_grid, roi_center=self._roi_center)
            removed = max(0.0, self._orig_shadow_ratio - shadow_final)

        # 개선 추적
        if shadow_final < self._best_final_shadow - MIN_IMPROVE_DELTA:
            self._best_final_shadow = shadow_final; self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1

        # 정렬/거리 (뒤돌아봄 감지용만 사용 — XY)
        align, dist = self._forward_alignment()

        # === 리워드 ===
        reward = 0.0
        reward += -REWARD_FINAL_K * float(np.clip(shadow_final, 0.0, 1.0))
        reward += REWARD_REMOVED_K * (removed - self._prev_removed)

        # 제어/안전 패널티
        edge_cnt = int(np.sum((jp <= (self.sim_safe_lo + self._EPS_BOUND)) |
                              (jp >= (self.sim_safe_hi - self._EPS_BOUND))))
        u_l2 = float(np.sum(np.square(np.asarray(action, dtype=np.float32))))
        reward += -0.3 * edge_cnt
        reward += -0.01 * u_l2
        if self.prev_q is not None:
            dq_vec = jp - self.prev_q
            reward += -0.1 * float(np.sum(dq_vec * dq_vec))
        self.prev_q = jp.copy()

        # 편법 방지 보상: ROI 조준/입사각/높이
        fwd_vec = self.scene.get_robot_light_forward()
        cx, cy, cz = self._roi_center if self._roi_center is not None else self.scene.get_plane_center()
        to_roi = np.array([cx - Lr[0], cy - Lr[1], cz - Lr[2]], dtype=np.float32)
        to_roi /= (np.linalg.norm(to_roi) + 1e-8)
        cos_inc = float(fwd_vec[0]*to_roi[0] + fwd_vec[1]*to_roi[1] + fwd_vec[2]*to_roi[2])
        pointing = max(0.0, cos_inc)
        reward += 0.6 * pointing
        COS_MIN = 0.342  # cos(70°)
        reward += -0.8 * max(0.0, COS_MIN - cos_inc)
        min_clearance = 0.05
        clearance = (Lr[2] - plane_z) - min_clearance
        reward += (-2.0 * max(0.0, -clearance)) + (0.1 * max(0.0, min(clearance, 0.03)))

        self._prev_removed = removed
        self._step_count += 1

        # === 종료 ===
        success_ready = (self._step_count >= self.min_steps_before_success)
        success = success_ready and (shadow_final <= SUCCESS_SHADOW)
        plateau_stop = (self._no_improve_steps >= NO_IMPROVE_PATIENCE)
        backview_stop = (self._step_count > 25) and (align < -0.2)

        terminated = bool(success)
        truncated = (
            (self._step_count >= self.episode_steps) or
            plateau_stop or
            backview_stop or
            (hit_limit >= 4 and self._step_count > 60)
        )

        if success: reward += 8.0
        if truncated and not terminated: reward += -1.0

        print(
            f"[STEP] ep={self.episode_idx:04d} t={self._step_count:03d} "
            f"orig={self._orig_shadow_ratio:.3f} final={shadow_final:.3f} removed={removed:.3f} "
            f"best_final={self._best_final_shadow:.3f} noImp={self._no_improve_steps:02d}/{NO_IMPROVE_PATIENCE} "
            f"alignXY={align:.3f} dist={dist:.3f} rew={reward:.3f} ray_used={self._diag['ray_used']}"
        )

        # --- W&B info ---
        cx_log = float(self._roi_center[0]) if self._roi_center is not None else float("nan")
        cy_log = float(self._roi_center[1]) if self._roi_center is not None else float("nan")
        ray_calls = float(self._diag["ray_calls"]); ray_used = float(self._diag["ray_used"])
        ray_err = float(self._diag["ray_errors"]); ray_hits = float(self._diag["ray_hand_hits"])
        sphere_hits = float(self._diag["sphere_hits"]); mism = float(self._diag["mismatches"])
        info = {
            "orig_shadow": float(self._orig_shadow_ratio),
            "final_shadow": float(shadow_final),
            "removed_shadow": float(removed),
            "best_final_shadow": float(self._best_final_shadow),
            "no_improve_steps": int(self._no_improve_steps),
            "dist": float(dist),
            "align": float(align),
            "hit_limit": int(hit_limit),
            "step": int(self._step_count),
            "reward": float(reward),
            "success": 1.0 if success else 0.0,
            "plateau": 1.0 if plateau_stop else 0.0,
            # diag
            "diag/roi_cx": cx_log, "diag/roi_cy": cy_log,
            "diag/plane_z": float(self.scene.get_plane_center()[2]),
            "diag/hand_z": float(fz), "diag/Lt_z": float(Lt[2]), "diag/Lr_z": float(Lr[2]),
            "diag/shadow_grid": float(self._shadow_grid),
            "diag/ray_mode": {"geom":0.0, "any":1.0, "closest":2.0, "ray":3.0, "physx-if":4.0}.get(self._ray_mode, -1.0),
            "diag/ray_calls": ray_calls, "diag/ray_used": ray_used, "diag/ray_errors": ray_err,
            "diag/ray_hand_hits": ray_hits, "diag/sphere_hits": sphere_hits,
            "diag/ray_vs_sphere_mismatch": mism,
            "diag/ray_ok_rate": (ray_used / max(1.0, ray_calls)),
            "diag/ray_hand_hit_rate": (ray_hits / max(1.0, ray_used)) if ray_used > 0 else 0.0,
            "diag/cos_inc": float(cos_inc), "diag/pointing": float(pointing),
            "diag/clearance": float(Lr[2] - plane_z),
        }

        obs = self._get_obs(jp_override=jp, hand_override=self._hand_fixed)
        return obs, float(reward), terminated, truncated, info

    # ---------- Obs/Reward utils ----------
    def _get_obs(self, jp_override=None, hand_override=None):
        if jp_override is None:
            jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
        else:
            jp = np.asarray(jp_override, dtype=np.float32)

        if hand_override is None and self._hand_fixed is not None:
            hx, hy, hz = self._hand_fixed
        elif hand_override is None:
            hx, hy, hz = self.scene.get_hand_pos()
        else:
            hx, hy, hz = hand_override

        rx, ry, rz = self.scene.get_robot_light_world_pos()
        vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
        dist = float(math.sqrt(vx*vx + vy*vy + vz*vz))
        return np.array([*jp, hx, hy, hz, rx, ry, rz, vx, vy, vz, dist], dtype=np.float32)

    def _forward_alignment(self):
        """뒤돌아봄 감지용: XY 평면에서의 정렬만 사용."""
        fwd = self.scene.get_robot_light_forward()
        toT, d = self.scene.get_vec_robot_to_hand()
        fx, fy = fwd[0], fwd[1]; tx, ty = toT[0], toT[1]
        n1 = math.sqrt(fx*fx + fy*fy) + 1e-8
        n2 = math.sqrt(tx*tx + ty*ty) + 1e-8
        align_xy = (fx/n1)*(tx/n2) + (fy/n1)*(ty/n2)
        return float(align_xy), float(d)

    def render(self):
        pass
