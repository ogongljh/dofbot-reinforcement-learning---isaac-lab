import math
from typing import List, Optional, Tuple, Callable, Dict, Any, cast

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .lighting_config import (
    SAFE_DEG_LO, SAFE_DEG_HI, HOME_DEG, MAX_DELTA, CONTROL_REPEAT, DRIVE_VELOCITY,
    RANDOM_WARMUP_STEPS, CURRICULUM, HAND_Z_RANGE, MIN_SPAWN_DIST, MAX_XY_FROM_HAND,
    MIN_STEPS_BEFORE_SUCCESS, PLANE_PATH, HAND_PATH, TRACKING_LIGHT_PATH, ROBOT_LIGHT_PATH, DOFBOT_PATH, JOINT_NAMES,
    USE_SHADOW_REWARD, HAND_RADIUS, ROI_RADIUS,
    SUCCESS_SHADOW, MIN_IMPROVE_DELTA, NO_IMPROVE_PATIENCE,
)
from .lighting_utils import deg2rad, polar_sample, aim_pitch_deg
from .lighting_scene import LightingScene

try:
    from omni.isaac.core.utils.scene_query import raycast_closest_hit, raycast_any_hit
except Exception:
    raycast_closest_hit = None
    raycast_any_hit = None


def _ns(prefix: str, i: int, rel: str) -> str:
    """'/World' 루트 유지 + env_i 네임스페이스."""
    assert rel.startswith("/World/")
    return f"/World/{prefix}_{i}" + rel[len("/World") :]


class LightingEnvMulti(gym.Env):
    """
    한 스테이지에 로봇 세트 N개를 복제해 병렬 학습.
      - action: (num*4,), obs: (num*14,)
      - reward: 개별 보상의 평균
      - done: 모두 성공 or 전원 plateau or step limit
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, world_path: str, sim_app, num_actors: int = 4, episode_steps: int = 400):
        super().__init__()
        assert num_actors >= 1
        self._sim_app = sim_app
        self._world_path = world_path
        self.num = int(num_actors)
        self.episode_steps = int(episode_steps)

        # Isaac World/DC
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
        from pxr import UsdGeom, Sdf, PhysxSchema

        # Stage 로드
        self.scene = LightingScene(
            sim_app=self._sim_app,
            world_path=self._world_path,
            plane_path=PLANE_PATH,
            hand_path=HAND_PATH,
            tracking_light_path=TRACKING_LIGHT_PATH,
            robot_light_path=ROBOT_LIGHT_PATH,
        )
        stage = self.scene.robot_light_prim.GetStage()

        self.world = World()
        self.world.reset();  self._sim_app.update()
        self.world.play();   self._sim_app.update()
        for _ in range(2):
            self.world.step(False); self._sim_app.update()

        # 원본 경로
        self.base_paths = dict(
            dof=DOFBOT_PATH, hand=HAND_PATH, plane=PLANE_PATH,
            track=TRACKING_LIGHT_PATH, light=ROBOT_LIGHT_PATH,
        )

        # 네임스페이스 루트 만들기
        self.names = [f"env_{i}" for i in range(self.num)]
        for i in range(self.num):
            root = Sdf.Path(f"/World/{self.names[i]}")
            if not stage.GetPrimAtPath(root):
                UsdGeom.Xform.Define(stage, root)

        # ▶ 복제: 단순 DefinePrim 대신 "내부 참조(Reference)"로 원본 서브트리 연결 (핵심!)
        for i in range(self.num):
            for _, base in self.base_paths.items():
                src = stage.GetPrimAtPath(base)
                if not src or not src.IsValid():
                    raise RuntimeError(f"Missing base prim: {base}")
                dst_path = Sdf.Path(_ns(self.names[i], i, base))
                dst = stage.DefinePrim(dst_path, src.GetTypeName())
                refs = dst.GetReferences()
                refs.ClearReferences()
                refs.AddInternalReference(src.GetPath())  # 관절/링크 전체가 복제됨
        self._sim_app.update()

        # DC
        self.dc = _acquire_dc()

        # 각 로봇 핸들/한계
        self.joints: List[List[int]] = []
        self.joint_limits: List[np.ndarray] = []
        self.sim_safe_lo_list: List[np.ndarray] = []
        self.sim_safe_hi_list: List[np.ndarray] = []
        self.sim_safe_lo: Optional[np.ndarray] = None   # (num,4)
        self.sim_safe_hi: Optional[np.ndarray] = None
        self.home_pose = deg2rad(HOME_DEG)

        for i in range(self.num):
            dof_path = _ns(self.names[i], i, DOFBOT_PATH)
            art = self.dc.get_articulation(dof_path)
            if art is None:
                # 네임스페이스 내부 경로로 탐색(백업)
                art = self._find_articulation(stage, dof_path, PhysxSchema)
            if art is None:
                raise RuntimeError(f"[{i}] articulation not found at {dof_path}")

            handles = [self.dc.find_articulation_dof(art, n) for n in JOINT_NAMES]
            if any(h == 0 for h in handles):
                raise RuntimeError(f"[{i}] invalid DOF handle(s).")
            self.joints.append(handles)

            jlim = []
            for h in handles:
                props = self.dc.get_dof_properties(h)
                if getattr(props, "hasLimits", False):
                    lo = float(np.array(getattr(props, "lower")).reshape(-1)[0])
                    hi = float(np.array(getattr(props, "upper")).reshape(-1)[0])
                else:
                    lo, hi = -math.pi, math.pi
                jlim.append((lo, hi))
            jlim = np.array(jlim, dtype=np.float32)
            self.joint_limits.append(jlim)

            lo = np.maximum(deg2rad(SAFE_DEG_LO), jlim[:, 0])
            hi = np.minimum(deg2rad(SAFE_DEG_HI), jlim[:, 1])
            for k in (1, 2, 3):
                lo[k] = max(lo[k], np.radians(-120.0))
            self.sim_safe_lo_list.append(lo)
            self.sim_safe_hi_list.append(hi)

        self.sim_safe_lo = np.stack(self.sim_safe_lo_list, axis=0)
        self.sim_safe_hi = np.stack(self.sim_safe_hi_list, axis=0)

        # 제어 파라미터
        self.max_delta = np.asarray(MAX_DELTA, dtype=np.float32)
        self.drive_velocity = float(DRIVE_VELOCITY)
        self.joint_sign = np.array([+1, -1, -1, -1], dtype=np.float32)

        # 상태 버퍼/시그널
        self.spawn_radius = float(CURRICULUM[0][1])
        self.hand_z_min, self.hand_z_max = HAND_Z_RANGE
        self.min_spawn_dist = float(MIN_SPAWN_DIST)
        self.max_xy_from_hand = float(MAX_XY_FROM_HAND)

        self._step = 0
        self._best_shadow = np.ones(self.num, dtype=np.float32)
        self._noimp = np.zeros(self.num, dtype=np.int32)
        self._phi_prev = np.full(self.num, np.nan, dtype=np.float32)

        # 타입 명시(에디터 경고 제거)
        self.tracking_base: List[Optional[Tuple[float, float, float]]] = [None] * self.num
        self.hand_fixed: List[Optional[Tuple[float, float, float]]] = [None] * self.num
        self.prev_q: List[Optional[np.ndarray]] = [None] * self.num

        # Spaces
        self.obs_dim_single = 4 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim_single * self.num,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4 * self.num,), dtype=np.float32
        )

    # ------------- Gym API -------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._best_shadow.fill(1.0)
        self._noimp.fill(0)
        self._phi_prev.fill(np.nan)
        self.prev_q = [None] * self.num

        assert self.sim_safe_lo is not None and self.sim_safe_hi is not None

        # 초기 조인트 세팅
        margin = np.radians(3.0)
        for i in range(self.num):
            q_lo = self.sim_safe_lo[i] + margin
            q_hi = self.sim_safe_hi[i] - margin
            for k in (1, 2, 3):
                q_hi[k] = min(q_hi[k], np.radians(-5.0))
            q0 = np.random.uniform(q_lo, q_hi).astype(np.float32)
            for h, q in zip(self.joints[i], q0):
                self.dc.set_dof_position_target(h, float(q))
                self.dc.set_dof_velocity_target(h, self.drive_velocity)

        for _ in range(30):
            self.world.step(False); self._sim_app.update()
        for i in range(self.num):
            for h in self.joints[i]:
                self.dc.set_dof_velocity_target(h, 0.0)

        # 각 배우별 tracking base & 손 스폰 & 추적광 정렬
        for i in range(self.num):
            lx0, ly0, lz0 = self._get_tracking_base(i)
            self.tracking_base[i] = (float(lx0), float(ly0), float(lz0))

            cx, cy, cz = self._get_plane_center(i)
            rl = self._get_robot_light_pos(i)

            fake = None
            for _ in range(40):
                rx, ry = polar_sample(cx, cy, 0.3*self.spawn_radius, self.spawn_radius, np.random)
                rz = float(np.random.uniform(self.hand_z_min, self.hand_z_max)) + cz
                if math.dist((rx, ry, rz), rl) >= self.min_spawn_dist:
                    fake = (rx, ry, rz); break
            if fake is None:
                fake = (rx, ry, rz)
            self._set_hand_pos(i, *fake)
            self.hand_fixed[i] = (float(fake[0]), float(fake[1]), float(fake[2]))

            # 추적광 정렬
            dx, dy = fake[0] - lx0, fake[1] - ly0
            dxy = math.hypot(dx, dy)
            if dxy > 1e-9:
                nx, ny = dx/dxy, dy/dxy
                nlx, nly = fake[0] - nx*self.max_xy_from_hand, fake[1] - ny*self.max_xy_from_hand
            else:
                nlx, nly = lx0, ly0
            pitch = aim_pitch_deg((nlx, nly, lz0), fake)
            self._set_tracking_pose(i, nlx, nly, lz0, pitch)

        for _ in range(3):
            self.world.step(False); self._sim_app.update()

        obs = self._get_obs_all()
        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.num, 4)

        # 손 위치 고정(에피소드 내 불변)
        for i in range(self.num):
            hf = self.hand_fixed[i]; assert hf is not None
            fx, fy, fz = hf
            hx, hy, hz = self._get_hand_pos(i)
            if (abs(hx - fx) + abs(hy - fy) + abs(hz - fz)) > 1e-7:
                self._set_hand_pos(i, fx, fy, fz)

        # 조인트 명령
        hit_limit = np.zeros(self.num, dtype=np.int32)
        MIN_SPAN = np.radians(5.0)
        assert self.sim_safe_lo is not None and self.sim_safe_hi is not None

        for i in range(self.num):
            d_sim = (np.clip(action[i], -1.0, 1.0) * self.max_delta) * self.joint_sign
            for k, h in enumerate(self.joints[i]):
                cur = self.dc.get_dof_position(h)
                lo_hw, hi_hw = self.joint_limits[i][k]
                lo_raw, hi_raw = float(self.sim_safe_lo[i][k]), float(self.sim_safe_hi[i][k])
                lo = max(lo_hw, lo_raw + np.radians(1.0))
                hi = min(hi_hw, hi_raw - np.radians(1.0))
                if hi - lo < MIN_SPAN:
                    mid = 0.5 * (lo + hi); lo = mid - 0.5 * MIN_SPAN; hi = mid + 0.5 * MIN_SPAN
                if (cur >= hi - np.radians(1.0) and d_sim[k] > 0) or (cur <= lo + np.radians(1.0) and d_sim[k] < 0):
                    d_sim[k] = -d_sim[k]
                tgt = float(np.clip(cur + float(d_sim[k]), lo, hi))
                if tgt <= lo + 1e-6 or tgt >= hi - 1e-6:
                    hit_limit[i] += 1
                self.dc.set_dof_position_target(h, tgt)

        # 시뮬 스텝
        for i in range(self.num):
            for h in self.joints[i]:
                self.dc.set_dof_velocity_target(h, self.drive_velocity)
        for _ in range(CONTROL_REPEAT):
            self.world.step(False); self._sim_app.update()
        for i in range(self.num):
            for h in self.joints[i]:
                self.dc.set_dof_velocity_target(h, 0.0)

        # 추적광 유지
        for i in range(self.num):
            tb = self.tracking_base[i]; assert tb is not None
            lx0, ly0, lz0 = tb
            fx, fy, fz = self.hand_fixed[i]; assert fx is not None
            dx, dy = fx - lx0, fy - ly0
            dxy = math.hypot(dx, dy)
            if dxy > self.max_xy_from_hand:
                nx, ny = dx/dxy, dy/dxy
                nlx = fx - nx * self.max_xy_from_hand
                nly = fy - ny * self.max_xy_from_hand
            else:
                nlx, nly = lx0, ly0
            pitch = aim_pitch_deg((nlx, nly, lz0), (fx, fy, fz))
            self._set_tracking_pose(i, nlx, nly, lz0, pitch)

        # 보상/종료
        rewards = np.zeros(self.num, dtype=np.float32)
        successes = np.zeros(self.num, dtype=bool)
        plateaus = np.zeros(self.num, dtype=bool)

        for i in range(self.num):
            jp = np.array([self.dc.get_dof_position(h) for h in self.joints[i]], dtype=np.float32)
            rx, ry, rz = self._get_robot_light_pos(i)
            fx, fy, fz = self.hand_fixed[i]; assert fx is not None
            plane_z = self._get_plane_center(i)[2]

            # 그림자
            shadow = 0.0
            if USE_SHADOW_REWARD and (rz > plane_z + 1e-3) and (fz > plane_z + 1e-3):
                if raycast_closest_hit is None and raycast_any_hit is None:
                    shadow = self._shadow_ratio_point_light(i)
                else:
                    caster = cast(Callable[..., Dict[str, Any]], raycast_closest_hit or raycast_any_hit)
                    if caster is None:
                        shadow = self._shadow_ratio_point_light(i)
                    else:
                        shadow = self._shadow_ratio_raycast_with(caster, i, grid=48)

            # best/plateau 갱신
            if shadow < self._best_shadow[i] - MIN_IMPROVE_DELTA:
                self._best_shadow[i] = shadow; self._noimp[i] = 0
            else:
                self._noimp[i] += 1

            align, dist = self._forward_alignment(i)

            # potential + 규제
            phi = self._potential(dist, align, shadow)
            dphi = 0.0 if np.isnan(self._phi_prev[i]) else (self._phi_prev[i] - phi)
            self._phi_prev[i] = phi

            edge_cnt = int(np.sum((jp <= (self.sim_safe_lo[i] + np.radians(1.0))) |
                                  (jp >= (self.sim_safe_hi[i] - np.radians(1.0)))))
            u_l2 = float(np.sum(np.square(action[i])))
            rew = 3.0 * float(dphi) - 0.3 * edge_cnt - 0.01 * u_l2

            prev = self.prev_q[i]  # type: Optional[np.ndarray]
            if isinstance(prev, np.ndarray):
                dq = jp - prev
                rew += -0.1 * float(np.sum(dq * dq))
            self.prev_q[i] = jp.copy()

            rew += -0.1 * max(0.0, -align)
            rewards[i] = rew

            success_ready = (self._step >= MIN_STEPS_BEFORE_SUCCESS)
            successes[i] = success_ready and (self._best_shadow[i] <= SUCCESS_SHADOW)
            plateaus[i] = (self._noimp[i] >= NO_IMPROVE_PATIENCE)

        self._step += 1

        terminated = bool(np.all(successes))
        truncated = bool((self._step >= self.episode_steps) or np.all(plateaus))

        reward = float(np.mean(rewards))
        obs = self._get_obs_all()
        info = {
            "shadow_ratio_mean": float(np.mean(self._best_shadow)),
            "success_frac": float(np.mean(successes.astype(np.float32))),
            "plateau_frac": float(np.mean(plateaus.astype(np.float32))),
            "step": int(self._step),
        }
        return obs, reward, terminated, truncated, info

    # ---------- 관측/유틸 ----------
    def _get_obs_all(self):
        obs_all = []
        for i in range(self.num):
            jp = np.array([self.dc.get_dof_position(h) for h in self.joints[i]], dtype=np.float32)
            hf = self.hand_fixed[i]; assert hf is not None
            hx, hy, hz = hf
            rx, ry, rz = self._get_robot_light_pos(i)
            vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
            dist = float(math.sqrt(vx*vx + vy*vy + vz*vz))
            obs_all.append(np.array([*jp, hx, hy, hz, rx, ry, rz, vx, vy, vz, dist], dtype=np.float32))
        return np.concatenate(obs_all, axis=0)

    def _forward_alignment(self, i: int):
        fwd = self._get_robot_light_forward(i)
        (vx, vy, vz), d = self._get_vec_robot_to_hand(i)
        align = float(fwd[0]*vx + fwd[1]*vy + fwd[2]*vz)
        return align, float(d)

    def _potential(self, d, align, shadow):
        w_d, w_a, w_s = 0.5, 0.3, 1.0
        d_norm = float(np.clip(d / 0.4, 0.0, 1.0))
        return (w_d * d_norm) + (w_a * 0.5 * (1.0 - align)) + (w_s * float(np.clip(shadow, 0.0, 1.0)))

    # ----- per-actor prim helpers -----
    def _prim_path(self, i: int, base: str) -> str:
        return _ns(self.names[i], i, base)

    def _get_plane_center(self, i: int):
        prim = self.scene.UsdGeom.Xformable(self.scene.plane_prim).GetPrim().GetStage().GetPrimAtPath(self._prim_path(i, PLANE_PATH))
        p = prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])

    def _get_hand_pos(self, i: int):
        prim = self.scene.UsdGeom.Xformable(self.scene.hand_prim).GetPrim().GetStage().GetPrimAtPath(self._prim_path(i, HAND_PATH))
        p = prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])

    def _set_hand_pos(self, i: int, x, y, z):
        from pxr import UsdGeom
        prim = UsdGeom.Xformable(self.scene.hand_prim).GetPrim().GetStage().GetPrimAtPath(self._prim_path(i, HAND_PATH))
        tx = UsdGeom.Xformable(prim)
        ops = tx.GetOrderedXformOps()
        trn = next((op for op in ops if op.GetOpName()=="xformOp:translate"), None) or tx.AddTranslateOp()
        tx.SetXformOpOrder([trn] + [op for op in ops if op is not trn])
        trn.Set(self.scene.Gf.Vec3f(float(x), float(y), float(z)))

    def _get_tracking_base(self, i: int) -> Tuple[float, float, float]:
        prim = self.scene.UsdGeom.Xformable(self.scene.track_prim).GetPrim().GetStage().GetPrimAtPath(self._prim_path(i, TRACKING_LIGHT_PATH))
        p = prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])

    def _get_robot_light_pos(self, i: int):
        self.scene._xf_cache.Clear()
        prim = self.scene.robot_light_prim.GetStage().GetPrimAtPath(self._prim_path(i, ROBOT_LIGHT_PATH))
        xf = self.scene._xf_cache.GetLocalToWorldTransform(prim)
        p = xf.ExtractTranslation()
        return float(p[0]), float(p[1]), float(p[2])

    def _set_tracking_pose(self, i: int, x, y, z, pitch_deg):
        from pxr import UsdGeom
        prim = self.scene.track_prim.GetStage().GetPrimAtPath(self._prim_path(i, TRACKING_LIGHT_PATH))
        tx = UsdGeom.Xformable(prim)
        ops = tx.GetOrderedXformOps()
        rot = next((op for op in ops if op.GetOpName()=="xformOp:rotateXYZ"), None) or tx.AddRotateXYZOp()
        trn = next((op for op in ops if op.GetOpName()=="xformOp:translate"), None) or tx.AddTranslateOp()
        tx.SetXformOpOrder([trn, rot])
        trn.Set(self.scene.Gf.Vec3f(float(x), float(y), float(z)))
        rot.Set(self.scene.Gf.Vec3f(float(pitch_deg), 0.0, 0.0))

    def _get_robot_light_forward(self, i: int):
        self.scene._xf_cache.Clear()
        prim = self.scene.robot_light_prim.GetStage().GetPrimAtPath(self._prim_path(i, ROBOT_LIGHT_PATH))
        xf = self.scene._xf_cache.GetLocalToWorldTransform(prim)
        v = xf.TransformDir(self.scene.Gf.Vec3f(0.0, 0.0, -1.0))
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        n = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
        return vx/n, vy/n, vz/n

    def _get_vec_robot_to_hand(self, i: int):
        hf = self.hand_fixed[i]; assert hf is not None
        hx, hy, hz = hf
        rx, ry, rz = self._get_robot_light_pos(i)
        vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
        d = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
        return (vx/d, vy/d, vz/d), d

    # ---- 레이캐스트: caster를 인자로 받아 호출(타입 고정) ----
    def _shadow_ratio_raycast_with(
        self, caster: Callable[..., Dict[str, Any]], i: int, grid: int = 48, only_hand: bool = True
    ) -> float:
        Lx, Ly, Lz = self._get_robot_light_pos(i)
        plane_cx, plane_cy, plane_cz = self._get_plane_center(i)
        xs = np.linspace(-1.0, 1.0, grid)
        ys = np.linspace(-1.0, 1.0, grid)
        hit = total = 0
        hand_root = self._prim_path(i, HAND_PATH)

        for u in xs:
            for v in ys:
                if u*u + v*v > 1.0:
                    continue
                px = plane_cx + u * float(ROI_RADIUS)
                py = plane_cy + v * float(ROI_RADIUS)
                pz = plane_cz
                total += 1
                result = caster(start_position=(Lx, Ly, Lz), end_position=(px, py, pz))
                if not result:
                    continue
                collider_path = str(result.get("collider", ""))
                if only_hand:
                    if collider_path.startswith(hand_root):
                        hit += 1
                else:
                    hit += 1
        return float(hit / total) if total > 0 else 0.0

    def _shadow_ratio_point_light(self, i: int):
        """레이캐스트 불가 시 간단 근사."""
        Lx, Ly, Lz = self._get_robot_light_pos(i)
        fx, fy, fz = self.hand_fixed[i]; assert fx is not None
        plane_z = self._get_plane_center(i)[2]
        s_den = (fz - Lz)
        if abs(s_den) < 1e-6:
            return 0.0
        s = (plane_z - Lz) / s_den
        if s <= 0.0:
            return 0.0
        R = max(0.0, s * float(HAND_RADIUS))
        return float(min(1.0, (R / (ROI_RADIUS + 1e-6))**2))

    def _find_articulation(self, stage, hint_path, PhysxSchema):
        prim = stage.GetPrimAtPath(hint_path)
        if prim and prim.IsValid():
            return self.dc.get_articulation(hint_path)
        for p in stage.Traverse():
            try:
                if p.HasAPI(PhysxSchema.PhysxArticulationRootAPI):
                    return self.dc.get_articulation(p.GetPath().pathString)
            except Exception:
                continue
        return None
