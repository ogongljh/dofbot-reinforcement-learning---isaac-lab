#!/usr/bin/env python3
import argparse, os, math, numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from omni.isaac.kit import SimulationApp

# =============================
# Env Factory
# =============================
def make_env(usd_path, sim_app, episode_steps=400, use_curriculum=True):
    class LightingEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 30}

        def __init__(self):
            # ---- ACTION / OBS ----
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            # [4 joints] + [hand(3)] + [robot_light(3)] + [vector(3)] + [dist(1)]
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 3 + 3 + 3 + 1,), dtype=np.float32)

            # ---- Isaac deps ----
            from omni.isaac.core.utils.prims import get_prim_at_path
            from omni.isaac.core.utils.stage import open_stage
            from omni.isaac.dynamic_control import _dynamic_control
            from omni.isaac.core import World
            from pxr import Gf, UsdGeom
            self.get_prim = get_prim_at_path
            self.open_stage = open_stage
            self.dc = _dynamic_control.acquire_dynamic_control_interface()
            self.World = World
            self.Gf = Gf
            self.UsdGeom = UsdGeom

            # ---- Stage/World ----
            self.open_stage(usd_path); sim_app.update()
            self.world = self.World(); self.world.reset(); sim_app.update()

            # ---- Prims ----
            self.plane_path          = "/World/respone_area"
            self.hand_path           = "/World/fake_hand"
            self.tracking_light_path = "/World/another_dof"                           # follows fake_hand
            self.robot_light_path    = "/World/dofbot/link4/Xform/RectLight"          # learning target
            self.dof_path            = "/World/dofbot"

            self.plane_prim          = self.get_prim(self.plane_path)
            self.hand_prim           = self.get_prim(self.hand_path)
            self.tracking_light_prim = self.get_prim(self.tracking_light_path)
            self.robot_light_prim    = self.get_prim(self.robot_light_path)
            if not (self.plane_prim.IsValid() and self.hand_prim.IsValid()
                    and self.tracking_light_prim.IsValid() and self.robot_light_prim.IsValid()):
                raise RuntimeError("❌ Invalid prim path(s): check plane/hand/lights.")

            # ---- Articulation ----
            self.art = self.dc.get_articulation(self.dof_path)
            if self.art is None:
                raise RuntimeError("❌ DOFBOT articulation not found at /World/dofbot")
            self.joint_names = ["joint1", "joint2", "joint3", "joint4"]
            self.joints = [self.dc.find_articulation_dof(self.art, n) for n in self.joint_names]
            if any(h == 0 for h in self.joints):
                raise RuntimeError("❌ One or more DOF handles invalid. Check joint names.")

            # HW limits from sim
            self.joint_limits = []
            for h in self.joints:
                props = self.dc.get_dof_properties(h)
                if getattr(props, "hasLimits", False):
                    lo, hi = float(props.lower), float(props.upper)
                else:
                    lo, hi = -3.14159, 3.14159
                self.joint_limits.append((lo, hi))
            self.joint_limits = np.array(self.joint_limits, dtype=np.float32)

            # ---- SAFE ANGLE RANGES (SIM RAD) — your request ----
            # joint1~4: [-30°, +30°] each
            deg_lo = np.array([-30.0, -30.0, -30.0, -30.0], dtype=np.float32)
            deg_hi = np.array([+30.0, +30.0, +30.0, +30.0], dtype=np.float32)
            self.sim_safe_lo = np.radians(deg_lo)
            self.sim_safe_hi = np.radians(deg_hi)

            # Intersect with HW limits just in case
            self.sim_safe_lo = np.maximum(self.sim_safe_lo, self.joint_limits[:,0])
            self.sim_safe_hi = np.minimum(self.sim_safe_hi, self.joint_limits[:,1])

            # ---- HOME (INITIAL) POSE — your request [0, 45, -45, -125] deg ----
            home_deg = np.array([0.0, 45.0, -45.0, -125.0], dtype=np.float32)
            self.home_pose = np.radians(home_deg)

            # ---- Control params (no correction, no direction locks) ----
            self.max_delta = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32)  # ≈1.1°/step
            self.control_repeat = 5
            self.drive_velocity = 0.3

            # ---- Lights behavior ----
            self.max_xy_from_hand = 0.18
            self._xf_cache = self.UsdGeom.XformCache()
            self.tracking_light_base = None  # set on reset()

            # ---- Episode ----
            self.episode_steps = episode_steps
            self._step_count = 0
            self.best_reward = -1e9
            self._no_improve_count = 0
            self.patience = 80
            self.success_dist = 0.10
            self.success_align = 0.90
            self.prev_q = None

            # ---- Curriculum / Spawn ----
            self.use_curriculum = use_curriculum
            self.episode_idx = 0
            self.spawn_radius = 0.12
            self.hand_z_min, self.hand_z_max = 0.02, 0.08

        # ---------- helpers ----------
        def _move_to_home(self, max_steps=300, tol=1e-3):
            # clamp home into safe
            q_home = np.clip(self.home_pose, self.sim_safe_lo, self.sim_safe_hi)
            for h, q in zip(self.joints, q_home):
                self.dc.set_dof_position_target(h, float(q))
                self.dc.set_dof_velocity_target(h, self.drive_velocity)
            for _ in range(max_steps):
                self.world.step(render=False); sim_app.update()
                cur = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
                if np.max(np.abs(cur - q_home)) < tol:
                    break
            for h in self.joints:
                self.dc.set_dof_velocity_target(h, 0.0)

        def _apply_curriculum(self):
            if not self.use_curriculum:
                return
            if   self.episode_idx < 200: self.spawn_radius = 0.12
            elif self.episode_idx < 600: self.spawn_radius = 0.16
            else:                        self.spawn_radius = 0.20

        def _world_pos(self, prim):
            xf = self._xf_cache.GetLocalToWorldTransform(prim)
            p = xf.ExtractTranslation()
            return self.Gf.Vec3f(p[0], p[1], p[2])

        # ---------- Gym API ----------
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0
            self.best_reward = -1e-9
            self._no_improve_count = 0
            self._apply_curriculum()

            self._move_to_home(max_steps=300, tol=1e-3)

            # cache tracking light base position (world)
            self.tracking_light_base = self._get_tracking_light_pos()

            # spawn fake_hand near plane center using Gym RNG
            center = self.plane_prim.GetAttribute("xformOp:translate").Get()
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            rx = float(self.np_random.uniform(cx - self.spawn_radius, cx + self.spawn_radius))
            ry = float(self.np_random.uniform(cy - self.spawn_radius, cy + self.spawn_radius))
            rz = float(self.np_random.uniform(self.hand_z_min, self.hand_z_max)) + cz
            fake_pos = self.Gf.Vec3f(rx, ry, rz)
            self.hand_prim.GetAttribute("xformOp:translate").Set(fake_pos)

            # aim the tracking light (another_dof) to fake_hand
            self._update_light_pose_aim(fake_pos)

            for _ in range(3):
                self.world.step(render=False); sim_app.update()

            self.prev_q = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)

            obs = self._get_obs()
            self.episode_idx += 1
            print(f"[RESET] ep={self.episode_idx} hand=({float(fake_pos[0]):.6f}, {float(fake_pos[1]):.6f}, {float(fake_pos[2]):.6f})")
            return obs, {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)

            # action -> delta rad (NO correction, NO direction lock)
            d_sim = np.clip(action, -1.0, 1.0) * self.max_delta

            hit_limit = 0
            for i, h in enumerate(self.joints):
                cur = self.dc.get_dof_position(h)
                tgt = cur + float(d_sim[i])

                # 1) clip by HW
                lo_hw, hi_hw = self.joint_limits[i]
                tgt = float(np.clip(tgt, lo_hw, hi_hw))
                # 2) clip by SAFE range (your -30~+30 deg)
                tgt = float(np.clip(tgt, self.sim_safe_lo[i], self.sim_safe_hi[i]))
                if tgt <= self.sim_safe_lo[i] + 1e-4 or tgt >= self.sim_safe_hi[i] - 1e-4:
                    hit_limit += 1

                self.dc.set_dof_position_target(h, tgt)

            # conservative velocity
            for h in self.joints:
                self.dc.set_dof_velocity_target(h, self.drive_velocity)

            # keep tracking light aimed at fake_hand
            fake_pos = self._get_hand_pos()
            self._update_light_pose_aim(fake_pos)

            for _ in range(self.control_repeat):
                self.world.step(render=False); sim_app.update()

            obs = self._get_obs()
            reward = self._compute_reward()

            # progress track (simple)
            if reward > self.best_reward + 1e-3:
                self.best_reward = reward; self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            # success / termination (robot light vs fake_hand)
            hand = np.array(self._get_hand_pos(), dtype=np.float32)
            light = np.array(self._get_robot_light_pos(), dtype=np.float32)
            v = hand - light
            dist = float(np.linalg.norm(v))
            flat = math.sqrt(v[0]**2 + v[1]**2)
            align = math.cos(math.atan2(-v[2], flat)) if flat > 1e-6 else 1.0

            self._step_count += 1
            success = (dist <= self.success_dist) and (align >= self.success_align)
            terminated = bool(success)
            truncated = (
                (self._step_count >= self.episode_steps) or
                (hit_limit >= 2) or
                (self._no_improve_count >= self.patience)
            )
            return obs, float(reward), terminated, truncated, {}

        # ---------- Light pose (tracking light only) ----------
        def _update_light_pose_aim(self, fp):
            fx, fy, fz = float(fp[0]), float(fp[1]), float(fp[2])
            base = self.tracking_light_base if self.tracking_light_base is not None else self._get_tracking_light_pos()
            lx0, ly0, lz0 = float(base[0]), float(base[1]), float(base[2])

            dx, dy = fx - lx0, fy - ly0
            dxy = math.sqrt(dx*dx + dy*dy)
            if dxy > self.max_xy_from_hand:
                nx, ny = dx / max(dxy, 1e-8), dy / max(dxy, 1e-8)
                new_x = fx - nx * self.max_xy_from_hand
                new_y = fy - ny * self.max_xy_from_hand
                light_pos = self.Gf.Vec3f(new_x, new_y, lz0)
            else:
                light_pos = self.Gf.Vec3f(lx0, ly0, lz0)

            self.tracking_light_prim.GetAttribute("xformOp:translate").Set(light_pos)

            xform = self.UsdGeom.Xformable(self.tracking_light_prim)
            ops = [op for op in xform.GetOrderedXformOps() if op.GetOpName()=="xformOp:rotateXYZ"]
            rotate_op = ops[0] if ops else xform.AddRotateXYZOp()
            t_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpName()=="xformOp:translate"]
            xform.SetXformOpOrder(([t_ops[0]] if t_ops else []) + [rotate_op])

            dz = float(fz - light_pos[2])
            flat = math.sqrt((fx - light_pos[0])**2 + (fy - light_pos[1])**2)
            angle_deg = math.degrees(math.atan2(-dz, flat)) if flat > 1e-6 else 0.0
            rotate_op.Set(self.Gf.Vec3f(angle_deg, 0.0, 0.0))

        # ---------- Pos getters ----------
        def _get_hand_pos(self):
            p = self.hand_prim.GetAttribute("xformOp:translate").Get()
            return self.Gf.Vec3f(p[0], p[1], p[2])

        def _get_tracking_light_pos(self):
            p = self.tracking_light_prim.GetAttribute("xformOp:translate").Get()
            return self.Gf.Vec3f(p[0], p[1], p[2])

        def _get_robot_light_pos(self):
            xf = self._xf_cache.GetLocalToWorldTransform(self.robot_light_prim)
            p = xf.ExtractTranslation()
            return self.Gf.Vec3f(p[0], p[1], p[2])

        # ---------- Obs / Reward (robot light used) ----------
        def _get_obs(self):
            jp = np.array([self.dc.get_dof_position(h) for h in self.joints], dtype=np.float32)
            hand  = self._get_hand_pos()
            light = self._get_robot_light_pos()   # robot light!
            hand_np  = np.array([hand[0], hand[1], hand[2]], dtype=np.float32)
            light_np = np.array([light[0], light[1], light[2]], dtype=np.float32)
            v = hand_np - light_np
            dist = np.array([np.linalg.norm(v)], dtype=np.float32)
            return np.concatenate([jp, hand_np, light_np, v, dist], axis=0)

        def _compute_reward(self):
            hand  = np.array(self._get_hand_pos(), dtype=np.float32)
            light = np.array(self._get_robot_light_pos(), dtype=np.float32)  # robot light!
            v = hand - light
            d = float(np.linalg.norm(v))
            flat = math.sqrt(v[0]**2 + v[1]**2)
            align = math.cos(math.atan2(-v[2], flat))
            bright = 1.0 / max(d*d, 0.05**2); bright = min(bright, 400.0)
            occ_pen = self._occlusion_penalty(light, hand)
            return float(1.0*align + 0.02*bright - 0.5*occ_pen)

        def _occlusion_penalty(self, light_np, hand_np):
            t_vals = np.linspace(0.1, 0.9, 5, dtype=np.float32)
            occ = 0.0; base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for t in t_vals:
                p = light_np * (1.0 - t) + hand_np * t
                dist_base = float(np.linalg.norm(p - base_pos))
                if dist_base < 0.05:
                    occ += (0.05 - dist_base) * 4.0
            return occ

        def render(self):
            pass

    return LightingEnv()

# =============================
# Main
# =============================
def main():
    sim_app = SimulationApp({
        "headless": False,
        "enable_extension": [
            "omni.kit.window.stage",
            "omni.kit.window.file",
            "omni.kit.window.property",
            "omni.kit.widget.stage",
            "omni.isaac.ui",
            "omni.isaac.dynamic_control",
        ],
    })

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_path",      type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--save_path",       type=str, default="./models")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--episode_steps",   type=int, default=400)
    parser.add_argument("--no_curriculum",   action="store_true", help="disable spawn curriculum")
    parser.add_argument("--headless",        action="store_true")
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_path, exist_ok=True)

    # build env
    env_fn = lambda: make_env(args.world_path, sim_app, episode_steps=args.episode_steps, use_curriculum=not args.no_curriculum)
    vec = DummyVecEnv([env_fn])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    cb  = CheckpointCallback(save_freq=10_000, save_path=args.save_path, name_prefix="sac")

    model = SAC(
        policy="MlpPolicy",
        env=vec,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./tb",
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=500_000,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256,256])
    )

    print("\n=== Training start ===")
    model.learn(total_timesteps=args.total_timesteps, callback=cb)
    model.save(os.path.join(args.save_path, "sac_final"))
    vec.save(os.path.join(args.save_path, "vecnorm.pkl"))
    print(f"=== Done. Model saved to {os.path.join(args.save_path, 'sac_final')}.zip")

    sim_app.close()

if __name__ == "__main__":
    main()
