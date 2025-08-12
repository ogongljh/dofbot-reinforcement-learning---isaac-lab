
import math
import random
import time
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pxr import UsdGeom, Gf
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.dynamic_control import _dynamic_control

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg

class LightingEnvCfg(DirectRLEnvCfg):
    def __init__(self, world_path: str):
        super().__init__()
        self.world_path = world_path
        self.decimation = 1
        self.episode_length_s = 10.0

class LightingEnv(DirectRLEnv):
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: LightingEnvCfg, render_mode: Optional[str] = None):
        self.render_mode = render_mode or "none"
        self._cfg = cfg
        self._plane_path = "/World/respone_area"
        self._hand_path = "/World/fake_hand"
        self._light_path = "/World/another_dof"
        self._dofbot_path = "/World/dofbot"
        self._joint_names = ["joint1", "joint2", "joint3", "joint4"]
        self._vel_target = 0.05
        self._initial_light_pos = Gf.Vec3f(0.0, 0.23968, 0.31541)
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 3 + 3 + 3 + 1,), dtype=np.float32)
        super().__init__(cfg=self._cfg)

    def _design_scene(self):
        from omni.isaac.core.utils.stage import open_stage
        open_stage(self._cfg.world_path)
        time.sleep(1.0)
        self._plane_prim = get_prim_at_path(self._plane_path)
        self._hand_prim = get_prim_at_path(self._hand_path)
        self._light_prim = get_prim_at_path(self._light_path)
        self._art = self._dc.get_articulation(self._dofbot_path)
        if self._art is None:
            raise RuntimeError("DOFBOT articulation not found at /World/dofbot")
        self._dof_handles = []
        for name in self._joint_names:
            h = self._dc.find_articulation_dof(self._art, name)
            if h == 0:
                raise RuntimeError(f"DOF handle not found for {name}")
            self._dof_handles.append(h)
        real_angles_deg = [90, 135, 45, 150]
        target_angles_rad = [
            math.radians(real_angles_deg[0] - 90),
            math.radians(-real_angles_deg[1] + 180),
            math.radians(-real_angles_deg[2]),
            math.radians(-real_angles_deg[3] + 25),
        ]
        for h, angle in zip(self._dof_handles, target_angles_rad):
            self._dc.set_dof_position_target(h, angle)
            self._dc.set_dof_velocity_target(h, self._vel_target)
        self._xform = UsdGeom.Xformable(self._light_prim)
        ops = self._xform.GetOrderedXformOps()
        rotate_ops = [op for op in ops if op.GetOpName() == "xformOp:rotateXYZ"]
        self._rotate_op = rotate_ops[0] if rotate_ops else self._xform.AddRotateXYZOp()
        translate_ops = [op for op in ops if op.GetOpName() == "xformOp:translate"]
        if translate_ops:
            self._xform.SetXformOpOrder([translate_ops[0], self._rotate_op])
        else:
            self._xform.SetXformOpOrder([self._rotate_op])
        self._time_since_reset = 0.0

    def _reset_idx(self, env_ids):
        center = self._plane_prim.GetAttribute("xformOp:translate").Get()
        x = random.uniform(center[0] - 0.2, center[0] + 0.2)
        y = random.uniform(center[1] - 0.2, center[1] + 0.2)
        z = random.uniform(0.0, 0.1)
        fake_pos = Gf.Vec3f(x, y, z)
        self._hand_prim.GetAttribute("xformOp:translate").Set(fake_pos)
        diff_xy = Gf.Vec2f(fake_pos[0] - self._initial_light_pos[0], fake_pos[1] - self._initial_light_pos[1])
        dist_xy = diff_xy.GetLength()
        if dist_xy > 0.3:
            dir_xy = diff_xy
            dir_xy.Normalize()
            new_x = fake_pos[0] - dir_xy[0] * 0.3
            new_y = fake_pos[1] - dir_xy[1] * 0.3
            light_pos = Gf.Vec3f(new_x, new_y, self._initial_light_pos[2])
        else:
            light_pos = self._initial_light_pos
        self._light_prim.GetAttribute("xformOp:translate").Set(light_pos)
        dz = fake_pos[2] - light_pos[2]
        dx = fake_pos[0] - light_pos[0]
        dy = fake_pos[1] - light_pos[1]
        flat = math.sqrt(dx * dx + dy * dy)
        pitch_deg = math.degrees(math.atan2(-dz, flat)) if flat > 1e-6 else 0.0
        self._rotate_op.Set(Gf.Vec3f(pitch_deg, 0.0, 0.0))
        self._time_since_reset = 0.0

    def _pre_physics_step(self, actions):
        acts = np.clip(actions, self.action_space.low, self.action_space.high).astype(np.float32)
        for h, delta in zip(self._dof_handles, acts):
            cur = self._dc.get_dof_position(h)
            self._dc.set_dof_position_target(h, cur + float(delta))
            self._dc.set_dof_velocity_target(h, self._vel_target)

    def _get_observations(self):
        joint_pos = [self._dc.get_dof_position(h) for h in self._dof_handles]
        hand = self._hand_prim.GetAttribute("xformOp:translate").Get()
        light = self._light_prim.GetAttribute("xformOp:translate").Get()
        link4 = get_prim_at_path("/World/dofbot/link4")
        link4_pos = link4.GetAttribute("xformOp:translate").Get() if link4.IsValid() else Gf.Vec3f(0, 0, 0)
        v = np.array([hand[0] - light[0], hand[1] - light[1], hand[2] - light[2]], dtype=np.float32)
        flat = np.linalg.norm(v[:2]) + 1e-6
        pitch_to_target = math.atan2(-v[2], flat)
        pitch_deg = self._rotate_op.GetAttr().Get()[0]
        pitch_cur = math.radians(pitch_deg)
        align_cos = math.cos(abs(pitch_to_target - pitch_cur))
        obs = np.array(
            joint_pos
            + [hand[0], hand[1], hand[2]]
            + [light[0], light[1], light[2]]
            + [link4_pos[0], link4_pos[1], link4_pos[2]]
            + [align_cos],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self):
        hand = self._hand_prim.GetAttribute("xformOp:translate").Get()
        light = self._light_prim.GetAttribute("xformOp:translate").Get()
        obs = self._get_observations()
        align_cos = float(obs[-1])
        target_under = Gf.Vec3f(hand[0], hand[1], 0.0)
        dir_vec = Gf.Vec3f(target_under[0] - light[0], target_under[1] - light[1], target_under[2] - light[2])
        link4 = get_prim_at_path("/World/dofbot/link4")
        link4_pos = link4.GetAttribute("xformOp:translate").Get() if link4.IsValid() else Gf.Vec3f(10, 10, 10)
        ray_o = np.array([light[0], light[1], light[2]], dtype=np.float32)
        ray_d = np.array([dir_vec[0], dir_vec[1], dir_vec[2]], dtype=np.float32)
        ray_d = ray_d / (np.linalg.norm(ray_d) + 1e-8)
        p = np.array([link4_pos[0], link4_pos[1], link4_pos[2]], dtype=np.float32)
        t = float(np.dot(p - ray_o, ray_d))
        closest = ray_o + t * ray_d
        dist = float(np.linalg.norm(p - closest))
        visibility = 1.0 if dist > 0.05 else -1.0
        control_penalty = 0.0 if align_cos > 0.95 else 0.01
        reward = 1.5 * align_cos + 0.7 * visibility - control_penalty
        return reward

    def _is_done(self):
        self._time_since_reset += self.step_dt
        timeout = self._time_since_reset > self.max_episode_length_s
        return bool(timeout)

    def render(self):
        return None
