#!/usr/bin/env python3
import argparse, os, math, numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# ===== Isaac Sim GUI App =====
from omni.isaac.kit import SimulationApp

def make_env(usd_path, sim_app, episode_steps=400):
    class LightingEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 30}

        def __init__(self):
            # --- action/obs spaces ---
            # 4 DOFs: small delta positions per step
            self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(4,), dtype=np.float32)
            # obs: 4 joint pos + 3 (hand xyz) + 3 (light xyz) + 1 (alignment)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

            # Isaac Sim modules (must run under python.sh)
            from omni.isaac.core.utils.prims import get_prim_at_path
            from omni.isaac.core.utils.stage import open_stage
            from omni.isaac.dynamic_control import _dynamic_control
            from omni.isaac.core import World

            self.get_prim = get_prim_at_path
            self.open_stage = open_stage
            self.dc = _dynamic_control.acquire_dynamic_control_interface()
            self.World = World
            self.usd_path = usd_path

            # Open stage and one frame update to settle
            self.open_stage(self.usd_path)
            sim_app.update()

            # World instance and reset once
            self.world = self.World()
            self.world.reset()
            sim_app.update()

            # prim handles
            self.plane = self.get_prim("/World/respone_area")
            self.hand  = self.get_prim("/World/fake_hand")
            self.light = self.get_prim("/World/another_dof")
            self.art = self.dc.get_articulation("/World/dofbot")
            if self.art is None:
                raise RuntimeError("❌ Articulation /World/dofbot not found.")
            self.joints = []
            for n in ["joint1", "joint2", "joint3", "joint4"]:
                j = self.dc.find_articulation_dof(self.art, n)
                if j == _invalid_handle():
                    raise RuntimeError(f"❌ DOF '{n}' not found on /World/dofbot.")
                self.joints.append(j)

            # episode control
            self.episode_steps = episode_steps
            self._step_count = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0

            # (optional) small randomization at reset:
            # keep current joint positions but you could set targets here if needed.

            obs = self._get_obs()
            info = {}
            return obs, info

        def step(self, action):
            # apply delta target to 4 joints
            for h, delta in zip(self.joints, action):
                cur = self.dc.get_dof_position(h)
                self.dc.set_dof_position_target(h, cur + float(delta))

            # one physics step + redraw
            self.world.step(render=False)
            sim_app.update()

            obs = self._get_obs()
            reward = float(obs[-1])  # alignment term used as a placeholder reward

            self._step_count += 1
            terminated = False
            truncated = self._step_count >= self.episode_steps
            info = {}

            return obs, reward, terminated, truncated, info

        def _get_obs(self):
            # read xform translate for hand and light
            # note: requires the prims to have xformOp:translate
            hand_t  = self.hand.GetAttribute("xformOp:translate").Get()
            light_t = self.light.GetAttribute("xformOp:translate").Get()
            hand = np.array(hand_t,  dtype=np.float32)
            light = np.array(light_t, dtype=np.float32)

            # simple alignment heuristic
            v = hand - light
            flat = np.linalg.norm(v[:2]) + 1e-6
            align = math.cos(math.atan2(-float(v[2]), float(flat)))

            # joint positions
            jp = [self.dc.get_dof_position(h) for h in self.joints]
            jp = np.array(jp, dtype=np.float32)

            obs = np.concatenate([jp, hand, light, np.array([align], dtype=np.float32)], axis=0)
            return obs

        def render(self):
            # GUI is driven by sim_app.update() already
            pass

    # helper for invalid handle comparison
    def _invalid_handle():
        # omni.isaac.dynamic_control returns 0 as invalid handle sentinel
        return 0

    return LightingEnv()

def main():
    # 1) Start Isaac Sim GUI
    sim_app = SimulationApp({"headless": False})

    # 2) Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_path",      type=str, required=True, help="USD file path")
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--save_path",       type=str, default="./models")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--episode_steps",   type=int, default=400)
    args, _ = parser.parse_known_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 3) Env / Model / Callback
    env = make_env(args.world_path, sim_app, episode_steps=args.episode_steps)
    cb  = CheckpointCallback(save_freq=10_000, save_path=args.save_path, name_prefix="sac")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./tb",
    )

    # 4) Train
    print("\n=== Training start ===")
    model.learn(total_timesteps=args.total_timesteps, callback=cb)
    model.save(os.path.join(args.save_path, "sac_final"))
    print(f"=== Done. Model saved to {os.path.join(args.save_path, 'sac_final')}.zip")

    # 5) Close Isaac Sim
    sim_app.close()

if __name__ == "__main__":
    main()
