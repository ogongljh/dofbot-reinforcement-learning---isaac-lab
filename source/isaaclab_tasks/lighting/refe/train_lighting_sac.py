#!/usr/bin/env python3
import argparse
import os

# 1) SimulationApp (UI 모드) 로드 — omni.isaac.core 를 쓸 수 있게 해 줍니다.
from omni.isaac.kit import SimulationApp

# Stable-Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    # --- 1) 시뮬레이터 실행 ---
    sim_app = SimulationApp({"headless": False})

    # --- 2) 이제야 안전하게 omni.isaac.core 모듈 import ---
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.dynamic_control import _dynamic_control
    from omni.isaac.core.utils.stage import open_stage
    from omni.isaac.core import World
    import numpy as np, math, random

    # --- 3) 간단한 LightingEnv 정의 ---
    class LightingEnv:
        def __init__(self, usd_path, render_mode="human"):
            self.usd_path = usd_path
            self._dc = _dynamic_control.acquire_dynamic_control_interface()
            self._plane = self._hand = self._light = None
            self._joints = []
        def reset(self):
            open_stage(self.usd_path)
            self._plane = get_prim_at_path("/World/respone_area")
            self._hand  = get_prim_at_path("/World/fake_hand")
            self._light = get_prim_at_path("/World/another_dof")
            art = self._dc.get_articulation("/World/dofbot")
            self._joints = [ self._dc.find_articulation_dof(art, n)
                             for n in ["joint1","joint2","joint3","joint4"] ]
            return self._get_obs()
        def step(self, action):
            for h,delta in zip(self._joints, action):
                cur = self._dc.get_dof_position(h)
                self._dc.set_dof_position_target(h, cur+delta)
            World.get_instance().step()
            obs = self._get_obs()
            reward = float(1.5 * obs[-1])
            return obs, reward, False, {}
        def _get_obs(self):
            hand = np.array(self._hand.GetAttribute("xformOp:translate").Get())
            light= np.array(self._light.GetAttribute("xformOp:translate").Get())
            v = hand - light
            align = math.cos(math.atan2(-v[2], np.linalg.norm(v[:2])+1e-6))
            jp = [ self._dc.get_dof_position(h) for h in self._joints ]
            return np.array(jp + hand.tolist() + light.tolist() + [align], np.float32)

    # --- 4) Argument parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_path",      type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--save_path",       type=str, default="./models")
    parser.add_argument("--seed",            type=int, default=42)
    args, _ = parser.parse_known_args()

    # --- 5) Env/Model/Callback 생성 ---
    env = LightingEnv(args.world_path, render_mode="human")
    cb  = CheckpointCallback(save_freq=10000,
                             save_path=args.save_path,
                             name_prefix="sac")
    model = SAC("MlpPolicy", env,
                verbose=1,
                seed=args.seed,
                tensorboard_log="./tb")

    # --- 6) 학습 ---
    print("\n--- Training start ---")
    model.learn(total_timesteps=args.total_timesteps, callback=cb)
    model.save(os.path.join(args.save_path, "sac_final"))
    print(f"--- Done. Model saved to {args.save_path}/sac_final")

    # --- 7) 시뮬레이터 종료 ---
    sim_app.close()

if __name__ == "__main__":
    main()
