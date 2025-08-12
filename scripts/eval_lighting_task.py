#!/usr/bin/env python3
import argparse, os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from omni.isaac.kit import SimulationApp

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "source")
if SRC not in sys.path:
    sys.path.append(SRC)

from isaaclab_tasks.lighting import LightingEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_path", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, required=True)
    ap.add_argument("--episode_steps", type=int, default=400)
    args = ap.parse_args()

    sim_app = SimulationApp({"headless": False})

    def env_fn():
        return LightingEnv(world_path=args.world_path, sim_app=sim_app, episode_steps=args.episode_steps)

    vec = DummyVecEnv([env_fn])
    vec = VecNormalize.load(args.vecnorm_path, vec)
    vec.training = False
    vec.norm_reward = False
    if hasattr(vec, "venv"):  # 안전하게 내부 venv도 끄기
        try:
            vec.venv.training = False
            vec.venv.norm_reward = False
        except Exception:
            pass

    model = SAC.load(args.model_path, env=vec)
    obs = vec.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = vec.step(action)
        if done[0]:
            obs = vec.reset()

    sim_app.close()

if __name__ == "__main__":
    main()
