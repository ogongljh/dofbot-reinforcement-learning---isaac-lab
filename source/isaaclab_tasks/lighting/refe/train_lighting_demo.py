#!/usr/bin/env python3
import argparse, os
import numpy as np, math
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# ===== Isaac Sim GUI App =====
from omni.isaac.kit import SimulationApp

def make_env(usd_path, sim_app):
    class LightingEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 30}
        def __init__(self):
            # 액션·관측 정의
            self.action_space = spaces.Box(-0.05, 0.05, (4,), np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, (4+3+3+1,), np.float32)

            # Isaac Sim 모듈 임포트 (GUI 모드 보장 필요)
            from omni.isaac.core.utils.prims import get_prim_at_path
            from omni.isaac.core.utils.stage import open_stage
            from omni.isaac.dynamic_control import _dynamic_control
            from omni.isaac.core import World

            self.get_prim = get_prim_at_path
            self.open_stage = open_stage
            self.dc = _dynamic_control.acquire_dynamic_control_interface()
            self.World = World
            self.usd_path = usd_path

            # 장면 열기 & prim 핸들링
            self.open_stage(self.usd_path)
            # prim들
            self.plane = self.get_prim("/World/respone_area")
            self.hand  = self.get_prim("/World/fake_hand")
            self.light = self.get_prim("/World/another_dof")
            art = self.dc.get_articulation("/World/dofbot")
            self.joints = [self.dc.find_articulation_dof(art, n)
                           for n in ["joint1","joint2","joint3","joint4"]]

            # 한 프레임 렌더
            sim_app.update()

        def reset(self):
            # (이미 open_stage() 되어 있으므로 단순히 첫 관측 반환)
            return self._get_obs()

        def step(self, action):
            # 조인트 목표 갱신
            for h,delta in zip(self.joints, action):
                cur = self.dc.get_dof_position(h)
                self.dc.set_dof_position_target(h, cur + float(delta))
            # 물리 한 스텝
            self.World.get_instance().step()
            # 화면 갱신
            sim_app.update()
            obs = self._get_obs()
            reward = float(obs[-1])
            done = False
            return obs, reward, done, {}

        def _get_obs(self):
            hand  = np.array(self.hand.GetAttribute("xformOp:translate").Get())
            light = np.array(self.light.GetAttribute("xformOp:translate").Get())
            v = hand - light
            flat = np.linalg.norm(v[:2]) + 1e-6
            align = math.cos(math.atan2(-v[2], flat))
            jp = [self.dc.get_dof_position(h) for h in self.joints]
            return np.array(jp + hand.tolist() + light.tolist() + [align], np.float32)

        def render(self, mode="human"):
            # sim_app.update() 로 화면이 이미 갱신됩니다.
            pass

    return LightingEnv()

def main():
    # 1) GUI 모드 Isaac Sim 실행
    sim_app = SimulationApp({"headless": False})

    # 2) 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_path",      type=str, required=True, help="USD 파일 경로")
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--save_path",       type=str, default="./models")
    parser.add_argument("--seed",            type=int, default=42)
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 3) 환경·모델·콜백 생성
    env   = make_env(args.world_path, sim_app)
    cb    = CheckpointCallback(save_freq=10_000,
                               save_path=args.save_path,
                               name_prefix="sac")
    model = SAC("MlpPolicy", env,
                verbose=1,
                seed=args.seed,
                tensorboard_log="./tb")

    # 4) 학습
    print("\n=== Training start ===")
    model.learn(total_timesteps=args.total_timesteps, callback=cb)
    model.save(os.path.join(args.save_path, "sac_final"))
    print(f"=== Done. Model saved to {args.save_path}/sac_final")

    # 5) 종료
    sim_app.close()

if __name__ == "__main__":
    main()
