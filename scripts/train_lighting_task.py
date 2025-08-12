#!/usr/bin/env python3
import argparse, os, time, random, sys
from typing import Dict, Any

# ── 프로젝트 소스 경로 추가 ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "source")
if SRC not in sys.path:
    sys.path.append(SRC)

# ── RL / VecEnv / Callbacks ──────────────────────────────────────────────
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor  # ✅ 에피소드 통계용

# ── Isaac Sim 런처 ───────────────────────────────────────────────────────
try:
    from isaacsim.simulation_app import SimulationApp
except Exception:
    from omni.isaac.kit import SimulationApp  # fallback

# ── 우리 환경 ────────────────────────────────────────────────────────────
from isaaclab_tasks.lighting.lighting_env import LightingEnv
# ── W&B ──────────────────────────────────────────────────────────────────
import wandb
from wandb.integration.sb3 import WandbCallback


def _make_seed(s: int | None) -> int:
    if s is None or s < 0:
        return (int(time.time() * 1e6) ^ os.getpid() ^ random.getrandbits(31)) % (2**31 - 1)
    return int(s)


class InfoLoggerCallback(BaseCallback):
    def __init__(self, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._last = 0
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last) < self.log_freq:
            return True
        self._last = self.num_timesteps
        infos = self.locals.get("infos", None)
        if not infos or len(infos) == 0 or not isinstance(infos[0], dict):
            return True
        info0: Dict[str, Any] = infos[0]
        payload = {}
        for k in ("dist", "align", "hit_limit", "step", "reward"):
            v = info0.get(k, None)
            if isinstance(v, (int, float)):
                payload[k] = float(v)
        if payload:
            wandb.log(payload, step=self.num_timesteps)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_path",      type=str, required=True)
    ap.add_argument("--total_timesteps", type=int, default=50_000)
    ap.add_argument("--save_path",       type=str, default="./models")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--episode_steps",   type=int, default=400)
    ap.add_argument("--no_curriculum",   action="store_true")
    ap.add_argument("--headless",        action="store_true")
    # wandb
    ap.add_argument("--wandb_project",   type=str, default="dofbot-lighting-rl")
    ap.add_argument("--wandb_entity",    type=str, default=None)
    ap.add_argument("--wandb_run",       type=str, default=None)
    ap.add_argument("--wandb_mode",      type=str, default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    seed = _make_seed(args.seed)
    print(f"[SEED] Using seed={seed}")

    sim_app = SimulationApp({
        "headless": args.headless,
        "enable_extension": [
            "omni.kit.window.stage",
            "omni.kit.window.file",
            "omni.kit.window.property",
            "omni.kit.widget.stage",
            "omni.isaac.ui",
            "omni.isaac.dynamic_control",
        ],
    })

    os.environ["WANDB_MODE"] = args.wandb_mode
    run_name = args.wandb_run or f"sac_{int(time.time())}"
    if wandb.run is not None:
        wandb.finish()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "algo": "SAC",
            "total_timesteps": args.total_timesteps,
            "episode_steps": args.episode_steps,
            "seed": seed,
            "headless": args.headless,
        },
    )

    def env_fn():
        env = LightingEnv(
            world_path=args.world_path,
            sim_app=sim_app,
            episode_steps=args.episode_steps,
            use_curriculum=(not args.no_curriculum),
        )
        env = Monitor(env)
        return env

    vec = DummyVecEnv([env_fn])
    vec.seed(seed)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=args.save_path, name_prefix="sac")
    wb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_freq=10_000,
        model_save_path=os.path.join(args.save_path, "wandb"),
        log="all",
        verbose=1,
    )
    info_cb = InfoLoggerCallback(log_freq=250, verbose=0)
    callbacks = CallbackList([ckpt_cb, wb_cb, info_cb])

    model = SAC(
        policy="MlpPolicy",
        env=vec,
        verbose=1,
        seed=seed,
        tensorboard_log=None,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=500_000,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=2_000,        # 탐색은 알고리즘에 맡김
        use_sde=True,                 # 궤적 반복 줄이기
        use_sde_at_warmup=True,
    )

    print("\n=== Training start (wandb) ===")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    final_path = os.path.join(args.save_path, "sac_final")
    model.save(final_path)
    vec.save(os.path.join(args.save_path, "vecnorm.pkl"))
    print(f"=== Done. Model saved at {final_path}.zip")

    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
