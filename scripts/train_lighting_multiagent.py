#!/usr/bin/env python3
import argparse, os, sys, time, random

# ── 프로젝트 소스 경로 ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "source")
if SRC not in sys.path:
    sys.path.append(SRC)

# ── RL / VecEnv / Callbacks ──────────────────────────────────────────────
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# ── Isaac Sim 런처 ───────────────────────────────────────────────────────
try:
    from isaacsim.simulation_app import SimulationApp
except Exception:
    from omni.isaac.kit import SimulationApp  # fallback

# ── 우리 환경 ────────────────────────────────────────────────────────────
from isaaclab_tasks.lighting import LightingEnvMulti

# ── 선택: W&B ───────────────────────────────────────────────────────────
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except Exception:
    wandb = None
    WandbCallback = None


def _make_seed(s: int | None) -> int:
    if s is None or s < 0:
        return (int(time.time() * 1e6) ^ os.getpid() ^ random.getrandbits(31)) % (2**31 - 1)
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_path",      type=str, required=True)
    ap.add_argument("--num_actors",      type=int, default=4, help="한 스테이지에 배치할 복제 로봇 수")
    ap.add_argument("--episode_steps",   type=int, default=400)
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--save_path",       type=str, default="./models_multi")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--headless",        action="store_true")

    # W&B 옵션(원하면 사용, 미사용 시 생략 가능)
    ap.add_argument("--wandb_project",   type=str, default=None, help="W&B 프로젝트명 (None이면 비활성)")
    ap.add_argument("--wandb_entity",    type=str, default=None)
    ap.add_argument("--wandb_run",       type=str, default=None)
    ap.add_argument("--wandb_mode",      type=str, default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    seed = _make_seed(args.seed)
    print(f"[SEED] Using seed={seed}")

    # ── Isaac App ────────────────────────────────────────────────────────
    sim_app = SimulationApp({"headless": args.headless})

    # ── 환경 팩토리 ──────────────────────────────────────────────────────
    def env_fn():
        env = LightingEnvMulti(
            world_path=args.world_path,
            sim_app=sim_app,
            num_actors=args.num_actors,
            episode_steps=args.episode_steps,
        )
        return Monitor(env)

    vec = DummyVecEnv([env_fn])
    vec.seed(seed)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── 콜백들 ───────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=args.save_path, name_prefix="sac_multi")

    cb_list = [ckpt_cb]

    # 선택: W&B 연결
    if args.wandb_project and wandb is not None and WandbCallback is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if wandb.run is not None:
            wandb.finish()
        run_name = args.wandb_run or f"sac_multi_{int(time.time())}"
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
                "num_actors": args.num_actors,
            },
        )
        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_freq=10_000,
            model_save_path=os.path.join(args.save_path, "wandb"),
            log="all",
            verbose=1,
        )
        cb_list.append(wb_cb)
    else:
        if args.wandb_project and wandb is None:
            print("[WARN] wandb 미설치. pip install wandb 후 사용 가능.")

    callbacks = CallbackList(cb_list)

    # ── 모델 ─────────────────────────────────────────────────────────────
    model = SAC(
        policy="MlpPolicy",
        env=vec,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=5_000,
        use_sde=True,
        use_sde_at_warmup=True,
    )

    print("\n=== Training start (multi-robots in a single stage) ===")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # ── 저장 ─────────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_path, "sac_multi_final")
    model.save(final_path)
    vec.save(os.path.join(args.save_path, "vecnorm.pkl"))
    print(f"=== Done. Model saved at {final_path}.zip")

    # 종료
    if args.wandb_project and wandb is not None:
        wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
