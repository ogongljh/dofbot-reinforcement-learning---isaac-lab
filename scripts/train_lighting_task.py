#!/usr/bin/env python3
import argparse, os, time, random, sys, pathlib
from typing import Dict, Any, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "source")
if SRC not in sys.path:
    sys.path.append(SRC)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

try:
    from isaacsim.simulation_app import SimulationApp
except Exception:
    from omni.isaac.kit import SimulationApp

from isaaclab_tasks.lighting.lighting_env import LightingEnv
from isaaclab_tasks.lighting.lighting_config import (
    ROI_RADIUS, REWARD_REMOVED_K, REWARD_FINAL_K, CURRICULUM, USE_SHADOW_REWARD
)

import wandb
from wandb.integration.sb3 import WandbCallback


def _make_seed(s: int | None) -> int:
    if s is None or s < 0:
        return (int(time.time() * 1e6) ^ os.getpid() ^ random.getrandbits(31)) % (2**31 - 1)
    return int(s)

def _wandb_get(cfg, key, default):
    try: return cfg.get(key, default)
    except Exception: return default


class InfoLoggerCallback(BaseCallback):
    def __init__(self, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose); self.log_freq = int(log_freq); self._last = 0
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last) < self.log_freq: return True
        self._last = self.num_timesteps
        infos = self.locals.get("infos", None)
        if not infos or len(infos) == 0 or not isinstance(infos[0], dict): return True
        info0: Dict[str, Any] = infos[0]
        payload = {}
        for k in (
            "orig_shadow", "final_shadow", "removed_shadow",
            "best_final_shadow", "no_improve_steps",
            "dist", "align", "hit_limit", "step", "reward",
            "diag/Lr_z", "diag/Lt_z", "diag/hand_z",
            "diag/cos_inc", "diag/pointing", "diag/clearance",
            "diag/ray_calls", "diag/ray_used", "diag/ray_errors",
            "diag/ray_hand_hits", "diag/sphere_hits", "diag/ray_vs_sphere_mismatch",
            "diag/ray_ok_rate", "diag/ray_hand_hit_rate",
        ):
            v = info0.get(k, None)
            if isinstance(v, (int, float)): payload[k] = float(v)
        if payload: wandb.log(payload, step=self.num_timesteps)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_path",      type=str, required=True)
    ap.add_argument("--total_timesteps", type=int, default=100_000)
    ap.add_argument("--save_path",       type=str, default="./models")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--episode_steps",   type=int, default=400)
    ap.add_argument("--no_curriculum",   action="store_true")
    ap.add_argument("--headless",        action="store_true")

    # sweeps 대상 하이퍼(스윕 시 wandb.config가 덮어씀)
    ap.add_argument("--learning_rate",   type=float, default=3e-4)
    ap.add_argument("--batch_size",      type=int,   default=256)
    ap.add_argument("--buffer_size",     type=int,   default=500_000)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--tau",             type=float, default=0.005)
    ap.add_argument("--ent_coef",        type=str,   default="auto")
    ap.add_argument("--hidden",          type=int,   default=256)

    ap.add_argument("--wandb_project",   type=str, default="dofbot-lighting-rl")
    ap.add_argument("--wandb_entity",    type=str, default=None)
    ap.add_argument("--wandb_mode",      type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags",      type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    seed = _make_seed(args.seed)
    print(f"[SEED] Using seed={seed}")

    # 디버깅 시 레이 로그 보고 싶으면 외부에서: export RAY_VERBOSE=1
    sim_app = SimulationApp({
        "headless": args.headless,
        "enable_extension": [
            "omni.kit.window.stage",
            "omni.kit.window.file",
            "omni.kit.window.property",
            "omni.kit.widget.stage",
            "omni.isaac.ui",
            "omni.isaac.core",
            "omni.isaac.dynamic_control",
            "omni.physx",
            "omni.physx.flatcache",
            "omni.physx.rigid_shape",
            "omni.physx.scene_query",  # ← scene query
            "omni.physx.raycast",      # ← 없으면 자동 무시/미로딩
        ],
    })

    # Env 생성
    base_env = LightingEnv(
        world_path=args.world_path, sim_app=sim_app,
        episode_steps=args.episode_steps, use_curriculum=(not args.no_curriculum),
    )
    base_env = Monitor(base_env)

    sig = {
        "ray_mode": getattr(base_env.unwrapped, "_ray_mode", "na"),
        "shadow_grid": int(getattr(base_env.unwrapped, "_shadow_grid", 0)),
        "roi_radius": float(ROI_RADIUS),
        "remK": float(REWARD_REMOVED_K),
        "finK": float(REWARD_FINAL_K),
        "use_shadow_reward": bool(USE_SHADOW_REWARD),
        "curriculum": (not args.no_curriculum),
    }
    world_name = pathlib.Path(args.world_path).stem
    base_name = f"sac-{world_name}-ray={sig['ray_mode']}-grid{sig['shadow_grid']}-roi{int(sig['roi_radius']*1000)}mm-rem{sig['remK']}-fin{sig['finK']}-eps{args.episode_steps}-seed{seed}"
    if args.no_curriculum: base_name += "-nocurr"
    if args.headless: base_name += "-headless"

    os.environ["WANDB_MODE"] = args.wandb_mode
    if wandb.run is not None: wandb.finish()

    tags: List[str] = [sig["ray_mode"], f"grid{sig['shadow_grid']}", f"roi{int(sig['roi_radius']*1000)}mm"]
    if args.wandb_tags: tags.extend([t.strip() for t in args.wandb_tags.split(",") if t.strip()])

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=base_name,
        tags=tags,
        config={
            "algo": "SAC",
            "world_name": world_name,
            "total_timesteps": args.total_timesteps,
            "episode_steps": args.episode_steps,
            "seed": seed,
            "headless": args.headless,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "ent_coef": args.ent_coef,
            "hidden": args.hidden,
            "sig/ray_mode": sig["ray_mode"],
            "sig/shadow_grid": sig["shadow_grid"],
            "sig/roi_radius": sig["roi_radius"],
            "sig/remK": sig["remK"],
            "sig/finK": sig["finK"],
            "sig/use_shadow_reward": sig["use_shadow_reward"],
            "sig/curriculum": sig["curriculum"],
        },
    )

    cfg = wandb.config
    lr        = _wandb_get(cfg, "learning_rate", args.learning_rate)
    batch_sz  = int(_wandb_get(cfg, "batch_size",  args.batch_size))
    buf_sz    = int(_wandb_get(cfg, "buffer_size", args.buffer_size))
    gamma     = _wandb_get(cfg, "gamma",          args.gamma)
    tau       = _wandb_get(cfg, "tau",            args.tau)
    ent_coef  = _wandb_get(cfg, "ent_coef",       args.ent_coef)
    hidden    = int(_wandb_get(cfg, "hidden",     args.hidden))

    run.name = f"{base_name}-lr{lr:g}-bs{batch_sz}-hid{hidden}"
    try: run.save()
    except Exception: pass

    def env_fn(): return base_env
    vec = DummyVecEnv([env_fn])
    vec.seed(seed)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=args.save_path, name_prefix="sac")
    wb_cb = WandbCallback(gradient_save_freq=100, model_save_freq=10_000,
                          model_save_path=os.path.join(args.save_path, "wandb"), log="all", verbose=1)
    info_cb = InfoLoggerCallback(log_freq=250, verbose=0)
    callbacks = CallbackList([ckpt_cb, wb_cb, info_cb])

    model = SAC(
        policy="MlpPolicy", env=vec, verbose=1, seed=seed, tensorboard_log=None,
        learning_rate=lr, batch_size=batch_sz, buffer_size=buf_sz,
        tau=tau, gamma=gamma, train_freq=1, gradient_steps=1, ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=[hidden, hidden]),
        learning_starts=2_000, use_sde=True, use_sde_at_warmup=True,
    )

    print("\n=== Training start (wandb sweeps-ready) ===")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    final_path = os.path.join(args.save_path, "sac_final")
    model.save(final_path)
    vec.save(os.path.join(args.save_path, "vecnorm.pkl"))
    print(f"=== Done. Model saved at {final_path}.zip")

    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
