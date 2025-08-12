import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# 이 스크립트는 Isaac Sim 환경 내에서 실행되는 것을 전제로 합니다.
# 따라서 lighting_env를 직접 임포트할 수 있습니다.
from lighting_env import LightingEnv, LightingEnvCfg

def main():
    parser = argparse.ArgumentParser()
    # 참고: Isaac Sim이 스크립트를 실행할 때 내부적으로 사용하는 인자가 있을 수 있으므로,
    #       알 수 없는 인자는 무시하도록 parse_known_args()를 사용합니다.
    args, _ = parser.parse_known_args()

    # 환경 생성
    # world_path는 이제 cfg에서 직접 설정되므로 인자에서 받을 필요가 없습니다.
    cfg = LightingEnvCfg(world_path="/home/user/Desktop/dofbot_rl.usd")
    env = LightingEnv(cfg, render_mode="human") # UI 모드이므로 'human'으로 변경

    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models",
        name_prefix="sac_model"
    )

    # SAC 모델 생성
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=42,
        tensorboard_log="./tb"
    )

    print("\n----------------------------------------")
    print("Starting training...")
    print("----------------------------------------\n")

    # 기본 타임스텝으로 학습
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback
    )

    # 최종 모델 저장
    model.save("./models/sac_model_final")
    print("\n----------------------------------------")
    print(f"Training finished. Final model saved to ./models/sac_model_final")
    print("----------------------------------------\n")

if __name__ == "__main__":
    main()
    # 스크립트가 끝나면 시뮬레이터는 자동으로 닫힙니다.