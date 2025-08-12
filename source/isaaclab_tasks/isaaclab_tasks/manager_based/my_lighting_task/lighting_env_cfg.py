
import os
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

##
# Scene
##

@configclass
class MySceneCfg:
    # 여기에 USD 파일 경로를 지정합니다.
    usd_path: str = "/home/user/Desktop/dofbot_rl.usd"

##
# Environment
##

@configclass
class LightingEnvCfg(ManagerBasedRLEnvCfg):
    # scene을 위에서 정의한 MySceneCfg로 설정합니다.
    scene: MySceneCfg = MySceneCfg()

    def __post_init__(self):
        # post_init에서 부모 클래스의 설정을 수정할 수 있습니다.
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (2.0, 2.0, 2.0)
