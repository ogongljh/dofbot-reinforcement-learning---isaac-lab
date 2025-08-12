# ~/IsaacLab/standalone/dofbot_pose_control.py
from omni.isaac.dynamic_control import _dynamic_control
import math
import time

time.sleep(2)  # 시뮬 시작 기다림

dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation("/World/dofbot")

if art is None:
    print("❌ DOFBOT articulation을 찾을 수 없습니다.")
    exit()
else:
    print("✅ DOFBOT articulation 연결됨.")

joint_names = ["joint1", "joint2", "joint3", "joint4"]
real_angles_deg = [90, 135, 45, 150]
target_angles_rad = [
    math.radians(real_angles_deg[0] - 90),
    math.radians(-real_angles_deg[1] + 180),
    math.radians(-real_angles_deg[2]),
    math.radians(-real_angles_deg[3] + 25)
]

for name, angle in zip(joint_names, target_angles_rad):
    dof_handle = dc.find_articulation_dof(art, name)
    dc.set_dof_position_target(dof_handle, angle)
    dc.set_dof_velocity_target(dof_handle, 0.05)
    time.sleep(0.2)

print("🟢 완료")
