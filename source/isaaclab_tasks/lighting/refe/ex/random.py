from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom
import math
import random

# 경로 설정
plane_path = "/World/respone_area"
target_path = "/World/fake_hand"
light_path = "/World/another_dof"

# Prim 불러오기
plane_prim = get_prim_at_path(plane_path)
target_prim = get_prim_at_path(target_path)
light_prim = get_prim_at_path(light_path)

# 광원 초기 고정 위치 (Z 고정)
initial_light_pos = Gf.Vec3f(0.0, 0.23968, 0.31541)

if not plane_prim.IsValid() or not target_prim.IsValid() or not light_prim.IsValid():
    print("❌ Prim 경로 오류.")
else:
    # ===== [1] fake_hand 무작위 위치 설정 =====
    center = plane_prim.GetAttribute("xformOp:translate").Get()
    x = random.uniform(center[0] - 0.2, center[0] + 0.2)
    y = random.uniform(center[1] - 0.2, center[1] + 0.2)
    z = random.uniform(0.0, 0.1)
    fake_pos = Gf.Vec3f(x, y, z)
    target_prim.GetAttribute("xformOp:translate").Set(fake_pos)
    print(f"✅ fake_hand 위치: ({x:.3f}, {y:.3f}, {z:.3f})")

    # ===== [2] 광원 이동 (30cm보다 멀면 가까이 이동) =====
    diff_xy = Gf.Vec2f(fake_pos[0] - initial_light_pos[0], fake_pos[1] - initial_light_pos[1])
    dist_xy = diff_xy.GetLength()

    if dist_xy > 0.3:
        dir_xy = diff_xy
        dir_xy.Normalize()
        new_x = fake_pos[0] - dir_xy[0] * 0.3
        new_y = fake_pos[1] - dir_xy[1] * 0.3
        light_pos = Gf.Vec3f(new_x, new_y, initial_light_pos[2])
    else:
        light_pos = initial_light_pos

    light_prim.GetAttribute("xformOp:translate").Set(light_pos)
    print(f"✅ 광원 위치: ({light_pos[0]:.3f}, {light_pos[1]:.3f}, {light_pos[2]:.3f})")

    # ===== [3] 광원 X축 회전 (rotateXYZ 사용) =====
    xform = UsdGeom.Xformable(light_prim)

    # 기존 rotateXYZ 찾기
    rotate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:rotateXYZ"]
    if rotate_ops:
        rotate_op = rotate_ops[0]
    else:
        rotate_op = xform.AddRotateXYZOp()

    # translate op도 찾기
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"]
    if translate_ops:
        xform.SetXformOpOrder([translate_ops[0], rotate_op])
    else:
        xform.SetXformOpOrder([rotate_op])

    # 회전각 계산 (X축만 조정)
    dz = fake_pos[2] - light_pos[2]
    dx = fake_pos[0] - light_pos[0]
    dy = fake_pos[1] - light_pos[1]
    flat_dist = math.sqrt(dx ** 2 + dy ** 2)

    if flat_dist > 1e-6:
        angle_deg = math.degrees(math.atan2(-dz, flat_dist))  # 음수는 아래로 기울이기
    else:
        angle_deg = 0.0
    rotate_op.Set(Gf.Vec3f(angle_deg, 0.0, 0.0))
    print(f"✅ 광원 X축 회전 적용됨: {angle_deg:.2f}°")