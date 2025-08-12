import math
class LightingScene:
    """Stage/Prim 핸들 및 좌표 유틸."""
    def __init__(self, sim_app, world_path, plane_path, hand_path, tracking_light_path, robot_light_path):
        # SimulationApp은 이미 밖에서 생성돼 있어야 함
        self._sim_app = sim_app

        # 🔻 pxr/isaac 유틸 지연 임포트 (SimulationApp 이후)
        from pxr import UsdGeom, Gf
        self.UsdGeom, self.Gf = UsdGeom, Gf
        try:
            from isaacsim.core.utils.stage import open_stage
            from isaacsim.core.utils.prims import get_prim_at_path
        except ImportError:
            from omni.isaac.core.utils.stage import open_stage
            from omni.isaac.core.utils.prims import get_prim_at_path

        # Stage 열고 1프레임 업데이트
        open_stage(world_path)
        self._sim_app.update()

        # Prim 핸들
        self.plane_prim = get_prim_at_path(plane_path)
        self.hand_prim  = get_prim_at_path(hand_path)
        self.track_prim = get_prim_at_path(tracking_light_path)
        self.robot_light_prim = get_prim_at_path(robot_light_path)

        assert self.plane_prim.IsValid(),  f"❌ Invalid plane prim: {plane_path}"
        assert self.hand_prim.IsValid(),   f"❌ Invalid hand prim: {hand_path}"
        assert self.track_prim.IsValid(),  f"❌ Invalid tracking light prim: {tracking_light_path}"
        assert self.robot_light_prim.IsValid(), f"❌ Invalid robot light prim: {robot_light_path}"

        # Xform 캐시
        self._xf_cache = self.UsdGeom.XformCache()

        # 🔒 tracking light에 필요한 XformOp(translate/rotate) 보장
        tx = self.UsdGeom.Xformable(self.track_prim)
        ops = tx.GetOrderedXformOps()
        self._rot_op = next((op for op in ops if op.GetOpName()=="xformOp:rotateXYZ"), None) or tx.AddRotateXYZOp()
        self._trn_op = next((op for op in ops if op.GetOpName()=="xformOp:translate"), None) or tx.AddTranslateOp()
        tx.SetXformOpOrder([self._trn_op, self._rot_op])

        # 🔒 hand에도 translate op 보장 (없으면 추가)
        hx = self.UsdGeom.Xformable(self.hand_prim)
        hops = hx.GetOrderedXformOps()
        self._hand_trn = next((op for op in hops if op.GetOpName()=="xformOp:translate"), None) or hx.AddTranslateOp()
        hx.SetXformOpOrder([self._hand_trn] + [op for op in hops if op is not self._hand_trn])

        # tracking light 기준점
        self._tracking_base = None
    # ---------- 기본 유틸 ----------
    def get_plane_center(self):
        p = self.plane_prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])
    def set_hand_pos(self, x, y, z):
        self._hand_trn.Set(self.Gf.Vec3f(float(x), float(y), float(z)))
    def get_hand_pos(self):
        p = self.hand_prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])
    def get_tracking_light_pos(self):
        p = self.track_prim.GetAttribute("xformOp:translate").Get()
        return float(p[0]), float(p[1]), float(p[2])
    def get_robot_light_world_pos(self):
        self._xf_cache.Clear()
        xf = self._xf_cache.GetLocalToWorldTransform(self.robot_light_prim)
        p = xf.ExtractTranslation()
        return float(p[0]), float(p[1]), float(p[2])
    def set_tracking_light_pose(self, x, y, z, pitch_deg):
        self._trn_op.Set(self.Gf.Vec3f(float(x), float(y), float(z)))
        self._rot_op.Set(self.Gf.Vec3f(float(pitch_deg), 0.0, 0.0))
    def cache_tracking_base(self):
        self._tracking_base = self.get_tracking_light_pos()
    def get_robot_light_forward(self):
        """로봇 라이트 로컬 -Z 축을 월드 방향으로 변환해 단위 벡터로 반환."""
        self._xf_cache.Clear()
        xf = self._xf_cache.GetLocalToWorldTransform(self.robot_light_prim)
        # 라이트는 통상 -Z 방향을 '앞'으로 간주
        v = xf.TransformDir(self.Gf.Vec3f(0.0, 0.0, -1.0))
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        n = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
        return vx / n, vy / n, vz / n
    def get_vec_robot_to_hand(self):
        """로봇 라이트 위치에서 fake hand까지의 (월드) 단위 방향 벡터와 거리."""
        hx, hy, hz = self.get_hand_pos()
        rx, ry, rz = self.get_robot_light_world_pos()
        vx, vy, vz = (hx - rx), (hy - ry), (hz - rz)
        d = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
        return (vx / d, vy / d, vz / d), d
    @property
    def tracking_base(self):
        return self._tracking_base
