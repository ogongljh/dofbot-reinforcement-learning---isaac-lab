import math
import numpy as np

def deg2rad(arr): return np.radians(np.asarray(arr, dtype=np.float32))
def clamp(v, lo, hi): return float(np.minimum(np.maximum(v, lo), hi))

def polar_sample(cx, cy, r_min, r_max, rng):
    r = float(rng.uniform(r_min, r_max))
    th = float(rng.uniform(0.0, 2*math.pi))
    return cx + r*math.cos(th), cy + r*math.sin(th)

def aim_pitch_deg(src_xyz, dst_xyz):
    # Returns rotation about X axis (pitch) in degrees so the light points to dst
    sx, sy, sz = src_xyz
    dx, dy, dz = dst_xyz 
    flat = math.sqrt((dx - sx)**2 + (dy - sy)**2)
    if flat < 1e-6:
        return 0.0
    # Negative Z is "down" in typical stage setups -> use atan2(-dz, flat)
    return math.degrees(math.atan2(-(dz - sz), flat))
