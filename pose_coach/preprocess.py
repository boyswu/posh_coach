# Author: wujiahang
import numpy as np

L_HIP, R_HIP = 12, 9
L_SHO, R_SHO = 5, 2


def normalize_skeleton(seq):
    out = []
    for f in seq:
        if np.isnan(f).all():
            out.append(f);
            continue
        hips = np.nanmean(f[[L_HIP, R_HIP], :2], axis=0)
        f2 = f.copy()
        f2[:, :2] -= hips
        shoulder_w = np.linalg.norm(f2[L_SHO, :2] - f2[R_SHO, :2])
        scale = shoulder_w if (shoulder_w and shoulder_w > 1e-6) else 1.0
        f2[:, :2] /= scale
        out.append(f2)
    return np.array(out, dtype=np.float32)
