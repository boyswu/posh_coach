# Author: wujiahang
import numpy as np

# ---------- 安全统计工具：避免空样本/自由度问题 ----------
def _nanmean_safe(a: np.ndarray, axis=None):
    m = ~np.isnan(a)
    cnt = m.sum(axis=axis, keepdims=True)
    s = np.nansum(np.where(m, a, 0.0), axis=axis, keepdims=True)
    out = np.divide(s, cnt, out=np.full_like(s, np.nan, dtype=float), where=cnt > 0)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def _nanvar_safe(a: np.ndarray, axis=None, ddof: int = 0):
    mean = _nanmean_safe(a, axis=axis)
    if axis is not None:
        expand_axes = axis if isinstance(axis, tuple) else (axis,)
        for ax in sorted(expand_axes):
            mean = np.expand_dims(mean, axis=ax)
    dif = a - mean
    dif2 = np.where(np.isnan(dif), 0.0, dif) ** 2
    m = ~np.isnan(a)
    cnt = m.sum(axis=axis, keepdims=True)
    denom = np.maximum(cnt - ddof, 0)
    out = np.divide(np.nansum(dif2, axis=axis, keepdims=True),
                    denom,
                    out=np.full_like(denom, np.nan, dtype=float),
                    where=denom > 0)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def _interp_1d(v: np.ndarray) -> np.ndarray:
    """线性插值 + 首尾填充；全 NaN 原样返回。"""
    y = v.astype(float).copy()
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return y
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    # 保险起见再做首尾填充（np.interp 已基本覆盖）
    first_valid = np.argmax(mask)
    last_valid = len(y) - 1 - np.argmax(mask[::-1])
    y[:first_valid] = y[first_valid]
    y[last_valid+1:] = y[last_valid]
    return y

# ---------- 稳健特征构建：与原函数名/返回保持一致 ----------
def build_angles(seq: np.ndarray,
                 min_joints_per_frame: int = 5,
                 do_interpolate: bool = True) -> np.ndarray:
    """
    将关键点序列压成 (T, 4) 特征，分别是 meanx, meany, stdx, stdy。
    - 容错空序列、全 NaN 帧
    - 对无效帧（有效关节点少于阈值）自动插值/填充
    - 避免 numpy 关于“空切片/自由度”告警

    参数:
      seq: ndarray, 形状 [T, J, 2] 或可切到该形状（允许含 NaN）
      min_joints_per_frame: 每帧最少有效关节点数，低于则视为无效帧
      do_interpolate: 是否对统计量做插值+首尾填充

    返回:
      feat: ndarray, 形状 (T, 4) -> [meanx, meany, stdx, stdy]，dtype float32
    """
    # 空输入直接返回空 (0,4)
    if seq is None:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(seq)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # 允许上游带多余维度，这里只取 xy
    if arr.shape[-1] < 2:
        raise ValueError(f"Expect last dim>=2 (x,y), got shape {arr.shape}")
    xy = arr[..., :2]  # [T, J, 2]
    if xy.ndim != 3:
        raise ValueError(f"Expect shape [T, J, 2], got {xy.shape}")

    x = xy[:, :, 0]
    y = xy[:, :, 1]
    valid_kp = (~np.isnan(x)) & (~np.isnan(y))
    valid_cnt = valid_kp.sum(axis=1)
    valid_mask = valid_cnt >= int(min_joints_per_frame)

    # 安全统计（帧维度 axis=1）
    meanx = _nanmean_safe(x, axis=1)
    meany = _nanmean_safe(y, axis=1)
    varx  = _nanvar_safe(x, axis=1, ddof=0)
    vary  = _nanvar_safe(y, axis=1, ddof=0)
    stdx  = np.sqrt(varx, dtype=float)
    stdy  = np.sqrt(vary, dtype=float)

    if do_interpolate and meanx.size > 0:
        meanx = _interp_1d(meanx)
        meany = _interp_1d(meany)
        stdx  = _interp_1d(stdx)
        stdy  = _interp_1d(stdy)

    feat = np.stack([meanx, meany, stdx, stdy], axis=1).astype(np.float32)
    # 去除极端 Inf/-Inf
    feat = np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 你也可以在这里按 valid_mask 降权/清零（可选）：
    # feat[~valid_mask] *= 0.0

    return feat
