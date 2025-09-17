# -*- coding: utf-8 -*-
# Author: wujiahang
"""
Patched scoring wrapper — fully backward compatible.

支持两种调用方式：
A) 角度序列对齐模式（你项目原来的用法）
   score = score_from_path(Ft, Fu, path, user_kps=..., tpl_kps=..., user_vis=..., tpl_vis=...)
   -> 返回 float

B) 视频路径模式（无需模板也能评分）
   score = score_from_path(tpl_video_path, usr_video_path, action='taichi', use_template=False)
   -> 返回 float
"""
import os, numpy as np, cv2
from .engine.scoring.score_engine import ScoreAggregator

# 兼容保留
# 自动检测 25点/33点 的 build_angle_series_from_kps
import numpy as np

def build_angle_series_from_kps(kps):
    """
    kps: list/array of frames, each frame shape ~ [J, C]，C>=2（x,y[,z[,vis]]）
    兼容：
      - MediaPipe Pose 33 点：0..32（项目中你已有）
      - OpenPose BODY_25  25 点：0..24
    返回:
      dict[str, np.ndarray[T]]: {"knee_L","knee_R","hip_L","hip_R","elbow_L","elbow_R"}
    """

    def _select_index_map(J: int):
        """
        返回一个索引字典:
          L_SH/R_SH: 左/右肩
          L_EL/R_EL: 左/右肘
          L_WR/R_WR: 左/右腕
          L_HIP/R_HIP: 左/右髋
          L_KNEE/R_KNEE: 左/右膝
          L_ANK/R_ANK: 左/右踝
        """
        if J >= 33:
            # MediaPipe 33 点（世界坐标/图像坐标）
            return {
                'L_SH':11, 'R_SH':12, 'L_EL':13, 'R_EL':14, 'L_WR':15, 'R_WR':16,
                'L_HIP':23, 'R_HIP':24, 'L_KNEE':25, 'R_KNEE':26, 'L_ANK':27, 'R_ANK':28
            }
        elif J >= 25:
            # OpenPose BODY_25（常见 2D/3D）
            # 参考: 0鼻 1颈 2右肩 3右肘 4右腕 5左肩 6左肘 7左腕 8髋中心 9右髋 10右膝 11右踝
            #      12左髋 13左膝 14左踝 15右眼 16左眼 17右耳 18左耳 19左大脚趾 20左小脚趾 21左脚跟
            #      22右大脚趾 23右小脚趾 24右脚跟
            return {
                'L_SH':5,  'R_SH':2,  'L_EL':6,  'R_EL':3,  'L_WR':7,  'R_WR':4,
                'L_HIP':12,'R_HIP':9, 'L_KNEE':13,'R_KNEE':10,'L_ANK':14,'R_ANK':11
            }
        else:
            # 未知/过少点，返回空映射（后面会产生 NaN）
            return {}

    def _safe_xyz(frame: np.ndarray, i: int):
        """安全取点坐标，支持 (x,y) 或 (x,y,z,vis)。越界/缺失返回 NaN 向量。"""
        if frame is None or frame.ndim != 2:  # 异常帧
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        if not (0 <= i < frame.shape[0]):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        # 取前3维；若只有2维则补 z=0
        if frame.shape[1] >= 3:
            p = frame[i, :3]
        else:
            xy = frame[i, :2]
            p = np.array([xy[0], xy[1], 0.0], dtype=float)
        # 若有极端无效值转成 NaN
        p = np.where(np.isfinite(p), p, np.nan).astype(float)
        return p

    def _angle(a, b, c):
        """
        计算以 b 为顶点的夹角 ∠ABC（度）。任一向量无效则返回 NaN。
        """
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
            return np.nan
        ab = a - b
        cb = c - b
        nab = np.linalg.norm(ab)
        ncb = np.linalg.norm(cb)
        if nab <= 1e-9 or ncb <= 1e-9:
            return np.nan
        cosv = np.dot(ab, cb) / (nab * ncb)
        cosv = np.clip(cosv, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosv)))

    # 准备输出容器
    out = {'knee_L': [], 'knee_R': [], 'hip_L': [], 'hip_R': [], 'elbow_L': [], 'elbow_R': []}

    # 容错：空输入
    if kps is None:
        return {k: np.array([], dtype=float) for k in out}
    kps = list(kps)
    if len(kps) == 0:
        return {k: np.array([], dtype=float) for k in out}

    # 逐帧处理（每帧可能 J 不同；每帧都动态检测一次更稳妥）
    for frame in kps:
        if frame is None:
            J = 0
        else:
            f = np.asarray(frame)
            J = f.shape[0] if f.ndim == 2 else 0
        idx = _select_index_map(J)

        def P(name):  # 语法糖：按名称取点
            if name not in idx:
                return np.array([np.nan, np.nan, np.nan], dtype=float)
            return _safe_xyz(f, idx[name]) if J > 0 else np.array([np.nan, np.nan, np.nan], dtype=float)

        # 角度定义与原版一致：
        #   knee_* : ∠(hip, knee, ankle)
        #   hip_*  : ∠(shoulder, hip, knee)
        #   elbow_*: ∠(shoulder, elbow, wrist)
        A = {}
        A['knee_L']  = _angle(P('L_HIP'), P('L_KNEE'), P('L_ANK'))
        A['knee_R']  = _angle(P('R_HIP'), P('R_KNEE'), P('R_ANK'))
        A['hip_L']   = _angle(P('L_SH'),  P('L_HIP'),  P('L_KNEE'))
        A['hip_R']   = _angle(P('R_SH'),  P('R_HIP'),  P('R_KNEE'))
        A['elbow_L'] = _angle(P('L_SH'),  P('L_EL'),   P('L_WR'))
        A['elbow_R'] = _angle(P('R_SH'),  P('R_EL'),   P('R_WR'))

        for k in out:
            out[k].append(A[k])

    # 转换为 ndarray
    for k in out:
        out[k] = np.asarray(out[k], dtype=float)

    return out

def correlation_bucket_score(tpl_series, usr_series):
    names=set(tpl_series.keys()) & set(usr_series.keys())
    vals=[]
    for k in names:
        a=np.asarray(tpl_series[k]); b=np.asarray(usr_series[k])
        n=min(len(a),len(b))
        if n<5: continue
        ca=np.corrcoef(a[:n], b[:n])[0,1]; ca=0.0 if np.isnan(ca) else float(ca)
        vals.append(max(0.0, ca)*100.0)
    if not vals: return 0.0, {}
    return float(np.mean(vals)), {k: float(vals[i]) for i,k in enumerate(list(names)[:len(vals)])}

def _iter_mediapipe_points(video_path, step=2):
    try:
        import mediapipe as mp
    except Exception:
        return []
    mp_pose=mp.solutions.pose
    cap=cv2.VideoCapture(video_path); frames=[]; idx=0
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame=cap.read()
            if not ok: break
            if idx % step != 0: idx+=1; continue
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)
            arr=np.full((33,4), np.nan, dtype=np.float32)
            if res.pose_landmarks:
                h,w = frame.shape[:2]
                for i,lm in enumerate(res.pose_landmarks.landmark):
                    if i<33:
                        arr[i,0]=lm.x*w; arr[i,1]=lm.y*h; arr[i,2]=lm.z; arr[i,3]=lm.visibility
            frames.append(arr); idx+=1
    cap.release(); return frames

def _score_aligned(Ft, Fu, path):
    """仅用已对齐的角度序列给出稳健分数（避免依赖外部模板）。"""
    Ft=np.asarray(Ft, dtype=float); Fu=np.asarray(Fu, dtype=float)
    if Ft.ndim!=2 or Fu.ndim!=2: return 10.0
    pairs = np.asarray(path, dtype=int)
    ai = np.clip(pairs[:,0], 0, Ft.shape[0]-1)
    bi = np.clip(pairs[:,1], 0, Fu.shape[0]-1)
    A = Ft[ai]; B = Fu[bi]        # [N, D]
    if A.size==0 or B.size==0: return 10.0
    # 1) 角度 MAE -> 0..100
    mae = float(np.mean(np.abs(A-B)))
    mae_score = float(np.clip(100.0 - 1.0*mae, 0.0, 100.0))
    # 2) 逐维相关 -> 0..100
    cors = []
    for d in range(A.shape[1]):
        ca = np.corrcoef(A[:,d], B[:,d])[0,1] if A.shape[0]>2 else 0.0
        if np.isnan(ca): ca = 0.0
        cors.append(max(0.0, float(ca))*100.0)
    corr_score = float(np.mean(cors)) if cors else 0.0
    # 融合
    final = 0.6*corr_score + 0.4*mae_score
    return float(np.clip(final, 0.0, 100.0))

def score_from_path(*args, **kwargs):
    """
    兼容入口：根据参数类型自动分流。
    - 若第一个参数是 ndarray/list -> 视为 (Ft, Fu, path) 旧接口，返回 float；
    - 否则视为视频路径模式，返回 float。
    """
    # 角度序列旧接口
    if len(args)>=3 and (hasattr(args[0], "__array__") or isinstance(args[0], (list, tuple))):
        Ft, Fu, path = args[0], args[1], args[2]
        # 接收但不强制使用的兼容参数
        user_kps = kwargs.get("user_kps", None)
        tpl_kps  = kwargs.get("tpl_kps",  None)
        user_vis = kwargs.get("user_vis", None)
        tpl_vis  = kwargs.get("tpl_vis",  None)
        # 直接用对齐角度做稳健分数（与项目主流程保持一致）
        return _score_aligned(Ft, Fu, path)

    # 视频路径新接口
    if len(args)>=2 and isinstance(args[0], (str, type(None))) and isinstance(args[1], (str, type(None))):
        tpl_video_path, usr_video_path = args[0], args[1]
        action = kwargs.get("action", "taichi")
        use_template = kwargs.get("use_template", False)
        cfg_dir = kwargs.get("config_dir", os.path.join(os.path.dirname(__file__), "config", "actions"))
        if (not use_template) or (not tpl_video_path) or (not os.path.exists(tpl_video_path)):
            scorer=ScoreAggregator(action_name=action, config_dir=cfg_dir)
            for arr in _iter_mediapipe_points(usr_video_path, step=2):
                scorer.add_frame(arr)
            reps=scorer.flush_completed_reps()
            return float(np.mean([r['score'] for r in reps])) if reps else 10.0
        # 模板对照 + 新引擎融合
        usr_kps=_iter_mediapipe_points(usr_video_path, step=2)
        tpl_kps=_iter_mediapipe_points(tpl_video_path, step=2)
        from_series = lambda kps: build_angle_series_from_kps(kps)
        corr,_ = correlation_bucket_score(from_series(tpl_kps), from_series(usr_kps))
        scorer=ScoreAggregator(action_name=action, config_dir=cfg_dir)
        for arr in usr_kps: scorer.add_frame(arr)
        reps=scorer.flush_completed_reps()
        agg = float(np.mean([r['score'] for r in reps])) if reps else 0.0
        final = 0.5*agg + 0.5*corr
        if agg < 40.0: final = min(final, 60.0)
        return float(final)

    # 兜底
    raise TypeError("score_from_path: unsupported arguments. Expected (Ft, Fu, path, ...) or (tpl_video, usr_video, ...)")
