# Author: wujiahang
import numpy as np, cv2, os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import mediapipe as mp

    _OK = True
except Exception:
    _OK = False

BODY25_MAP = {0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 9: 24, 10: 26, 11: 28, 12: 23, 13: 25, 14: 27, 8: 24}


class MediaPipeBackend:
    def name(self):
        return 'MediaPipe'

    def video_to_keypoints(self, video_path: str, step: int = 1):
        if not _OK: raise RuntimeError('未安装 mediapipe')
        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(video_path);
        frames = [];
        idx = 0
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok: break
                if idx % step != 0:
                    idx += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                arr = np.full((25, 3), np.nan, dtype=np.float32)
                if res.pose_landmarks:
                    h, w = frame.shape[:2];
                    lm = res.pose_landmarks.landmark
                    for b, m in BODY25_MAP.items():
                        p = lm[m];
                        arr[b, 0] = p.x * w;
                        arr[b, 1] = p.y * h;
                        arr[b, 2] = getattr(p, 'visibility', 0.9)
                    if not np.isnan(arr[9, 0]) and not np.isnan(arr[12, 0]):
                        arr[8, :2] = (arr[9, :2] + arr[12, :2]) / 2.0;
                        arr[8, 2] = (arr[9, 2] + arr[12, 2]) / 2.0
                frames.append(arr);
                idx += 1
        cap.release()
        return np.stack(frames, axis=0) if frames else np.empty((0, 25, 3), dtype=np.float32)