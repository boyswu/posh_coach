# -*- coding: utf-8 -*-
"""
Author: wujiahang
用途：异步视频解码与播放（长视频不卡 UI）
特性：
- 后台解码线程（QThread）
- 有界队列，积压时按策略丢帧（保证流畅度）
- 可选“解码降采样”和“最大FPS限制”
- 可中途停止/切换视频，避免野线程
- 兼容 OpenCV (cv2.VideoCapture)
"""
from collections import deque
import cv2
import time
from PySide6.QtCore import QThread, Signal, QObject

class VideoDecodeWorker(QThread):
    frame_ready = Signal(object)        # 发送 BGR ndarray
    progress = Signal(float)            # 发送进度 0~1
    opened = Signal(dict)               # 打开成功后发送元信息
    error = Signal(str)
    finished_ok = Signal()

    def __init__(self, path: str, max_queue=50, drop_policy="drop_old",
                 max_fps=None, downscale=1, prefer_ffmpeg=True, parent: QObject=None):
        super().__init__(parent)
        self.path = path
        self.max_queue = max_queue
        self.drop_policy = drop_policy   # "drop_old" 或 "drop_new"
        self.max_fps = max_fps           # e.g. 30；None 表示不限制
        self.downscale = max(1, int(downscale))
        self.prefer_ffmpeg = prefer_ffmpeg
        self._queue = deque()
        self._stop = False
        self._cap = None

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG if self.prefer_ffmpeg else 0)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频：{self.path}")
                return
            self._cap = cap

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            meta = {"fps": fps, "total": total, "width": w, "height": h, "path": self.path}
            self.opened.emit(meta)

            # 解码节流（保证不把 CPU/GPU 打满又不阻塞）
            next_deadline = time.perf_counter()
            read_idx = 0

            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                read_idx += 1

                # 可选：下采样降低分辨率，减少传输/推理开销
                if self.downscale > 1:
                    frame = frame[::self.downscale, ::self.downscale, :]

                # 有界队列 & 丢帧策略
                if len(self._queue) >= self.max_queue:
                    if self.drop_policy == "drop_old":
                        self._queue.popleft()  # 丢最旧
                    else:
                        # 丢最新：直接跳过当前帧
                        pass

                self._queue.append(frame)

                # 推进进度
                if total > 0:
                    self.progress.emit(min(1.0, read_idx / total))

                # 按最大 FPS 限速（不然后台线程会过度狂奔）
                if self.max_fps:
                    # 与下次输出的时间点对齐
                    next_deadline += 1.0 / float(self.max_fps)
                    now = time.perf_counter()
                    if next_deadline > now:
                        time.sleep(next_deadline - now)

                # 将队列最新帧吐给 GUI（避免 GUI 读队列再卡）
                # —— 这里采用“总是发送当前最新的一帧”，保证流畅不延时
                latest = self._queue.pop()
                self._queue.clear()
                self.frame_ready.emit(latest)

            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(f"解码线程异常：{e}")
        finally:
            try:
                if self._cap is not None:
                    self._cap.release()
            except:
                pass
