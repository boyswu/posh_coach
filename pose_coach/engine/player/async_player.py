# Author: wujiahang
import threading, queue, time
try:
    import av; _HAS_PYAV=True
except Exception:
    _HAS_PYAV=False
class AsyncVideoPipeline:
    def __init__(self, path, max_q=64, target_fps=30):
        import queue as _q
        self.path=path; self.decode_q=_q.Queue(maxsize=max_q); self.render_q=_q.Queue(maxsize=max_q)
        self.stop_flag=threading.Event(); self.target_fps=target_fps
    def start(self, infer_fn):
        self.t_decode=threading.Thread(target=self._decode_loop, daemon=True)
        self.t_infer=threading.Thread(target=self._infer_loop, args=(infer_fn,), daemon=True)
        self.t_decode.start(); self.t_infer.start()
    def stop(self): self.stop_flag.set()
    def _decode_loop(self):
        if _HAS_PYAV:
            container=av.open(self.path); stream=container.streams.video[0]; stream.thread_type='AUTO'
            for frame in container.decode(video=0):
                if self.stop_flag.is_set(): break
                img=frame.to_ndarray(format='bgr24')
                if self.decode_q.full():
                    try: self.decode_q.get_nowait()
                    except: pass
                self.decode_q.put(img, timeout=0.01)
        else:
            import cv2
            cap=cv2.VideoCapture(self.path)
            while cap.isOpened() and not self.stop_flag.is_set():
                ok,img=cap.read()
                if not ok: break
                if self.decode_q.full():
                    try: self.decode_q.get_nowait()
                    except: pass
                self.decode_q.put(img, timeout=0.01)
            cap.release()
    def _infer_loop(self, infer_fn):
        interval=1.0/max(1,self.target_fps); last_t=0.0
        while not self.stop_flag.is_set():
            try: img=self.decode_q.get(timeout=0.1)
            except: continue
            now=time.time()
            if now-last_t < interval*0.5 and not self.render_q.empty():
                continue
            out=infer_fn(img)
            if self.render_q.full():
                try: self.render_q.get_nowait()
                except: pass
            self.render_q.put(out, timeout=0.01); last_t=now
    def get_latest(self):
        item=None
        while not self.render_q.empty():
            try: item=self.render_q.get_nowait()
            except: break
        return item
