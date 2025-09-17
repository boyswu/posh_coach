# -*- coding: utf-8 -*-
# Author: wujiahang
import os
import sys
import time
import json
import tempfile
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

from PySide6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

from .ui.ui_main import Ui_MainWindow
from .ui.AspectLabel import AspectLabel
from .backends import MediaPipeBackend
from .preprocess import normalize_skeleton
from .features import build_angles
from .align import align_dtw
from .score import (
    score_from_path,
    build_angle_series_from_kps,
    correlation_bucket_score,
)
from .feedback import make_advice, suggest_recipes
from .db import (
    init_db,
    insert_analysis,
    insert_image,
    list_analyses,
    get_analysis_detail,
)

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "data"))
STORAGE_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "storage"))
os.makedirs(STORAGE_DIR, exist_ok=True)

EDGES = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)
]

JOINT_GROUPS = {
    "arms": ["LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist"],
    "legs": ["LHip", "LKnee", "LAnkle", "RHip", "RKnee", "RAnkle"],
    "torso": ["Neck", "LShoulder", "RShoulder", "LHip", "RHip"],
}
ANGLE_MAP = {
    "LElbow": 5, "RElbow": 6, "LKnee": 12, "RKnee": 13,
    "LHip": 10, "RHip": 11, "LShoulder": 3, "RShoulder": 4,
    "LAnkle": 14, "RAnkle": 15, "Neck": 2,
}


def load_manifest():
    path = os.path.join(DATA_DIR, "templates", "manifest.json")
    if not os.path.exists(path):
        return {"recommended": [], "samples": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_group_errors(Ft, Fu, path, scale=1.0):
    D = Ft.shape[1]
    errs = {name: [] for name in ANGLE_MAP.keys()}
    for tt, uu in path:
        if tt < 0 or uu < 0 or tt >= len(Ft) or uu >= len(Fu):
            continue
        for joint, idx in ANGLE_MAP.items():
            if idx >= D:
                continue
            e = abs(Fu[uu, idx] - Ft[tt, idx]) / max(scale, 1e-6)
            errs[joint].append(e)

    joint_top = {j: float(np.percentile(v, 90)) if len(v) else 0.0 for j, v in errs.items()}
    group_scores = {}
    for g, joints in JOINT_GROUPS.items():
        vals = [joint_top.get(j, 0.0) for j in joints]
        group_scores[g] = float(np.mean(vals)) if any(vals) else 0.0

    details = {}
    for g, joints in JOINT_GROUPS.items():
        pairs = [(j, joint_top.get(j, 0.0)) for j in joints]
        pairs.sort(key=lambda x: x[1], reverse=True)
        details[g] = pairs[:2]

    main_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    return group_scores, details, main_groups


class CameraWorker(QtCore.QObject):
    frameCaptured = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self._running = True

    @QtCore.Slot()
    def loop(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.finished.emit()
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            self.frameCaptured.emit(frame)
            QtCore.QThread.msleep(10)
        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False


class VideoPlayer(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.fps = 30
        self.frame_idx = 0
        self.kps = None
        self.kps_step = 1
        self.loop = True

    def load(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = int(fps) if fps and fps > 1 else 30
        self.frame_idx = 0

    def clear(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.timer.stop()
        super().setPixmap(QtGui.QPixmap())

    def attach_keypoints(self, kps, step=1):
        self.kps = kps
        self.kps_step = max(1, int(step))

    def play(self):
        if not self.cap:
            return
        self.timer.start(int(1000 / max(self.fps, 1)))

    def pause(self):
        self.timer.stop()

    def _render_pix(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        if self.kps is not None and len(self.kps) > 0 and self.cap is not None:
            idx = min(self.frame_idx // self.kps_step, len(self.kps) - 1)
            pts = self.kps[idx]
            painter = QtGui.QPainter(pix)
            pen = QtGui.QPen(QtGui.QColor(80, 220, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            for a, b in EDGES:
                if a < pts.shape[0] and b < pts.shape[0]:
                    xa, ya = pts[a, 0], pts[a, 1]
                    xb, yb = pts[b, 0], pts[b, 1]
                    if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                        painter.drawLine(int(xa), int(ya), int(xb), int(yb))
            brush = QtGui.QBrush(QtGui.QColor(255, 100, 100, 220))
            painter.setBrush(brush)
            for j in range(min(pts.shape[0], 25)):
                x, y = pts[j, 0], pts[j, 1]
                if not (np.isnan(x) or np.isnan(y)):
                    painter.drawEllipse(QtCore.QPoint(int(x), int(y)), 3, 3)
            painter.end()

        super().setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def next_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                ok, frame = self.cap.read()
                if not ok:
                    self.pause()
                    return
            else:
                self.pause()
                return
        self.frame_idx += 1
        self._render_pix(frame)

    def show_frame(self, frame):
        self._render_pix(frame)


class BusyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, text="正在生成结果…", eta_secs=None):
        super().__init__(parent)
        self.setWindowTitle("请稍候")
        self.setModal(True)
        v = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(text)
        v.addWidget(self.label)
        if eta_secs is not None:
            self.eta = QtWidgets.QLabel(f"预计完成：约 {int(eta_secs)} 秒")
            v.addWidget(self.eta)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        v.addWidget(self.bar)
        self.resize(360, 160)


class AnalyzeThread(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)

    def __init__(self, backend, tpl_video, usr_video, tpl_kps, step, bad_thresh, difficulty="medium"):
        super().__init__()
        self.backend = backend
        self.tpl_video = tpl_video
        self.usr_video = usr_video
        self.tpl_kps = tpl_kps
        self.step = step
        self.bad_thresh = bad_thresh
        self.difficulty = difficulty

    def _valid_brightness(self, frame, thr=18):
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
        return float(y.mean()) >= thr

    def run(self):
        try:
            res = self._run_impl()
            self.finished.emit(res)
        except Exception as e:
            self.finished.emit({"error": str(e)})

    def _annotate(self, img, pts, color=(0, 0, 255)):
        if pts is None or len(pts) == 0:
            return img
        canvas = img.copy()
        for a, b in EDGES:
            if a < pts.shape[0] and b < pts.shape[0]:
                xa, ya = pts[a, 0], pts[a, 1]
                xb, yb = pts[b, 0], pts[b, 1]
                if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                    cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), (80, 220, 255), 2)
        for j in range(min(pts.shape[0], 25)):
            x, y = pts[j, 0], pts[j, 1]
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(canvas, (int(x), int(y)), 3, color, -1)
        return canvas

    def _run_impl(self):
        self.progress.emit("提取用户关键点...")
        user_kps = self.backend.video_to_keypoints(self.usr_video, step=self.step)
        user_vis = getattr(self.backend, "last_visibility", None)
        tpl_vis = getattr(self.backend, "tpl_visibility", None)

        if user_kps is None or len(user_kps) == 0:
            raise RuntimeError("未检测到人体关键点，请确保画面清晰、全身入镜，或更换光照和机位后重试。")

        self.progress.emit("归一化特征...")
        t_norm = normalize_skeleton(self.tpl_kps)
        u_norm = normalize_skeleton(user_kps)

        Ft = build_angles(t_norm)
        Fu = build_angles(u_norm)
        if Ft.size == 0 or Fu.size == 0:
            raise RuntimeError("关键点特征为空，请检查模板视频与用户视频。")

        self.progress.emit("序列对齐与评分...")
        _, path = align_dtw(Ft, Fu)
        if not path:
            raise RuntimeError("对齐失败，视频过短或关键点缺失。")

        score = score_from_path(
            Ft, Fu, path,
            user_kps=np.array(user_kps),
            tpl_kps=np.array(self.tpl_kps) if self.tpl_kps is not None else None,
            user_vis=user_vis, tpl_vis=tpl_vis
        )
        # ========【插入起始：难度映射】========
        # 1) 按难度设定统一调节参数
        diff = (self.difficulty or "medium").lower()
        diff_map = {
            "low": {"diff_thresh": 0.28, "score_offset": +10.0, "score_scale": 1.00, "bad_adj": +8},
            "medium": {"diff_thresh": 0.18, "score_offset": +0.0, "score_scale": 1.00, "bad_adj": 0},
            "high": {"diff_thresh": 0.12, "score_offset": -10.0, "score_scale": 1.10, "bad_adj": -8},
        }
        cfg = diff_map.get(diff, diff_map["medium"])

        # 2) 最终分按难度修正（让“换难度”能明显影响结果）
        score = float(max(0.0, min(100.0, score * cfg["score_scale"] + cfg["score_offset"])))

        # 3) 更新“坏片段阈值”用于后续问题帧筛选
        diff_thresh = cfg["diff_thresh"]

        # 4) 如果你有“bad 阈值”（比如界面里 spinBad），也一并按难度微调
        bad_thresh_eff = max(0, min(100, (self.bad_thresh or 40) + cfg["bad_adj"]))
        # ========【插入结束：难度映射】========

        ignored = score < 15.0
        group_scores, group_details, main_groups = compute_group_errors(Ft, Fu, path)

        # diff_thresh = {"low": 0.25, "medium": 0.18, "high": 0.12}.get(self.difficulty, 0.18)

        self.progress.emit("抽取问题帧并生成对比图...")
        cap_u = cv2.VideoCapture(self.usr_video)
        cap_t = cv2.VideoCapture(self.tpl_video)
        total_u = int(cap_u.get(cv2.CAP_PROP_FRAME_COUNT))
        total_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_u = cap_u.get(cv2.CAP_PROP_FPS) or 25

        per_second_counter = {}
        out_dir = os.path.join(STORAGE_DIR, f"session_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        img_pairs = []

        pick_indices = []
        for tt, uu in path:
            if np.isnan(Fu[uu]).all():
                continue
            err_vec = [abs(Fu[uu, idx] - Ft[tt, idx]) for idx in ANGLE_MAP.values() if idx < Fu.shape[1]]
            if not err_vec:
                continue
            mean_err = float(np.mean(err_vec))
            if mean_err < diff_thresh:
                continue
            sec = int(uu / max(fps_u, 1))
            per_second_counter.setdefault(sec, 0)
            if per_second_counter[sec] >= 2:
                continue
            per_second_counter[sec] += 1
            pick_indices.append((tt, uu))

        for tt, uu in pick_indices[:12]:
            fu = min(int(uu * total_u / max(len(Fu), 1)), total_u - 1)
            ft = min(int(tt * total_t / max(len(Ft), 1)), total_t - 1)
            cap_u.set(cv2.CAP_PROP_POS_FRAMES, fu)
            oku, fru = cap_u.read()
            cap_t.set(cv2.CAP_PROP_POS_FRAMES, ft)
            okt, frt = cap_t.read()

            if not oku or not okt:
                continue
            if not self._valid_brightness(fru) or not self._valid_brightness(frt):
                continue

            kidx_u = min(uu, len(user_kps) - 1)
            kidx_t = min(tt, len(self.tpl_kps) - 1)
            fru_anno = self._annotate(fru, user_kps[kidx_u], (0, 0, 255))
            frt_anno = self._annotate(frt, self.tpl_kps[kidx_t], (0, 255, 0))

            p_u = os.path.join(out_dir, f"user_{fu}.jpg")
            p_t = os.path.join(out_dir, f"tpl_{ft}.jpg")
            cv2.imwrite(p_u, fru_anno)
            cv2.imwrite(p_t, frt_anno)

            desc = "该时刻主要部位与模板差异较大，请关注节奏与关键关节控制"
            img_pairs.append({"user": p_u, "tpl": p_t, "desc": desc})

        cap_u.release()
        cap_t.release()

        return {
            "score": score,
            "pairs": img_pairs,
            "user_kps": user_kps,
            "group_scores": group_scores,
            "group_details": group_details,
            "main_groups": main_groups,
            "ignored": ignored,
        }


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.backend = MediaPipeBackend()
        self.step = 1
        self.bad_thresh = 40
        self.difficulty_compare = "medium"
        self.difficulty_train = "medium"

        self.tpl_video = None
        self.usr_video = None
        self.tpl_kps = None

        self._rec_thread = None
        self._rec_worker = None
        self._rec_frames = []
        self._rec_h = None
        self._rec_w = None
        self._capture_enabled = False

        try:
            init_db()
            self.statusBar().showMessage("数据库就绪")
        except Exception as e:
            self.statusBar().showMessage(f"数据库异常：{e}")

        self._build_sidebar()
        self.apply_theme(False)
        self.ui.chkLightTheme.stateChanged.connect(
            lambda *args: self.apply_theme(args[0] == QtCore.Qt.Checked)
        )
        self.apply_theme(self.ui.chkLightTheme.isChecked())

        self.manifest = load_manifest()
        self.setup_home_feed()

        # 分类选择
        self.ui.btnCatDance.clicked.connect(lambda *args: self.open_category("跳舞"))
        self.ui.btnCatLongJump.clicked.connect(lambda *args: self.open_category("跳远"))
        self.ui.btnCatPingpong.clicked.connect(lambda *args: self.open_category("乒乓球"))
        self.ui.btnCatFitness.clicked.connect(lambda *args: self.open_category("健身"))
        self.ui.btnAllSports.clicked.connect(lambda *args: self.open_all_sports())

        # 播放器替换与等权伸展
        self.player_tpl = VideoPlayer()
        self.player_usr = VideoPlayer()
        self.player_train_tpl = VideoPlayer()
        self.player_train_usr = VideoPlayer()

        self.ui.layoutPlayers.replaceWidget(self.ui.playerTemplate, self.player_tpl)
        self.ui.playerTemplate.deleteLater()
        self.ui.layoutPlayers.replaceWidget(self.ui.playerUser, self.player_usr)
        self.ui.playerUser.deleteLater()
        self.ui.layoutPlayers.setStretch(0, 1)
        self.ui.layoutPlayers.setStretch(1, 1)

        self.ui.playerTrainTpl.setParent(None)
        self.ui.playerTrainUsr.setParent(None)
        self.ui.splitTrain.insertWidget(0, self.player_train_tpl)
        self.ui.splitTrain.insertWidget(1, self.player_train_usr)
        self.ui.splitTrain.setSizes([1, 1])

        # 难度 & 查看结果
        self.ui.cmbDifficultyCompare.currentIndexChanged.connect(self._on_diff_compare_changed)
        self.ui.cmbDifficultyTrain.currentIndexChanged.connect(self._on_diff_train_changed)
        self.ui.btnViewResults.clicked.connect(lambda *args: self.ui.stack.setCurrentWidget(self.ui.pageResults))
        self.ui.btnViewResultsTrain.clicked.connect(lambda *args: self.ui.stack.setCurrentWidget(self.ui.pageResults))

        # 顶部返回
        self.ui.btnBackGlobal.clicked.connect(lambda *args: self.ui.stack.setCurrentWidget(self.ui.pageHome))

        # 上传/播放/分析
        self.ui.btnUploadUsr.clicked.connect(lambda *args: self.load_usr_for_compare())
        self.ui.btnPlayTpl.clicked.connect(lambda *args: self.toggle_play(self.player_tpl))
        self.ui.btnPlayUsr.clicked.connect(lambda *args: self.toggle_play(self.player_usr))
        self.ui.btnAnalyze.clicked.connect(lambda *args: self.start_analyze(context="compare"))

        # 训练
        self.ui.btnBeginTrain.clicked.connect(lambda *args: self.begin_train())
        self.ui.btnPlayTplTrain.clicked.connect(lambda *args: self.toggle_play(self.player_train_tpl))
        self.ui.btnToggleTrain.clicked.connect(lambda *args: self.toggle_train_capture())
        self.ui.btnFinishTrain.clicked.connect(lambda *args: self.finish_train())

        # 页面切换清理/历史自动刷新
        self.ui.tabModes.currentChanged.connect(lambda *args: self._on_page_changed(0))
        self.ui.stack.currentChanged.connect(self._on_page_changed)
        self.ui.stack.currentChanged.connect(
            lambda *args: self.populate_history() if self.ui.stack.currentWidget() == self.ui.pageHistory else None
        )
        self.ui.stack.setCurrentWidget(self.ui.pageHome)

        # 历史记录交互
        self.ui.btnOpenHistory.clicked.connect(lambda *args: self.open_selected_history())
        self.ui.listHistory.itemDoubleClicked.connect(lambda *args: self.open_selected_history())

    # ===== 主题 =====
    def _light_css(self) -> str:
        return (
            "QMainWindow{background:#f6f7fb;color:#1d2433;}"
            "QLabel{color:#1d2433;}"
            "QTextEdit,QListWidget,QLineEdit{background:#ffffff;color:#1d2433;border:1px solid #d8dee9;border-radius:10px;}"
            "QListWidget::item{padding:10px;margin:4px;}"
            "QPushButton{background:#4c6fff;color:#ffffff;border:none;padding:8px 14px;border-radius:10px;}"
            "QPushButton:hover{filter:brightness(1.05);}"
            "QFrame#Card{background:#ffffff;border:1px solid #e5e9f2;border-radius:12px;}"
        )

    def _dark_css(self) -> str:
        return (
            "QMainWindow{background:#0b1220;color:#e5ecff;}"
            "QLabel{color:#e5ecff;}"
            "QTextEdit,QListWidget,QLineEdit{background:#0f162e;color:#e5ecff;border:1px solid #24314e;border-radius:10px;}"
            "QListWidget::item{padding:10px;margin:4px;}"
            "QPushButton{background:#3b5be6;color:#ffffff;border:none;padding:8px 14px;border-radius:10px;}"
            "QPushButton:hover{filter:brightness(1.05);}"
            "QFrame#Card{background:#0f162e;border:1px solid #24314e;border-radius:12px;}"
        )

    def apply_theme(self, light: bool):
        pal = QtGui.QPalette()
        if light:
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f6f7fb"))
            pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1d2433"))
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#1d2433"))
        else:
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#0b1220"))
            pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e5ecff"))
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#0f162e"))
            pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#e5ecff"))
        QtWidgets.QApplication.instance().setPalette(pal)
        extras = (
            "QMessageBox{background:#ffffff;color:#000;}"
            "QMessageBox QLabel{color:#000;}"
        )
        css = self._light_css() if light else self._dark_css()
        QtWidgets.QApplication.instance().setStyleSheet(css + extras)

    # ===== 首页推荐 =====
    def setup_home_feed(self):
        recs = load_manifest().get("recommended", [])
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QGridLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(16)

        wrapper = QtWidgets.QWidget()
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addStretch(1)
        wrapper_layout.addWidget(content, 0, QtCore.Qt.AlignTop)
        wrapper_layout.addStretch(1)
        content.setMaximumWidth(1280)

        cols = 3
        r = c = 0
        for item in recs:
            card = QtWidgets.QFrame(objectName="Card")
            v = QtWidgets.QVBoxLayout(card)
            cover = AspectLabel()
            cover.setMinimumHeight(220)
            cover_path = os.path.join(os.path.dirname(APP_DIR), item.get("cover", ""))
            pix = QtGui.QPixmap(cover_path) if os.path.exists(cover_path) else QtGui.QPixmap()
            cover.setPixmap(pix)
            h = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel(item.get("title", ""))
            play = QtWidgets.QPushButton("播放")
            play.clicked.connect(lambda *args, it=item: self.open_recommend(it))
            h.addWidget(title)
            h.addStretch(1)
            h.addWidget(play)
            v.addWidget(cover)
            v.addLayout(h)
            content_layout.addWidget(card, r, c)
            c += 1
            if c >= cols:
                c = 0
                r += 1

        self.ui.scrollArea.setWidget(wrapper)

    # ===== 侧边导航 =====
    def _build_sidebar(self):
        dock = QtWidgets.QDockWidget("导航", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        def nav(btn_text, target):
            btn = QtWidgets.QPushButton(btn_text)

            def go():
                self._on_page_changed(0)
                self.ui.stack.setCurrentWidget(getattr(self.ui, target))

            btn.clicked.connect(lambda *args: go())
            v.addWidget(btn)

        nav("首页", "pageHome")
        nav("样本", "pageSamples")
        nav("播放", "pageDetail")
        nav("结果", "pageResults")
        nav("历史记录", "pageHistory")
        v.addStretch(1)
        panel.setLayout(v)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    # ===== 分类/播放 =====
    def open_all_sports(self):
        all_cats = list(load_manifest().get("samples", {}).keys())
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("全部运动")
        layout = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        for c in all_cats:
            listw.addItem(c)
        btn = QtWidgets.QPushButton("打开分类")
        layout.addWidget(listw)
        layout.addWidget(btn)

        def open_selected():
            it = listw.currentItem()
            if not it:
                return
            dlg.accept()
            self.open_category(it.text())

        btn.clicked.connect(lambda *args: open_selected())
        dlg.exec()

    def open_recommend(self, item):
        self.tpl_video = os.path.join(os.path.dirname(APP_DIR), item["video"])
        self.ui.labelDetailTitle.setText(item.get("title", "播放页"))
        if os.path.exists(self.tpl_video):
            for player in (self.player_tpl, self.player_train_tpl):
                player.load(self.tpl_video)
                player.play()
            try:
                self.tpl_kps = self.backend.video_to_keypoints(self.tpl_video, step=self.step)
                for player in (self.player_tpl, self.player_train_tpl):
                    player.attach_keypoints(self.tpl_kps, step=self.step)
            except Exception as e:
                self.statusBar().showMessage(f"模板提取失败：{e}")
        self.ui.stack.setCurrentWidget(self.ui.pageDetail)

    def open_category(self, cat: str):
        self.ui.listSamples.clear()
        samples = load_manifest().get("samples", {}).get(cat, [])
        for s in samples:
            it = QtWidgets.QListWidgetItem(f"{cat} - {s['title']}")
            it.setData(QtCore.Qt.UserRole, s)
            it.setSizeHint(QtCore.QSize(0, 44))
            self.ui.listSamples.addItem(it)
        self.ui.listSamples.itemClicked.connect(self.on_sample_clicked)
        self.ui.stack.setCurrentWidget(self.ui.pageSamples)

    def on_sample_clicked(self, item: QtWidgets.QListWidgetItem):
        s = item.data(QtCore.Qt.UserRole)
        self.tpl_video = os.path.join(os.path.dirname(APP_DIR), s["video"])
        self.ui.labelDetailTitle.setText(s["title"])
        if os.path.exists(self.tpl_video):
            for player in (self.player_tpl, self.player_train_tpl):
                player.load(self.tpl_video)
                player.play()
            try:
                self.tpl_kps = self.backend.video_to_keypoints(self.tpl_video, step=self.step)
                for player in (self.player_tpl, self.player_train_tpl):
                    player.attach_keypoints(self.tpl_kps, step=self.step)
            except Exception as e:
                self.statusBar().showMessage(f"模板提取失败：{e}")
        self.ui.stack.setCurrentWidget(self.ui.pageDetail)

    # ===== 比对/训练 =====
    def load_usr_for_compare(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择用户视频", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        if not p:
            return
        self.usr_video = p
        self.player_usr.load(p)
        self.player_usr.play()

    def _on_diff_compare_changed(self, idx: int):
        self.difficulty_compare = {0: "low", 1: "medium", 2: "high"}.get(idx, "medium")

    def _on_diff_train_changed(self, idx: int):
        self.difficulty_train = {0: "low", 1: "medium", 2: "high"}.get(idx, "medium")

    def begin_train(self):
        self._stop_camera()
        self._rec_frames = []
        self._rec_h = None
        self._rec_w = None
        self._capture_enabled = True
        self._rec_timer = QtCore.QElapsedTimer()
        self._rec_timer.start()

        self._rec_thread = QtCore.QThread()
        self._rec_worker = CameraWorker()
        self._rec_worker.moveToThread(self._rec_thread)
        self._rec_thread.started.connect(self._rec_worker.loop)
        self._rec_worker.frameCaptured.connect(self.on_rec_frame_train)
        self._rec_worker.finished.connect(self._rec_thread.quit)
        self._rec_worker.finished.connect(self._rec_worker.deleteLater)
        self._rec_thread.finished.connect(self._rec_thread.deleteLater)
        self._rec_thread.start()
        self.statusBar().showMessage("开始录制…（实时预览中）")

    def on_rec_frame_train(self, frame):
        self.player_train_usr.show_frame(frame)
        if self._rec_h is None:
            self._rec_h, self._rec_w = frame.shape[:2]
        if self._capture_enabled:
            self._rec_frames.append(frame)

    def toggle_train_capture(self):
        self._capture_enabled = not self._capture_enabled
        state = "继续采集" if self._capture_enabled else "暂停采集"
        self.statusBar().showMessage(state)

    def finish_train(self):
        self._stop_camera()
        if not self._rec_frames:
            QtWidgets.QMessageBox.information(self, "提示", "没有采集到视频帧")
            return
        h, w = self._rec_h, self._rec_w
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for fr in self._rec_frames:
            out.write(fr)
        out.release()
        self.usr_video = tmp.name
        self.player_train_usr.load(tmp.name)
        self.player_train_usr.play()
        self.statusBar().showMessage(f"录制完成，时长 {self._rec_timer.elapsed() / 1000:.1f}s，即将分析…")
        self.start_analyze(context="train")
        self._rec_frames = []

    def _stop_camera(self):
        if self._rec_worker is not None:
            self._rec_worker.stop()
        if self._rec_thread is not None:
            self._rec_thread.quit()
            self._rec_thread.wait(1500)
        self._rec_thread = None
        self._rec_worker = None

    def _on_page_changed(self, _idx: int):
        self._stop_camera()
        for p in (self.player_usr, self.player_train_usr):
            p.clear()
        self.usr_video = None

    def toggle_play(self, player: VideoPlayer):
        if player.timer.isActive():
            player.pause()
        else:
            player.play()

    # ===== 分析 =====
    def start_analyze(self, context: str = "compare"):
        if self.tpl_video is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先选择或播放一个模板视频")
            return
        if self.usr_video is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先上传或录制用户视频")
            return
        if self.tpl_kps is None or len(self.tpl_kps) == 0:
            try:
                self.tpl_kps = self.backend.video_to_keypoints(self.tpl_video, step=self.step)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "失败", f"模板关键点提取失败：{e}")
                return

        cap = cv2.VideoCapture(self.usr_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100
        cap.release()
        eta = max(3, int(frames / fps * 0.6))

        self.busy = BusyDialog(self, "正在生成结果…", eta_secs=eta)
        self.busy.show()

        difficulty = self.difficulty_compare if context == "compare" else self.difficulty_train
        self.bad_thresh = getattr(self.ui, "spinBad", None).value() if hasattr(self.ui, "spinBad") else 40

        self.worker = AnalyzeThread(
            self.backend, self.tpl_video, self.usr_video, self.tpl_kps, self.step, self.bad_thresh, difficulty
        )
        self.worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self.worker.finished.connect(self.on_analyze_finished)
        self.worker.finished.connect(self.busy.close)
        self.statusBar().showMessage("分析中...")
        self.worker.start()

    def _build_rank_card(self, group_scores, group_details, main_groups):
        card = QtWidgets.QFrame(objectName="Card")
        v = QtWidgets.QVBoxLayout(card)
        title = QtWidgets.QLabel("问题部位排名")
        v.addWidget(title)
        parts_map = {"arms": "上肢", "legs": "下肢", "torso": "躯干"}
        for key, val in main_groups[:3]:
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(parts_map.get(key, key))
            bar = QtWidgets.QProgressBar()
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(int(min(100, max(0, val * 100))))
            bar.setStyleSheet(
                "QProgressBar{background:#1f2937;border:1px solid #3b82f6;border-radius:6px;text-align:center;}"
                "QProgressBar::chunk{background:#e74c3c;border-radius:6px;}"
            )
            row.addWidget(lab)
            row.addWidget(bar, 1)
            v.addLayout(row)
            if key in group_details:
                sub = ", ".join([f"{j}:{s:.2f}" for j, s in group_details[key]])
                v.addWidget(QtWidgets.QLabel(f" · 关键关节：{sub}"))
        return card

    def _build_recipe_card(self, recipes):
        if not recipes:
            return None
        card = QtWidgets.QFrame(objectName="Card")
        v = QtWidgets.QVBoxLayout(card)
        v.addWidget(QtWidgets.QLabel("训练建议清单"))
        for line in recipes:
            lab = QtWidgets.QLabel("· " + line)
            lab.setWordWrap(True)
            v.addWidget(lab)
        return card

    def _build_result_card(self, user_img, tpl_img, desc):
        card = QtWidgets.QFrame(objectName="Card")
        lay = QtWidgets.QVBoxLayout(card)
        imgs = QtWidgets.QHBoxLayout()
        l1 = AspectLabel()
        l2 = AspectLabel()
        for lab, path in [(l1, tpl_img), (l2, user_img)]:
            pix = QtGui.QPixmap(path) if path and os.path.exists(path) else QtGui.QPixmap()
            lab.setPixmap(pix)
            lab.setMinimumSize(360, 220)
            lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            imgs.addWidget(lab, 1)
        lay.addLayout(imgs)
        txt = QtWidgets.QLabel(desc)
        txt.setWordWrap(True)
        lay.addWidget(txt)
        return card

    def _build_corr_card(self, per_r: dict) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame(objectName="Card")
        vbox = QtWidgets.QVBoxLayout(card)
        title = QtWidgets.QLabel("角度相关性（与模板的一致性）")
        vbox.addWidget(title)

        name_map = {
            "L_Knee": "左膝",
            "R_Knee": "右膝",
            "L_Elbow": "左肘",
            "R_Elbow": "右肘",
        }

        items = sorted(per_r.items(), key=lambda kv: kv[1])

        for key, r in items:
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(name_map.get(key, key))
            pct = int(max(0.0, min(1.0, (r + 1.0) / 2.0)) * 100)
            bar = QtWidgets.QProgressBar()
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(pct)
            bar.setStyleSheet(
                "QProgressBar{background:#1f2937;border:1px solid #3b82f6;border-radius:6px;text-align:center;}"
                "QProgressBar::chunk{background:#e74c3c;border-radius:6px;}"
            )
            val = QtWidgets.QLabel(f"r={r:.2f}")
            val.setMinimumWidth(60)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(lab)
            row.addWidget(bar, 1)
            row.addWidget(val)
            vbox.addLayout(row)

        return card

    def on_analyze_finished(self, result: dict):
        if "error" in result:
            QtWidgets.QMessageBox.critical(self, "失败", result["error"])
            return
        if result.get("ignored"):
            QtWidgets.QMessageBox.information(self, "提示", "判定为无关动作，本次不记录结果")
            return

        score = result["score"]
        struct = {
            "overall_score": float(score),
            "group_scores": result.get("group_scores", {}),
            "group_details": result.get("group_details", {}),
            "main_groups": result.get("main_groups", []),
        }
        advice = make_advice(struct, mode="rule")

        user_kps = result.get("user_kps")
        if user_kps is not None and len(user_kps) > 0:
            self.player_usr.attach_keypoints(user_kps, step=self.step)
            self.player_train_usr.attach_keypoints(user_kps, step=self.step)

        # 保存分析与对比图（带描述）
        try:
            aid = insert_analysis("guest", "auto", self.tpl_video or "", self.usr_video or "", float(score), advice)
            pair_paths = []
            for pair in result.get("pairs", []):
                t = pair.get("tpl", "")
                u = pair.get("user", "")
                d = pair.get("desc", "") or "该时刻主要部位与模板差异较大，请关注节奏与关键关节控制"
                try:
                    insert_image(aid, t, u, d)  # 新库：四参（含 desc）
                except TypeError:
                    insert_image(aid, t, u)  # 旧库：三参回退
                pair_paths.append((t, u, d))
        except Exception as e:
            self.statusBar().showMessage(f"保存失败：{e}")
            pair_paths = [(p.get("tpl", ""), p.get("user", ""), p.get("desc", "") or
                           "该时刻主要部位与模板差异较大，请关注节奏与关键关节控制")
                          for p in result.get("pairs", [])]

        for i in reversed(range(self.ui.resultsVBox.count())):
            item = self.ui.resultsVBox.takeAt(i)
            w = item.widget()
            if w:
                w.deleteLater()

        header_text = f"综合得分：{score:.1f}  —  建议：{advice}"
        header = QtWidgets.QLabel(header_text)
        header.setWordWrap(True)
        self.ui.resultsVBox.addWidget(header)

        rank_card = self._build_rank_card(struct["group_scores"], struct["group_details"], struct["main_groups"])
        self.ui.resultsVBox.addWidget(rank_card)

        recipes = suggest_recipes(struct["main_groups"])
        recipe_card = self._build_recipe_card(recipes)
        if recipe_card:
            self.ui.resultsVBox.addWidget(recipe_card)

        try:
            tpl_series = build_angle_series_from_kps(self.tpl_kps) if self.tpl_kps is not None else {}
            usr_series = build_angle_series_from_kps(result.get("user_kps")) if result.get("user_kps") is not None else {}
            _corr_sub, per_r = correlation_bucket_score(tpl_series, usr_series)
            if per_r:
                corr_card = self._build_corr_card(per_r)
                self.ui.resultsVBox.addWidget(corr_card)
        except Exception as e:
            self.statusBar().showMessage(f"相关性可视化失败：{e}")

        for t, u, desc in pair_paths:
            card = self._build_result_card(u, t, desc)
            self.ui.resultsVBox.addWidget(card)

        self.ui.resultsVBox.addStretch(1)
        self.ui.stack.setCurrentWidget(self.ui.pageResults)

    # ===== 历史 =====
    def populate_history(self):
        try:
            rows = list_analyses(200)
            self.ui.listHistory.clear()
            for r in rows:
                aid, created, act, score = r
                it = QtWidgets.QListWidgetItem(f"#{aid}  {created}  {act}  分数 {score:.1f}")
                it.setData(QtCore.Qt.UserRole, aid)
                it.setSizeHint(QtCore.QSize(0, 48))
                self.ui.listHistory.addItem(it)
        except Exception as e:
            self.statusBar().showMessage(f"读取历史失败：{e}")

    def open_selected_history(self) -> None:
        """打开历史记录：带描述、红色进度条、建议卡；若无结构数据则轻量复算一次。"""
        item = self.ui.listHistory.currentItem()
        if not item:
            QtWidgets.QMessageBox.information(self, "提示", "请先选择一条记录")
            return

        aid = item.data(QtCore.Qt.UserRole)
        try:
            detail = get_analysis_detail(aid)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "失败", f"读取记录失败：{e}")
            return

        # --- 统一数据结构 ---
        head = {}
        images = []
        if isinstance(detail, dict) and "head" in detail:
            head = detail["head"]
            images = detail.get("images", [])
            score = float(head.get("score", 0.0))
            advice = head.get("advice", "")
            tpl_video = head.get("template_video", "")
            usr_video = head.get("user_video", "")
        else:
            # 向后兼容（旧 get_analysis_detail 返回 tuple）
            score = 0.0
            advice = ""
            tpl_video = usr_video = ""
            if isinstance(detail, (list, tuple)) and len(detail) >= 1:
                h = detail[0]
                if isinstance(h, (list, tuple)) and len(h) >= 7:
                    tpl_video, usr_video = h[3], h[4]
                    score = float(h[5] or 0.0)
                    advice = h[6] or ""
            if isinstance(detail, (list, tuple)) and len(detail) >= 2:
                imgs = detail[1]
                for r in imgs:
                    if isinstance(r, (list, tuple)):
                        images.append({"tpl": r[0], "user": r[1], "desc": (r[2] if len(r) > 2 else "")})
                    elif isinstance(r, dict):
                        images.append({"tpl": r.get("tpl", ""), "user": r.get("user", ""), "desc": r.get("desc", "")})

        # --- 轻量复算：分组误差 + 相关性（只有当我们能拿到视频路径时才复算） ---
        group_scores = {}
        group_details = {}
        main_groups = []
        corr_per = {}

        try:
            if os.path.exists(tpl_video) and os.path.exists(usr_video):
                # 提取关键点（使用现有后端）
                tpl_kps = self.backend.video_to_keypoints(tpl_video, step=self.step)
                usr_kps = self.backend.video_to_keypoints(usr_video, step=self.step)

                from .preprocess import normalize_skeleton
                from .features import build_angles
                from .align import align_dtw
                from .score import build_angle_series_from_kps, correlation_bucket_score

                Ft = build_angles(normalize_skeleton(tpl_kps))
                Fu = build_angles(normalize_skeleton(usr_kps))
                if Ft.size and Fu.size:
                    _, path = align_dtw(Ft, Fu)
                    if path:
                        # 复用与你在线分析一致的分组误差计算
                        gs, gd, mg = compute_group_errors(Ft, Fu, path)
                        group_scores, group_details, main_groups = gs, gd, mg

                # 角度相关性（和在线显示一致）
                tpl_series = build_angle_series_from_kps(tpl_kps) if tpl_kps is not None else {}
                usr_series = build_angle_series_from_kps(usr_kps) if usr_kps is not None else {}
                _, corr_per = correlation_bucket_score(tpl_series, usr_series)
        except Exception as e:
            self.statusBar().showMessage(f"历史复算失败：{e}")

        # --- 渲染到结果页（与在线结果页一致） ---
        for i in reversed(range(self.ui.resultsVBox.count())):
            it = self.ui.resultsVBox.takeAt(i)
            if it.widget():
                it.widget().deleteLater()

        header = QtWidgets.QLabel(f"综合得分：{score:.1f}  —  建议：{advice}")
        header.setWordWrap(True)
        self.ui.resultsVBox.addWidget(header)

        # 红色进度条卡片（有数据才显示）
        if main_groups:
            rank_card = self._build_rank_card(group_scores, group_details, main_groups)
            self.ui.resultsVBox.addWidget(rank_card)

            # 建议卡（沿用在线规则）
            from .feedback import suggest_recipes
            recipes = suggest_recipes(main_groups)
            if recipes:
                recipe_card = self._build_recipe_card(recipes)
                self.ui.resultsVBox.addWidget(recipe_card)

        # 角度相关性卡片
        if corr_per:
            corr_card = self._build_corr_card(corr_per)
            self.ui.resultsVBox.addWidget(corr_card)

        # 图片对（带 desc）
        DEFAULT_DESC = "该时刻主要部位与模板差异较大，请关注节奏与关键关节控制"
        for p in images[:12]:
            card = self._build_result_card(
                user_img=p.get("user", ""),
                tpl_img=p.get("tpl", ""),
                desc=p.get("desc", "") or DEFAULT_DESC,
            )
            self.ui.resultsVBox.addWidget(card)

        self.ui.resultsVBox.addStretch(1)
        self.ui.stack.setCurrentWidget(self.ui.pageResults)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if __package__ in (None, ""):
        import sys as _s
        import os as _o
        _s.path.append(_o.path.dirname(_o.path.dirname(__file__)))
    main()
