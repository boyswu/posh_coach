# Author: wujiahang
from PySide6 import QtWidgets, QtCore, QtGui


class AspectLabel(QtWidgets.QLabel):
    """保持比例缩放，用于封面/结果预览/播放器容器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pix = QtGui.QPixmap()

    def setPixmap(self, pix):
        self._pix = pix if isinstance(pix, QtGui.QPixmap) else QtGui.QPixmap()
        super().setPixmap(self._scaled())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._pix.isNull():
            super().setPixmap(self._scaled())

    def _scaled(self):
        if self._pix.isNull():
            return QtGui.QPixmap()
        return self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)