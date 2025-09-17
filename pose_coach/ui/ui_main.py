
# Author: wujiahang
from PySide6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(1320, 860)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(True)
        self.vbox = QtWidgets.QVBoxLayout(self.centralwidget)

        # 顶栏
        self.topBar = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Pose Coach")
        f = QtGui.QFont()
        f.setPointSize(18)
        f.setBold(True)
        self.title.setFont(f)
        self.chkLightTheme = QtWidgets.QCheckBox("浅色主题")
        self.topBar.addWidget(self.title)
        self.topBar.addStretch(1)
        # 紧凑返回键（替代页面内的长按钮）
        self.btnBackGlobal = QtWidgets.QToolButton()
        self.btnBackGlobal.setText("返回")
        self.btnBackGlobal.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.topBar.addWidget(self.btnBackGlobal)
        self.topBar.addWidget(self.chkLightTheme)
        self.vbox.addLayout(self.topBar)

        # 页面堆栈
        self.stack = QtWidgets.QStackedWidget()

        # 首页
        self.pageHome = QtWidgets.QWidget()
        self.layoutHome = QtWidgets.QVBoxLayout(self.pageHome)
        self.layoutCats = QtWidgets.QHBoxLayout()
        self.btnCatDance = QtWidgets.QPushButton("跳舞")
        self.btnCatLongJump = QtWidgets.QPushButton("跳远")
        self.btnCatPingpong = QtWidgets.QPushButton("乒乓球")
        self.btnCatFitness = QtWidgets.QPushButton("健身")
        self.btnAllSports = QtWidgets.QPushButton("全部运动")
        for b in [self.btnCatDance, self.btnCatLongJump, self.btnCatPingpong, self.btnCatFitness, self.btnAllSports]:
            b.setFixedHeight(36)
            self.layoutCats.addWidget(b)
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(QtCore.Qt.AlignHCenter)
        # 历史入口统一在侧边栏，这里不再放按钮
        self.layoutHome.addLayout(self.layoutCats)
        self.layoutHome.addWidget(self.scrollArea)
        self.stack.addWidget(self.pageHome)

        # 样本列表
        self.pageSamples = QtWidgets.QWidget()
        self.layoutSamples = QtWidgets.QVBoxLayout(self.pageSamples)
        self.labelSamples = QtWidgets.QLabel("样本视频")
        self.listSamples = QtWidgets.QListWidget()
        self.layoutSamples.addWidget(self.labelSamples)
        self.layoutSamples.addWidget(self.listSamples)
        self.stack.addWidget(self.pageSamples)

        # 播放 / 训练 详情页（两种模式 Tab）
        self.pageDetail = QtWidgets.QWidget()
        self.layoutDetail = QtWidgets.QVBoxLayout(self.pageDetail)
        self.labelDetailTitle = QtWidgets.QLabel("详情页")
        self.layoutDetail.addWidget(self.labelDetailTitle)

        self.tabModes = QtWidgets.QTabWidget()
        self.tabModes.setTabPosition(QtWidgets.QTabWidget.North)

        # —— 模式一：模板对比
        self.tabCompare = QtWidgets.QWidget()
        layoutCmp = QtWidgets.QVBoxLayout(self.tabCompare)

        self.layoutPlayers = QtWidgets.QHBoxLayout()
        self.playerTemplate = QtWidgets.QLabel()
        self.playerTemplate.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.playerTemplate.setMinimumSize(320, 180)
        self.playerUser = QtWidgets.QLabel()
        self.playerUser.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.playerUser.setMinimumSize(320, 180)
        self.layoutPlayers.addWidget(self.playerTemplate, 1)
        self.layoutPlayers.addWidget(self.playerUser, 1)
        layoutCmp.addLayout(self.layoutPlayers)

        hBtnsCmp = QtWidgets.QHBoxLayout()
        self.btnUploadUsr = QtWidgets.QPushButton("上传用户视频")
        self.btnPlayTpl = QtWidgets.QPushButton("播放/暂停样例视频")
        self.btnPlayUsr = QtWidgets.QPushButton("播放/暂停用户视频")
        self.btnAnalyze = QtWidgets.QPushButton("立即分析")
        self.btnViewResults = QtWidgets.QPushButton("查看结果")
        for b in [self.btnUploadUsr, self.btnPlayTpl, self.btnPlayUsr, self.btnAnalyze, self.btnViewResults]:
            hBtnsCmp.addWidget(b, 1)
        layoutCmp.addLayout(hBtnsCmp)

        hThresh = QtWidgets.QHBoxLayout()
        self.labelThresh = QtWidgets.QLabel("难度：")
        self.cmbDifficultyCompare = QtWidgets.QComboBox()
        self.cmbDifficultyCompare.setObjectName("cmbDifficultyCompare")
        self.cmbDifficultyCompare.addItems(["低要求", "中等要求", "高等要求"])
        hThresh.addWidget(self.labelThresh)
        hThresh.addWidget(self.cmbDifficultyCompare)
        layoutCmp.addLayout(hThresh)

        # 注意：删除多余的顶部“查看结果”按钮，避免重复
        self.tabModes.addTab(self.tabCompare, "模板对比")

        # —— 模式二：训练录制（使用 QSplitter 左右自适应）
        self.tabTrain = QtWidgets.QWidget()
        layoutTrain = QtWidgets.QVBoxLayout(self.tabTrain)

        self.splitTrain = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.playerTrainTpl = QtWidgets.QLabel()
        self.playerTrainTpl.setMinimumSize(560, 360)
        self.playerTrainTpl.setScaledContents(True)
        self.playerTrainUsr = QtWidgets.QLabel()
        self.playerTrainUsr.setMinimumSize(560, 360)
        self.playerTrainUsr.setScaledContents(True)
        self.splitTrain.addWidget(self.playerTrainTpl)
        self.splitTrain.addWidget(self.playerTrainUsr)
        self.splitTrain.setSizes([1, 1])
        layoutTrain.addWidget(self.splitTrain, 1)

        hBtnsTrain = QtWidgets.QHBoxLayout()
        self.btnBeginTrain = QtWidgets.QPushButton("开始运动（录制）")
        self.btnPlayTplTrain = QtWidgets.QPushButton("播放/暂停样例视频")
        self.btnToggleTrain = QtWidgets.QPushButton("开始/暂停运动（录制）")
        self.btnFinishTrain = QtWidgets.QPushButton("结束运动（分析结果）")
        self.btnViewResultsTrain = QtWidgets.QPushButton("查看结果")
        for b in [self.btnBeginTrain, self.btnPlayTplTrain, self.btnToggleTrain, self.btnFinishTrain, self.btnViewResultsTrain]:
            hBtnsTrain.addWidget(b, 1)
        layoutTrain.addLayout(hBtnsTrain)

        hThresh2 = QtWidgets.QHBoxLayout()
        self.labelThresh2 = QtWidgets.QLabel("难度：")
        self.cmbDifficultyTrain = QtWidgets.QComboBox()
        self.cmbDifficultyTrain.setObjectName("cmbDifficultyTrain")
        self.cmbDifficultyTrain.addItems(["低要求", "中等要求", "高等要求"])
        hThresh2.addWidget(self.labelThresh2)
        hThresh2.addWidget(self.cmbDifficultyTrain)
        layoutTrain.addLayout(hThresh2)

        self.tabModes.addTab(self.tabTrain, "训练录制")
        self.layoutDetail.addWidget(self.tabModes)
        self.stack.addWidget(self.pageDetail)

        # 结果页（可滚动）
        self.pageResults = QtWidgets.QWidget()
        self.layoutResults = QtWidgets.QVBoxLayout(self.pageResults)
        self.labelResults = QtWidgets.QLabel("分析结果")
        self.scrollResults = QtWidgets.QScrollArea()
        self.scrollResults.setWidgetResizable(True)
        self.resultsContainer = QtWidgets.QWidget()
        self.resultsVBox = QtWidgets.QVBoxLayout(self.resultsContainer)
        self.scrollResults.setWidget(self.resultsContainer)
        self.layoutResults.addWidget(self.labelResults)
        self.layoutResults.addWidget(self.scrollResults, 1)
        self.stack.addWidget(self.pageResults)

        # 历史页
        self.pageHistory = QtWidgets.QWidget()
        self.layoutHistory = QtWidgets.QVBoxLayout(self.pageHistory)
        self.labelHistory = QtWidgets.QLabel("历史记录")
        self.listHistory = QtWidgets.QListWidget()
        self.btnOpenHistory = QtWidgets.QPushButton("打开选中记录")
        self.layoutHistory.addWidget(self.labelHistory)
        self.layoutHistory.addWidget(self.listHistory)
        self.layoutHistory.addWidget(self.btnOpenHistory)
        self.stack.addWidget(self.pageHistory)

        self.vbox.addWidget(self.stack)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # 样式交由 Main.apply_theme 控制
        MainWindow.setStyleSheet("")