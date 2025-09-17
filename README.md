# Pose Coach

> Author: **wujiahang**  
> 基于 **PySide6 + MediaPipe** 的本地姿态训练教练应用：支持模板对比、训练录制、自动评分、结果可视化与历史记录。

---

## ✨ 功能概览

- **模板对比 / 训练录制**：对比模板视频或实时录制用户视频，自动对齐与分析。
- **自动评分（鲁棒）**  
  - 关键点 DTW（可见度加权 + 机位旋转/尺度归一）  
  - 关节角度 MAE  
  - 角度相关性（皮尔逊分档）  
  - 无动作/轻动作惩罚：静止≈10–20，轻微动作≤30（可调）
- **结果页可视化**  
  - 问题部位排行（红色进度条：躯干/上肢/下肢）  
  - 训练建议清单  
  - 角度相关性卡片  
  - 关键帧对比图 + 描述（历史记录也保留描述）
- **历史记录**：统一从导航栏进入，支持双击打开；旧数据自动兼容，必要时轻量复算。
- **主题切换**：支持浅/深色主题，全局统一调色。
- **UI 优化**：  
  - 首页卡片宽度限制（避免两侧留白随窗口扩大）  
  - 播放页按钮与难度选择统一布局  
  - 切页自动关闭摄像头与清空播放器  
  - 全局“返回”按钮置顶，尺寸适中

---

## 🧱 目录结构

```
pose-coach/
├─ start_app.py               # 启动入口
├─ requirements.txt
├─ README.md
└─ pose_coach/
   ├─ main.py                 # 主窗口与业务逻辑
   ├─ score.py                # 评分逻辑（DTW/MAE/相关性 + 惩罚/加成）
   ├─ db.py                   # 数据访问（读取 config / 环境变量）
   ├─ config.py               # 配置中心（从 env / config.json 加载）
   ├─ config.json             # 本地开发配置（示例）
   ├─ feedback.py             # 建议生成
   ├─ features.py / preprocess.py / align.py  # 特征/预处理/序列对齐
   ├─ backends.py             # MediaPipe 后端
   └─ ui/                     # Qt Designer 生成的 ui_xxx.py 与自定义控件
```

---

## ⚙️ 环境与安装

```bash
# 1) Python 环境（建议 3.10）
conda create -n pyside6_env python=3.10 -y
conda activate pyside6_env

# 2) 安装依赖
pip install -r requirements.txt
```

---

## 🔐 数据库配置（敏感信息外置）

项目不会在代码中硬编码数据库地址/账号/密码，统一在 `pose_coach/config.py` 读取：

优先级：**环境变量 > pose_coach/config.json > 默认值**

### 方式 A：环境变量（推荐生产）
```bash
export POSE_DB_HOST=127.0.0.1
export POSE_DB_PORT=3306
export POSE_DB_USER=root
export POSE_DB_PASSWORD=your_pass
export POSE_DB_NAME=posecoach
```

### 方式 B：`pose_coach/config.json`（本地开发）
```json
{
  "db": {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "your_password_here",
    "name": "posecoach"
  }
}
```

> 首次部署请确保数据库中存在相关表；如历史图片表缺少 `desc` 字段，可执行：  
> `ALTER TABLE analysis_images ADD COLUMN \`desc\` TEXT;`

---

## ▶️ 运行

```bash
python start_app.py
```

- 导航栏进入 “样本” 选择模板视频或“训练录制”进行采集  
- 点击“上传用户视频 / 播放 / 立即分析”  
- 结果页支持查看问题排行、建议、角度相关性和关键帧比对图  
- 历史记录统一在导航栏“历史记录”进入，支持双击打开

---

## 🧮 评分规则（可在 `score.py` 调整）

- **关键点 DTW 代价 → 分数**：对齐后每帧用加权 L2（关节权重×可见度）计算，骨架已做中心/尺度/肩线水平归一。  
- **角度 MAE 分**：模板与用户角度特征（由 `build_angles` 生成）沿对齐路径的 MAE。  
- **角度相关性分档**（鲁棒机位）：多关节三点角度的皮尔逊相关 resample 到同长度，≥0.98/0.90/0.75 → 1.0 / 0.7 / 0.4。  
- **融合权重**（默认）：`最终分 = 0.45*关键点评分 + 0.25*MAE分 + 0.30*相关性子分`  
- **质量门槛**：  
  - 几乎不动 → 10–20 分  
  - 轻微移动 → ≤ 30 分  
  - 覆盖率差（DTW 覆盖） → ≤ 60 分  
- **高相似加成**：关键点评分/MAE分/相关性均高且覆盖足够时，给少量加成，解决“机位不同但动作高度一致仍只有 50+”的问题。

---

## 🧭 使用小贴士

- **主题切换**：右上角“浅色主题”勾选即可全局切换。  
- **难度档位**：在模板对比和训练录制里均可选择“低/中/高”；互不影响。  
- **自动清理**：切换页面会关闭摄像头并清空用户播放器，避免资源占用。  
- **历史描述**：即时生成的关键帧对比图描述会随记录一起保存并显示在历史里；旧数据未存描述的会补默认文案。

---

## 🧰 开发与扩展

- 新增动作类别与模板：在 `data/templates/manifest.json` 里添加条目（封面、标题、视频路径）。  
- UI 统一风格：自定义样式在 `main.py` 的 `_light_css/_dark_css`；全局调色在 `apply_theme`。  
- 评分参数：在 `score.py` 中可调整 DTW 缩放、MAE 权重、相关分档阈值、惩罚/加成策略。  
- DB 访问：在 `db.py` 中使用 `config.py` 的 `DB_SETTINGS`，便于迁移到不同环境。

---

## 🛠️ 疑难排查

- **立即分析报错 lambda 形参个数**：请确保已更新 `score.py`（内部 `_dtw_path` 只向 `cost_fn(i, j)` 传两个索引）。  
- **历史记录无描述/卡片不显示**：请确认 `analysis_images` 表已增加 `desc`；`db.py/get_analysis_detail` 返回包含 `desc`。  
- **摄像头黑屏**：检查权限或更换 `cam_index`；光照不足会影响关键点识别。

---

## 📜 License

本项目仅用于学习与内部演示，不建议商用。如需商用请联系作者以获得授权。
