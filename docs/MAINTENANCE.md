# 停车位占用系统维护手册

## 目的与边界

本项目用于固定摄像头停车位占用检测：用 Ultralytics YOLO 检测车辆，再用人工标定的 ROI 和 122 个车位多边形计算占用状态。

当前本地代码实现的是周期性检测加车位状态防抖，不是多目标跟踪。`run_parking.py` 使用 OpenCV 读取 RTSP/视频源，并可写本地 `mp4v` MP4、JSONL 统计、运行日志和健康状态文件。MediaMTX、HLS、NVENC 或平台后端属于外部部署链路，不在本仓库代码内实现。

不要把本地 `.env` 当作服务器事实。服务器是否已经部署 V3，必须在服务器进程、工作目录、`.env` 和实际加载模型路径上确认。

## 架构与文件索引

- `run_parking.py`：本地运行入口，负责配置解析、启动校验、RTSP 重连、日志轮转、健康文件和视频输出。
- `run_parking_rtsp.py`：服务器运行入口，保留 FFmpeg NVENC 向 MediaMTX 发布的链路，并复用同一校准、检测、日志和健康实现。
- `managed_parking.py`：停车位占用核心，包含 ROI 过滤、重复检测框过滤、一车一位匹配、黄色疑似状态和状态防抖。
- `parking_config.py`：运行默认值的单一真相源；`.env` 和命令行只做覆盖。
- `parking_geometry.py`：ROI/车位缩放、校准校验、重复框几何、哈希和 JSON 工具。
- `parking_calibration.json`：当前固定摄像头校准元数据，引用 `first_frame.jpg`、`parking_roi.json`、`bounding_boxes.json` 的尺寸、数量和哈希。
- `bounding_boxes.json`：122 个车位多边形。保持旧格式列表，运行期通过 `parking_calibration.json` 获得参考分辨率。
- `parking_roi.json`：停车场有效检测区域。
- `simple_yolo_labeler.py`：逐图人工复核标注器；只有显式保存/确认的图片会写入确认清单。
- `iterative_batch_label.py`：按数据集隔离状态的自训练闭环。
- `prepare_dataset_split.py`：构建 YOLO 数据集、相对 `data.yaml` 和 `dataset_manifest.json`。
- `fixed_validation_images_v3.txt`：V3 固定 6 张保留验证图，默认不得进入后续训练 split。
- `release_model.py`：生成不可变模型发布目录、哈希和 release manifest。
- `releases/parking_vehicle_black_verified_v3/`：现有 V3 发布包清单、评估摘要、部署参考和本地权重。
- `tests/`：不加载大模型、不访问 RTSP 的快速单元测试。

## 环境安装

推荐 Python 3.12。创建虚拟环境后安装应用依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

已验证的 CUDA 130 PyTorch 组合可用：

```powershell
pip install -r requirements-torch-cu130.txt
pip install -r requirements.txt
```

CPU 或其他 CUDA 平台应先按目标平台安装匹配的 PyTorch/torchvision，再安装 `requirements.txt`。当前主依赖文件约束 PyTorch 大版本，但不强制所有机器使用同一个 CUDA wheel。

## 配置来源

配置优先级是：`parking_config.py` 默认值 -> `.env` -> 命令行参数。

复制 `.env.example` 为 `.env` 后再填写真实 RTSP；不要提交 `.env`。运行默认基线在 `parking_config.DEFAULTS`，文档只索引该源，避免另写一套参数表。现有 V3 发布包的不可变信息在 `releases/parking_vehicle_black_verified_v3/release_manifest.json` 和 `evaluation_summary.json`。

## 车位与 ROI 校准

校准文件链路：

1. `first_frame.jpg` 是当前参考帧，尺寸为 `2560x1440`。
2. `parking_roi.json` 记录停车场 ROI。
3. `bounding_boxes.json` 记录 122 个车位。
4. `parking_calibration.json` 记录参考尺寸、文件哈希和期望车位数量。

重新标定时：

```powershell
python capture_frame.py --output first_frame.jpg
python select_roi.py --image first_frame.jpg --output parking_roi.json
python select_spaces.py --image first_frame.jpg --json bounding_boxes.json
python update_calibration.py --image first_frame.jpg --regions bounding_boxes.json --roi parking_roi.json
```

启动时会校验 ROI、车位数量、文件哈希和摄像头分辨率。默认不允许分辨率变化后静默缩放；如确实确认同一机位只改变输出分辨率，才使用 `ALLOW_CALIBRATION_SCALE=true` 或 `--allow-calibration-scale`，并做人工画面对齐复核。同分辨率但机位物理移动无法仅靠 JSON 自动识别，必须用新参考帧重新校准。

## 日常启动

```powershell
.\start_parking.cmd
```

或：

```powershell
python run_parking.py --display
```

运行产物默认在 `logs/`：

- `parking_events.jsonl`：轮转 JSONL 统计。
- `parking_runtime.log`：结构化运行日志。
- `parking_health.json`：最近健康状态。
- `parking_result_时间_s编号.mp4`：每次 RTSP 连接会话一个文件，避免重连覆盖。

## 训练闭环

采集新帧后，用 `iterative_batch_label.py` 做“模型预标注 -> 人工逐图确认 -> 只用确认图训练”的闭环。状态目录默认按 dataset/images 派生到 `runs/parking_train/state/<dataset>-<hash>/`，不同轮次不会共享全局 `verified_images.txt`。

常用命令：

```powershell
python iterative_batch_label.py --show-progress
python iterative_batch_label.py
python iterative_batch_label.py --no-train
```

标注窗口里按 `s`、`n` 或 `p` 保存当前图片时，该图片才进入本次确认清单。直接退出或只查看的图片不会进入训练。`prepare_dataset_split.py` 默认读取 `fixed_validation_images_v3.txt` 并排除固定验证图。

## 验证标准

V3 是当前本地发布候选/发布模型，权重哈希见 release manifest。保留集结果和训练谱系以 `releases/parking_vehicle_black_verified_v3/evaluation_summary.json` 为准。

固定 6 张验证图仅来自同机位同一天，不代表夜间、雨天、逆光或服务器真实部署状态。发布或替换服务器前至少做：

- 固定验证集复测，命令参考 README。
- 新采集帧的人工对比，覆盖目标时段和天气。
- 启动健康文件、日志、模型路径和 `.env` 核查。
- 对黄色疑似车位、空位释放、防抖延迟做抽样人工检查。

## 模型发布

创建新发布包：

```powershell
python release_model.py --model runs/parking_train/<run>/weights/best.pt --name <release_name> --evaluation <summary.json>
```

发布目录必须包含：

- `weights/best.pt`
- `best.pt.sha256`
- `release_manifest.json`
- 评估摘要或部署说明

部署时使用 release 相对路径，不直接引用 `runs/` 训练目录。回滚就是把服务器 `.env` 的 `MODEL_PATH` 和相关参数切回上一份 release，并重启目标进程；回滚前后都要保存健康文件、日志片段和加载模型路径证据。

服务器维护包源文件位于 `deploy/server/`。安装脚本先校验模型哈希、依赖版本和 FFmpeg NVENC，再备份代码、`.env`、旧模型和日志；固定帧离线冒烟或 HLS 启动检查失败时自动调用回滚脚本。MediaMTX 仍是外部服务，本项目只向其 RTSP `/parking` 路径发布标注画面。

## 日志与监控

`run_parking.py` 写两类机器可读文件：

- JSONL 事件：每次检测一行，包含车位数、车辆检测数、ROI 外过滤数、重复框过滤数、匹配统计。
- 健康 JSON：当前状态、最近计数、运行模式和非敏感配置摘要。

这两个文件可由外部平台采集。日志轮转参数在 `parking_config.DEFAULTS`，部署可用 `.env` 覆盖。

## 故障排查

- 启动报模型不存在：检查 `MODEL_PATH` 是否指向 release 权重，且权重文件未被移动。
- 启动报校准哈希不匹配：重新运行 `update_calibration.py`，但只在确认 ROI/车位确实被重新标定后执行。
- 启动报分辨率不匹配：确认摄像头主/子码流、通道和输出尺寸；不要直接忽略。
- 黑车漏检：先复测 `evaluate_dark_vehicle_recall.py`，不要只凭总体 mAP 判断。
- 空位闪烁：检查 `EMPTY_CONFIRMATIONS`、检测间隔、RTSP 丢帧和黄色疑似车位。
- JSONL 过大：确认 `SAVE_JSONL` 指向 `logs/`，并检查 `JSONL_MAX_BYTES`/`JSONL_BACKUPS`。

## 数据备份

不要删除 `datasets/`、`runs/`、`releases/` 或根目录旧模型。`.gitignore` 会避免这些大文件进入版本库，但本地仍需单独备份：

- 原始帧和人工标签。
- 固定验证图与清单。
- 训练 run 目录。
- release 目录、哈希和部署说明。
- `.env` 的服务器私有副本。

## 修改检查清单

- 配置默认值是否只改了 `parking_config.py`，文档是否只索引配置源。
- ROI、车位或参考帧变更后，是否更新 `parking_calibration.json` 并人工复核画面对齐。
- 是否运行 `python -m unittest discover -s tests -v`。
- 是否避免加载大模型或访问 RTSP 的单元测试副作用。
- 是否确认固定 6 张验证图没有进入训练。
- 是否检查 `.env`、RTSP 凭据、模型权重、视频、数据集不会进入 Git。
- 是否保留旧模型和训练记录，发布新模型时是否生成 release manifest 与 SHA256。
