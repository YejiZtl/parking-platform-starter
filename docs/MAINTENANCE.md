# 停车位占用系统维护手册

> 适用版本：V3 `parking_vehicle_black_verified_v3`

快速测试：

```powershell
python -m unittest discover -s tests -v
```

当前基线应有 28 个测试通过。测试不会加载大型 YOLO 模型，也不会访问 RTSP。

2026-09-05 已完成 [消融实验](ABLATION.md)，包含 9 组检测对照、车位匹配诊断、合成时间防抖和可复核预测证据。数据审计发现固定验证集 6 张图中有 4 张与当前 V1 训练目录完全相同，且 V3 继承 V2/V1 权重；该集应作为固定回归验证集，不再称为独立测试集。

## 项目目的与边界

本项目用于固定摄像头停车位占用检测。系统先用 Ultralytics YOLO 检测车辆，再结合人工标定的停车场 ROI 和 122 个车位多边形判断每个车位的状态。

本仓库负责：

- 周期性车辆检测。
- ROI 外检测过滤和重复检测框过滤。
- 车辆与车位多边形匹配。
- 一车一位可选约束、疑似状态和占用状态防抖。
- 校准元数据及启动校验。
- JSONL、结构化日志、健康状态和平台状态 JSON。
- 本地 OpenCV 视频输出。
- 服务器侧通过 FFmpeg 向已有 MediaMTX 发布标注画面。
- 数据集构建、人工复核训练闭环、固定验证集和模型发布清单。

本仓库不负责：

- 摄像头、网络、防火墙和 RTSP 账号管理。
- MediaMTX 服务本身的安装、自启动和运维。
- HLS 网页播放器或完整停车业务后台。
- 多目标跟踪、车辆身份识别、车牌识别或跨帧轨迹分析。
- 夜间、雨天、逆光等场景的自动质量保证。

当前系统是“周期检测 + 车位状态防抖”，不是多目标跟踪。代码使用 `model.predict`，没有启用 Ultralytics tracker，也没有可信的车辆 track ID。

## 当前交接基线

以下内容是 2026-08-27 完成交接部署后的状态快照，用于判断后续改动是否偏离基线。它不是配置单一真相源；运行时仍须核实服务器 `.env` 和实际进程。

### 生产运行

- 项目目录：`/home/parkuser/parking-platform-starter`
- Conda 环境：`parking`
- 服务入口：`run_parking_rtsp.py`
- 编码器：`h264_nvenc`
- 运行日志：`logs/parking_rtsp.log`
- PID 文件：`logs/parking_rtsp.pid`
- MediaMTX 发布路径：`/parking`
- 服务器本机 HLS：`http://127.0.0.1:8887/parking/index.m3u8`
- 当前由 `nohup` 和 PID 文件管理，没有 systemd；服务器重启后需要人工确认服务。

### 当前模型

- 发布名称：`parking_vehicle_black_verified_v3`
- 本地不可变发布源：`releases/parking_vehicle_black_verified_v3/weights/best.pt`
- 服务器兼容安装路径：`runs/parking_train/vehicle_detector_black_verified_v3/weights/best.pt`
- SHA-256：`f59b763a4598960209043085882154c44a2e3350d10626c35fd70bcc9c24e3a3`
- 文件大小：`44182583` 字节

本地发布目录和服务器安装目录不同是为兼容现有部署结构。发布、归档和校验以 `releases/` 为准；交接时服务器 `.env` 仍引用 `runs/parking_train/...`。不要根据本地 `.env` 推断服务器模型。

### 已验证运行参数

交接部署已核实 `VEHICLE_CLASS_IDS=0`、`CONF=0.12`、`IOU=0.35`、`IMGSZ=1920`、`MAX_DET=500` 和 `SLOT_OVERLAP_THRESHOLD=0.30`。

代码默认 `ONE_VEHICLE_ONE_SPACE=true`，但服务器实际覆盖值为 `false`。这是生产配置与代码默认值的明确差异。修改前必须对同一固定帧做前后对比，不能因为默认值不同就直接覆盖服务器。

固定参考帧部署回归指纹：

- 车位总数：122
- 占用：115
- 空闲：7
- 重复过滤后车辆检测：132

这组数字用于验证“同一帧、同一模型、同一配置”的代码兼容性，不是停车场的永久正确答案，也不能替代人工精度评估。

### 环境兼容

本地完整验证环境：Python 3.12.13、Ultralytics 8.4.121、Torch 2.13.0+cu130、Torchvision 0.28.0+cu130、OpenCV 5.0.0.93、NumPy 2.5.2、lap 0.5.13、Shapely 2.1.2、python-dotenv 1.2.3。

服务器 A10 环境：Python 3.12.13、Ultralytics 8.4.117、Torch 2.7.1+cu126、Torchvision 0.22.1、OpenCV 5.0.0.93、NumPy 2.5.2、lap 0.5.13、Shapely 2.1.2、python-dotenv 1.2.2。

服务器没有为了版本完全一致而强制升级 CUDA/PyTorch。兼容依据是固定帧输出完全一致、实时 RTSP 正常、A10 CUDA 可用、NVENC 发布成功且 HLS 可访问。以后升级时应建立新环境并做相同对比，不要覆盖当前可运行环境。

## 处理链路

一次服务器检测的数据流：

1. `run_parking_rtsp.py` 从 `.env` 或命令行读取 RTSP、模型、校准和输出配置。
2. 启动前检查模型、ROI、车位和校准文件，并检查阈值范围。
3. OpenCV 打开固定摄像头画面，取得实际分辨率。
4. `parking_geometry.py` 校验参考分辨率、文件哈希和 122 个车位。
5. `managed_parking.py` 调用 YOLO `predict` 检测车辆。
6. 过滤 ROI 外车辆，再过滤高度重叠、中心和尺寸接近的重复框。
7. 计算车辆框与车位多边形的相交面积，按车位面积比例生成占用候选。
8. 根据一车一位设置、疑似阈值和防抖计数生成稳定状态。
9. 写入 JSONL、健康 JSON 和 `parking-web/data/status.json`。
10. FFmpeg 使用 `h264_nvenc` 将标注画面发布到 MediaMTX `/parking`。
11. RTSP 断开后等待并重连，检测与异常均保留结构化状态。

## 核心设计决策

### 周期检测而非跟踪

业务目标是固定车位是否占用，不要求知道车辆轨迹。周期检测配合车位防抖依赖更少，也更容易从 RTSP 断流恢复。若以后增加轨迹业务，应独立设计跟踪模块，不能把检测框序号当作 track ID。

### 几何逻辑独立

生产推理、标注和评估需要相同的 ROI、重复框和车位规则。`parking_geometry.py` 是稳定共享层，避免生产代码从标注脚本导入函数，也减少评估与生产算法不一致。

### 校准使用元数据和哈希

只有坐标无法说明多边形对应哪张图和什么分辨率。`parking_calibration.json` 将参考帧、ROI、车位文件、尺寸、哈希和数量绑定，使误换文件或分辨率变化在启动阶段失败，而不是静默错位。

### 状态防抖

车辆框可能因遮挡、反光或检测间隔偶尔消失。`OCCUPIED_CONFIRMATIONS` 和 `EMPTY_CONFIRMATIONS` 控制状态切换需要的连续确认次数。防抖会产生可预期延迟，不能误判为推理卡顿。

### 标注必须逐图确认

预标注不是人工真值。只有当前图片明确保存后才写入该数据集专属复核状态。关闭窗口或只浏览不会把未确认图片加入训练，避免错误标签在迭代训练中被放大。

### 配置分三层

`parking_config.py` 保存代码默认值，`.env` 保存机器覆盖，命令行用于一次性实验。模型训练参数和结果属于发布元数据，应写入 release manifest，不能混入运行默认值。

## 文件索引

| 文件或目录 | 维护职责 |
| --- | --- |
| `run_parking.py` | 本地入口；配置、重连、日志、健康状态和本地视频 |
| `run_parking_rtsp.py` | 服务器入口；复用检测并通过 FFmpeg 发布 |
| `managed_parking.py` | 推理、过滤、车位匹配、疑似状态和防抖 |
| `parking_config.py` | 默认值和启动参数校验的单一真相源 |
| `parking_geometry.py` | ROI/车位缩放、重复框、校准与哈希工具 |
| `parking_calibration.json` | 参考尺寸、哈希和 122 个车位约束 |
| `bounding_boxes.json` | 122 个车位多边形 |
| `parking_roi.json` | 停车场有效检测区域 |
| `first_frame.jpg` | 校准参考帧，不进入 Git，需要单独交接 |
| `parking-web/` | 读取状态 JSON 的轻量展示端 |
| `simple_yolo_labeler.py` | 单图人工复核工具 |
| `iterative_batch_label.py` | 数据集隔离的迭代标注和训练编排 |
| `prepare_dataset_split.py` | 可移植 YOLO 数据集和哈希清单 |
| `fixed_validation_images_v3.txt` | 固定 6 张验证图名单，禁止进入训练 |
| `evaluate_dark_vehicle_recall.py` | V3 暗车代理保留集评估 |
| `run_ablation.py` | 固定配置消融、原始预测保存、祖先训练集文件重合审计 |
| `verify_ablation.py` | 无模型/无图像的 CPU 证据重算与指标一致性检查 |
| `release_model.py` | 生成不可变发布、SHA-256 和 manifest |
| `releases/` | 发布元数据；权重通过 Release 或离线存储交接 |
| `deploy/server/` | 安装、回滚、环境检查和冒烟脚本源文件 |
| `tests/` | 不加载模型、不访问 RTSP 的快速测试 |

## 配置来源

优先级：`parking_config.py` 默认值 -> `.env` -> 命令行。

新机器从 `.env.example` 复制出 `.env`，再由有权限的维护者填写 RTSP。`.env` 不得提交 Git、发到群聊或写入文档。

服务器只检查非敏感键，不要执行 `cat .env`：

```bash
cd /home/parkuser/parking-platform-starter && grep -E '^(MODEL_PATH|VEHICLE_CLASS_IDS|CONF|IOU|IMGSZ|MAX_DET|SLOT_OVERLAP_THRESHOLD|ONE_VEHICLE_ONE_SPACE|DETECT_EVERY_N_FRAMES|STREAM_ENCODER)=' .env
```

调整参数时先备份 `.env` 和固定帧输出，每次只改变一个有明确目的的参数组，并保存同输入的前后对比。不要用本地 `.env` 覆盖服务器 `.env`。

## 环境安装

推荐 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA 13.0 环境：

```powershell
pip install -r requirements-torch-cu130.txt
pip install -r requirements.txt
```

CPU 环境：

```powershell
pip install -r requirements-torch-cpu.txt
pip install -r requirements.txt
```

其他 CUDA 平台先安装匹配的 Torch/Torchvision，再安装 `requirements.txt`。不要在生产 Conda 环境中直接试验大版本升级。

## 本地启动

```powershell
copy .env.example .env
python run_parking.py --display
```

也可执行：

```powershell
.\start_parking.cmd
```

连接 RTSP 前，优先用固定图片离线冒烟：

```powershell
python run_parking.py --source first_frame.jpg --max-detections 1 --output "" --save-jsonl logs/smoke.jsonl --health-file logs/smoke_health.json
```

本地运行产物默认在 `logs/`。视频按时间和重连会话编号生成，不会因 RTSP 重连覆盖同名文件。

## 服务器日常运维

以下命令在服务器 Bash 执行。

检查进程和 PID：

```bash
cd /home/parkuser/parking-platform-starter && test -f logs/parking_rtsp.pid && kill -0 "$(cat logs/parking_rtsp.pid)" && pgrep -af '[r]un_parking_rtsp.py'
```

检查当前 V3 哈希：

```bash
cd /home/parkuser/parking-platform-starter && sha256sum runs/parking_train/vehicle_detector_black_verified_v3/weights/best.pt
```

查看最近日志：

```bash
cd /home/parkuser/parking-platform-starter && tail -n 50 logs/parking_rtsp.log | sed -E 's#rtsp://[^[:space:]" ]+#rtsp://***#g'
```

查看健康和平台状态：

```bash
cd /home/parkuser/parking-platform-starter && python -m json.tool logs/parking_health.json && python -m json.tool parking-web/data/status.json
```

检查 HLS，`-L` 用于跟随 MediaMTX 重定向：

```bash
curl -fsSL http://127.0.0.1:8887/parking/index.m3u8 | head
```

正常停止：

```bash
cd /home/parkuser/parking-platform-starter && test ! -f logs/parking_rtsp.pid || kill -TERM "$(cat logs/parking_rtsp.pid)"; pkill -TERM -f '[r]un_parking_rtsp.py' 2>/dev/null || true
```

后台启动：

```bash
cd /home/parkuser/parking-platform-starter && mkdir -p logs && { nohup conda run --no-capture-output -n parking python run_parking_rtsp.py --encoder h264_nvenc > logs/parking_rtsp.log 2>&1 < /dev/null & echo $! > logs/parking_rtsp.pid; }
```

如果 `conda` 不在 PATH，依次检查 `$HOME/miniconda3/bin/conda` 和 `$HOME/anaconda3/bin/conda`。启动后等待约 15 秒，再检查进程、日志、健康 JSON 和 HLS。只有 PID 存在不代表推理、NVENC 和 MediaMTX 都正常。

## ROI 与车位校准

当前链路：

1. `first_frame.jpg`：参考帧，`2560x1440`。
2. `parking_roi.json`：有效区域。
3. `bounding_boxes.json`：122 个车位。
4. `parking_calibration.json`：参考尺寸、哈希和数量。

重新标定：

```powershell
python capture_frame.py --output first_frame.jpg
python select_roi.py --image first_frame.jpg --output parking_roi.json
python select_spaces.py --image first_frame.jpg --json bounding_boxes.json
python update_calibration.py --image first_frame.jpg --regions bounding_boxes.json --roi parking_roi.json
```

先备份旧文件，再逐个检查 122 个车位，最后用实际画面叠加确认。默认不允许分辨率变化后静默缩放。只有确认物理机位不变、仅输出分辨率改变时，才可临时启用 `ALLOW_CALIBRATION_SCALE=true`。同分辨率下的镜头移动、变焦和角度变化仍必须重新标定。

## 训练闭环与数据谱系

流程：采集候选帧 -> 模型预标注 -> 人工逐图确认 -> 构建训练集 -> 训练 -> 固定保留集和新场景验证 -> 发布。

```powershell
python iterative_batch_label.py --show-progress
python iterative_batch_label.py
python iterative_batch_label.py --no-train
```

标注窗口只有按 `s`、`n` 或 `p` 保存当前图片时才确认。关闭窗口、直接退出或只查看图片都不会加入训练。

状态默认隔离在 `runs/parking_train/state/<dataset>-<hash>/`。不要手工合并不同数据集的复核清单。`prepare_dataset_split.py` 会生成相对路径 `data.yaml` 和 `dataset_manifest.json`。

`fixed_validation_images_v3.txt` 中的 6 张图永远不得进入训练。移动目录或重新导出数据集时，也要通过清单和哈希继续排除。

## V3 训练与验证

V3 由 V2 微调，包含 24 张人工确认训练图、3026 个实例；固定验证集 6 张图、745 个实例。训练配置为 60 epochs、`imgsz=1536`、AdamW、`lr0=0.0005`、`hsv_v=0.55`。

该验证集参与过最佳权重选择与调参，且存在祖先 V1 训练图像重合风险，不能用作独立测试成绩。当前 V2/V3 训练目录排除这 6 张图片，并不能消除继承权重的历史数据接触。未来发布的独立评估需要未参与任何祖先训练及调参的新数据。

运行评估基线为 `conf=0.12`、`iou=0.35`、`imgsz=1920`。保留集结果：

- 总体召回：98.39%
- 暗车代理召回：97.40%
- 非暗车召回：100%
- 匹配精度：95.44%

“暗车”只是框中央 HSV V 第 40 百分位不高于 95 的分析代理，不是人工车型或颜色标签。固定验证图仅来自同一机位同一天，不能代表夜间、雨天、逆光、镜头污染和季节变化。

新模型发布前必须复测固定保留集、检查数据泄漏、抽检新场景、检查疑似/跨位/空位释放、做固定帧部署回归，并完成真实 RTSP、NVENC、MediaMTX 和 HLS 冒烟。

## 模型发布

```powershell
python release_model.py --model runs/parking_train/<run>/weights/best.pt --name <release_name> --evaluation <summary.json>
```

正式发布至少包含 `weights/best.pt`、`best.pt.sha256`、`release_manifest.json`、`evaluation_summary.json` 和 `SERVER_DEPLOY.txt`。

不要覆盖旧 release，也不要让正式部署长期依赖会继续变化的训练 run。Git 忽略 `.pt`，正式权重应上传为 GitHub Release 资产或放入公司受控模型存储，并核对 SHA-256。

## 部署与回滚

`deploy/server/` 保存部署脚本源文件，`deploy/package/` 和 `.tgz` 是构建产物，不进入 Git。安装应完成 manifest 和模型哈希校验、环境预检、旧代码与 `.env` 备份、固定帧冒烟、后台启动、日志与 HLS 检查。

最近维护版本回滚：

```bash
cd /home/parkuser/parking-platform-starter && bash deploy/rollback.sh
```

重要：该脚本默认读取 `logs/last_maintenance_backup.txt`，恢复最近一次维护部署前的代码、`.env` 和模型文件。交接当天经历过多次部署尝试，最新维护备份中的 `previous_model.pt` 仍可能是 V3。因此它是“最近维护版本回滚”，不等于“保证切回 V2”。

需要 V2 模型回滚时，应单独确认 V2 哈希、备份 `.env`、切换 `MODEL_PATH` 和匹配参数、重启并完成固定帧及 HLS 验证。

交接时必须保留：

- `backups/maintenance_v3_20260827_160333`：当前维护代码回滚依据。
- `backups/deploy_v3_20260827_090016`：包含 V2 旧模型和早期部署资料。

部署不得覆盖真实 `.env`，不得打印完整 RTSP，也不得在验证成功前删除回滚备份。

## 日志与监控

| 文件 | 含义 |
| --- | --- |
| `logs/parking_rtsp.log` | 服务器标准输出和 FFmpeg/运行错误 |
| `logs/parking_runtime.log` | 结构化运行事件 |
| `parking_events.jsonl` 或配置路径 | 检测数量和过滤统计，支持轮转 |
| `logs/parking_health.json` | 运行、重连、停止或失败状态 |
| `parking-web/data/status.json` | 平台展示的车位状态快照 |

健康状态包括 `running`、`reconnecting`、`degraded`、`stopped` 和 `failed`。监控不能只看 PID，还要检查健康文件更新时间、`detections_completed` 是否增长、最近计数、日志错误和 HLS。目前没有主动告警，需由平台或运维系统补充。

## 故障排查

- 启动后退出：查看 `logs/parking_rtsp.log`，检查模型、校准、车位数量、FFmpeg、NVENC 和 Conda。
- 有进程但 HLS 无画面：确认检测增长、publisher 启动、MediaMTX 监听和 `/parking` 路径；使用 `curl -L`。
- 车位框整体错位：检查摄像头移动、变焦、码流和分辨率，不要绕过校准检查。
- 黑车漏检：保存原始帧，区分 YOLO 漏框、ROI 过滤、重复过滤和车位匹配问题。
- 一车多位或漏占：检查跨位事实、`SLOT_OVERLAP_THRESHOLD`、`ONE_VEHICLE_ONE_SPACE` 和过滤统计。
- 空位变化慢：检查 `EMPTY_CONFIRMATIONS`、检测间隔和 RTSP 丢帧，防抖延迟可能是预期行为。
- JSONL/日志过大：确认轮转和写入路径；删除历史前先确认审计需求。
