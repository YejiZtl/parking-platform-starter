# Parking Platform Starter

固定摄像头停车位占用检测项目：Ultralytics YOLO 负责车辆检测，人工标定的 ROI 和 122 个车位多边形负责占用判断。

详细维护流程见 [docs/MAINTENANCE.md](docs/MAINTENANCE.md)。运行参数默认值以 `parking_config.py` 为准，`.env` 和命令行只做覆盖。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env`，填入真实 `RTSP_URL`。不要提交 `.env`。

启动：

```powershell
python run_parking.py --display
```

或：

```powershell
.\start_parking.cmd
```

默认运行产物写入 `logs/`，包括轮转 JSONL、结构化运行日志、健康状态 JSON 和按连接会话命名的 MP4。

## 常用维护命令

截取参考帧：

```powershell
python capture_frame.py --output first_frame.jpg
```

重画 ROI、车位并更新校准元数据：

```powershell
python select_roi.py --image first_frame.jpg --output parking_roi.json
python select_spaces.py --image first_frame.jpg --json bounding_boxes.json
python update_calibration.py --image first_frame.jpg --regions bounding_boxes.json --roi parking_roi.json
```

查看和推进逐图复核训练闭环：

```powershell
python iterative_batch_label.py --show-progress
python iterative_batch_label.py
```

复测 V3 暗车保留集：

```powershell
python evaluate_dark_vehicle_recall.py --model releases/parking_vehicle_black_verified_v3/weights/best.pt --images datasets/parking_vehicles_black_verified_v3/images/val --labels datasets/parking_vehicles_black_verified_v3/labels/val --conf 0.12 --iou 0.35 --imgsz 1920 --output runs/parking_train/dark_recall_v3
```

创建不可变模型发布包：

```powershell
python release_model.py --model runs/parking_train/<run>/weights/best.pt --name <release_name> --evaluation <summary.json>
```

运行快速测试：

```powershell
python -m unittest discover -s tests -v
```

## 当前边界

本地代码实现周期检测、ROI/重复框过滤、一车一位车位匹配、状态防抖、日志轮转和健康文件。它不内置 MediaMTX/HLS/NVENC 推流；这些属于外部部署链路。

服务器是否已经运行 V3 不能从本地 `.env` 推断，必须在服务器上核实实际进程和加载模型路径。
