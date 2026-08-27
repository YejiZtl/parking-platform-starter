#!/usr/bin/env bash
set -Eeuo pipefail

APP="/home/parkuser/parking-platform-starter"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$PACKAGE_DIR/payload"
LOG_FILE="$APP/logs/parking_rtsp.log"
PID_FILE="$APP/logs/parking_rtsp.pid"
FINAL_REL="runs/parking_train/vehicle_detector_black_verified_v3/weights/best.pt"
FINAL_MODEL="$APP/$FINAL_REL"
EXPECTED_HASH="f59b763a4598960209043085882154c44a2e3350d10626c35fd70bcc9c24e3a3"
EXPECTED_SIZE=44182583
ACTIVATED=0
BACKUP=""

RUNTIME_FILES=(
    run_parking_rtsp.py
    run_parking.py
    managed_parking.py
    parking_config.py
    parking_geometry.py
    parking_calibration.json
    bounding_boxes.json
    parking_roi.json
    first_frame.jpg
    requirements.txt
    requirements-torch-cu130.txt
    requirements-torch-cpu.txt
    README.md
    docs/MAINTENANCE.md
    deploy/rollback.sh
)

CONDA_BIN="$(command -v conda 2>/dev/null || true)"
[[ -n "$CONDA_BIN" ]] || [[ ! -x /home/parkuser/miniconda3/bin/conda ]] || CONDA_BIN=/home/parkuser/miniconda3/bin/conda
[[ -n "$CONDA_BIN" ]] || [[ ! -x /home/parkuser/anaconda3/bin/conda ]] || CONDA_BIN=/home/parkuser/anaconda3/bin/conda
[[ -n "$CONDA_BIN" ]] || { echo "Conda not found; no server files were changed."; exit 1; }

stop_runtime() {
    local pid=""
    [[ ! -f "$PID_FILE" ]] || pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
    pkill -TERM -f '[r]un_parking_rtsp.py' 2>/dev/null || true
    for _ in $(seq 1 10); do
        pgrep -f '[r]un_parking_rtsp.py' >/dev/null || break
        sleep 1
    done
    pgrep -f '[r]un_parking_rtsp.py' >/dev/null && pkill -KILL -f '[r]un_parking_rtsp.py' || true
    rm -f "$PID_FILE"
}

start_runtime() {
    cd "$APP"
    nohup "$CONDA_BIN" run --no-capture-output -n parking python run_parking_rtsp.py --encoder h264_nvenc > "$LOG_FILE" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$PID_FILE"
}

rollback_on_error() {
    local code=$?
    trap - ERR
    set +e
    if [[ "$ACTIVATED" -eq 1 && -n "$BACKUP" ]]; then
        echo "Deployment failed; restoring the previous runtime."
        bash "$PACKAGE_DIR/rollback.sh" "$BACKUP"
    fi
    exit "$code"
}
trap rollback_on_error ERR

[[ -d "$APP" && -f "$APP/.env" ]] || { echo "Project directory or .env is missing; no files were changed."; exit 1; }
[[ -f "$PACKAGE_DIR/MANIFEST.sha256" ]] || { echo "Package manifest is missing; no files were changed."; exit 1; }
(cd "$PACKAGE_DIR" && sha256sum -c MANIFEST.sha256)
for relative in "${RUNTIME_FILES[@]}"; do
    [[ -f "$PAYLOAD/$relative" ]] || { echo "Payload file missing: $relative"; exit 1; }
done
MODEL_SOURCE="$PAYLOAD/releases/parking_vehicle_black_verified_v3/weights/best.pt"
[[ -f "$MODEL_SOURCE" ]] || { echo "Release model is missing from the package."; exit 1; }
[[ "$(stat -c '%s' "$MODEL_SOURCE")" -eq "$EXPECTED_SIZE" ]] || { echo "Release model size mismatch."; exit 1; }
echo "$EXPECTED_HASH  $MODEL_SOURCE" | sha256sum -c -
command -v ffmpeg >/dev/null || { echo "FFmpeg not found; no files were changed."; exit 1; }
encoder_list="$(mktemp /tmp/parkingencoders.XXXXXX)"
ffmpeg -hide_banner -encoders > "$encoder_list" 2>/dev/null
grep -q h264_nvenc "$encoder_list" || { rm -f "$encoder_list"; echo "h264_nvenc is unavailable; no files were changed."; exit 1; }
rm -f "$encoder_list"
"$CONDA_BIN" run --no-capture-output -n parking python "$PACKAGE_DIR/check_environment.py"

preflight="$(mktemp -d /tmp/parkingpreflight.XXXXXX)"
"$CONDA_BIN" run --no-capture-output -n parking python "$PAYLOAD/run_parking_rtsp.py" --source "$PAYLOAD/first_frame.jpg" --regions "$PAYLOAD/bounding_boxes.json" --roi "$PAYLOAD/parking_roi.json" --calibration "$PAYLOAD/parking_calibration.json" --model "$MODEL_SOURCE" --no-publish --every-n-frames 1 --max-detections 1 --classes 0 --conf 0.12 --iou 0.35 --imgsz 1920 --device 0 --max-det 500 --slot-overlap-threshold 0.30 --no-one-vehicle-one-space --save-jsonl "$preflight/events.jsonl" --status-json "$preflight/status.json" --log-file "$preflight/runtime.log" --health-file "$preflight/health.json"
"$CONDA_BIN" run --no-capture-output -n parking python "$PACKAGE_DIR/verify_smoke.py" "$preflight/health.json"
[[ "$preflight" = /tmp/parkingpreflight.* ]] || { echo "Unexpected preflight path: $preflight"; exit 1; }
rm -rf -- "$preflight"

cd "$APP"
old_model_relative="$("$CONDA_BIN" run --no-capture-output -n parking python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("MODEL_PATH", ""))')"
[[ -n "$old_model_relative" ]] || { echo "MODEL_PATH is missing; no files were changed."; exit 1; }
if [[ "$old_model_relative" = /* ]]; then
    old_model="$old_model_relative"
else
    old_model="$APP/$old_model_relative"
fi
[[ -f "$old_model" ]] || { echo "Current model does not exist: $old_model"; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
BACKUP="$APP/backups/maintenance_v3_$stamp"
mkdir -p "$BACKUP/files" "$APP/logs"
printf '%s\n' "${RUNTIME_FILES[@]}" > "$BACKUP/deployed_files.txt"
for relative in "${RUNTIME_FILES[@]}"; do
    if [[ -e "$APP/$relative" ]]; then
        mkdir -p "$(dirname "$BACKUP/files/$relative")"
        cp -a "$APP/$relative" "$BACKUP/files/$relative"
    fi
done
cp -a "$APP/.env" "$BACKUP/.env"
printf '%s\n' "$old_model_relative" > "$BACKUP/previous_model_path.txt"
sha256sum "$old_model" > "$BACKUP/previous_model.sha256"
cp -a "$old_model" "$BACKUP/previous_model.pt"
[[ ! -f "$LOG_FILE" ]] || cp -a "$LOG_FILE" "$BACKUP/parking_rtsp.log"
printf '%s\n' "$BACKUP" > "$APP/logs/last_maintenance_backup.txt"
echo "Backup created: $BACKUP"

ACTIVATED=1
stop_runtime
for relative in "${RUNTIME_FILES[@]}"; do
    mkdir -p "$(dirname "$APP/$relative")"
    install -m 0644 "$PAYLOAD/$relative" "$APP/$relative"
done
chmod 0755 "$APP/deploy/rollback.sh"
mkdir -p "$(dirname "$FINAL_MODEL")"
install -m 0644 "$MODEL_SOURCE" "$FINAL_MODEL"
echo "$EXPECTED_HASH  $FINAL_MODEL" | sha256sum -c -

set_env() {
    local key="$1"
    local value="$2"
    if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$APP/.env"; then
        sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key}=${value}|" "$APP/.env"
    else
        printf '%s=%s\n' "$key" "$value" >> "$APP/.env"
    fi
}
other_before="$(awk '!/^[[:space:]]*(MODEL_PATH|VEHICLE_CLASS_IDS|CONF|IOU|IMGSZ|MAX_DET|SLOT_OVERLAP_THRESHOLD)[[:space:]]*=/' "$BACKUP/.env" | sha256sum | awk '{print $1}')"
set_env MODEL_PATH "$FINAL_REL"
set_env VEHICLE_CLASS_IDS 0
set_env CONF 0.12
set_env IOU 0.35
set_env IMGSZ 1920
set_env MAX_DET 500
set_env SLOT_OVERLAP_THRESHOLD 0.30
other_after="$(awk '!/^[[:space:]]*(MODEL_PATH|VEHICLE_CLASS_IDS|CONF|IOU|IMGSZ|MAX_DET|SLOT_OVERLAP_THRESHOLD)[[:space:]]*=/' "$APP/.env" | sha256sum | awk '{print $1}')"
[[ "$other_before" = "$other_after" ]] || { echo "Unexpected .env change outside the seven approved keys."; false; }
chmod 600 "$APP/.env"

smoke="$APP/logs/deploy_smoke_$stamp"
"$CONDA_BIN" run --no-capture-output -n parking python run_parking_rtsp.py --source first_frame.jpg --model "$FINAL_REL" --no-publish --every-n-frames 1 --max-detections 1 --save-jsonl "${smoke}.jsonl" --status-json "${smoke}_status.json" --log-file "${smoke}.log" --health-file "${smoke}_health.json"
"$CONDA_BIN" run --no-capture-output -n parking python "$PACKAGE_DIR/verify_smoke.py" "${smoke}_health.json"

start_runtime
sleep 15
new_pid="$(cat "$PID_FILE")"
kill -0 "$new_pid" 2>/dev/null && pgrep -f '[r]un_parking_rtsp.py' >/dev/null || { echo "New runtime exited during startup."; false; }
hls_file="/tmp/parkinghls.m3u8"
hls_code="$(curl -L --max-redirs 5 -sS --max-time 15 -o "$hls_file" -w '%{http_code}' http://127.0.0.1:8887/parking/index.m3u8)"
[[ "$hls_code" = 200 ]] && grep -q '#EXTM3U' "$hls_file" || { echo "HLS validation failed with HTTP $hls_code."; false; }
rm -f "$hls_file"

ACTIVATED=0
trap - ERR
echo "Deployment completed. PID=$new_pid"
echo "Model SHA256=$EXPECTED_HASH"
tail -n 50 "$LOG_FILE" | sed -E 's#rtsp://[^[:space:]"]+#rtsp://***#g'
