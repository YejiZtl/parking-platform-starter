#!/usr/bin/env bash
set -Eeuo pipefail

APP="/home/parkuser/parking-platform-starter"
LOG_FILE="$APP/logs/parking_rtsp.log"
PID_FILE="$APP/logs/parking_rtsp.pid"
BACKUP="${1:-}"

if [[ -z "$BACKUP" ]]; then
    BACKUP="$(cat "$APP/logs/last_maintenance_backup.txt")"
fi
[[ -d "$BACKUP" ]] || { echo "Backup directory not found: $BACKUP"; exit 1; }

CONDA_BIN="$(command -v conda 2>/dev/null || true)"
[[ -n "$CONDA_BIN" ]] || [[ ! -x /home/parkuser/miniconda3/bin/conda ]] || CONDA_BIN=/home/parkuser/miniconda3/bin/conda
[[ -n "$CONDA_BIN" ]] || [[ ! -x /home/parkuser/anaconda3/bin/conda ]] || CONDA_BIN=/home/parkuser/anaconda3/bin/conda
[[ -n "$CONDA_BIN" ]] || { echo "Conda not found."; exit 1; }

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

stop_runtime
while IFS= read -r relative; do
    [[ -n "$relative" ]] || continue
    if [[ -e "$BACKUP/files/$relative" ]]; then
        mkdir -p "$(dirname "$APP/$relative")"
        cp -a "$BACKUP/files/$relative" "$APP/$relative"
    else
        rm -f "$APP/$relative"
    fi
done < "$BACKUP/deployed_files.txt"

cp -a "$BACKUP/.env" "$APP/.env"
old_model_relative="$(cat "$BACKUP/previous_model_path.txt")"
if [[ "$old_model_relative" = /* ]]; then
    old_model="$old_model_relative"
else
    old_model="$APP/$old_model_relative"
fi
mkdir -p "$(dirname "$old_model")"
install -m 0644 "$BACKUP/previous_model.pt" "$old_model"
sha256sum -c "$BACKUP/previous_model.sha256"
chmod 600 "$APP/.env"
start_runtime
sleep 15
pid="$(cat "$PID_FILE")"
kill -0 "$pid" 2>/dev/null && pgrep -f '[r]un_parking_rtsp.py' >/dev/null || { tail -n 50 "$LOG_FILE" | sed -E 's#rtsp://[^[:space:]"]+#rtsp://***#g'; exit 1; }
echo "Rollback completed from: $BACKUP"
