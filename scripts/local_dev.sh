#!/usr/bin/env bash
# Docker-free local runner: two API processes sharing one SQLite file,
# mirroring the docker-compose setup (same env knobs, exit-file crash hook).
#
#   scripts/local_dev.sh start     # api1 -> :8001, api2 -> :8002
#   scripts/local_dev.sh restart   # hard-crash both, then start again
#   scripts/local_dev.sh stop
#
# Then run the demo scenarios, e.g.:
#   python3 scripts/demo.py all --base http://127.0.0.1:8001 \
#       --restart-cmd "scripts/local_dev.sh restart"
set -euo pipefail
cd "$(dirname "$0")/.."

DB_DIR="${DB_DIR:-/tmp/corpus-local}"
DB_PATH="$DB_DIR/corpus.db"
COMPARE_DELAY_MS="${COMPARE_DELAY_MS:-3000}"
WORKER_INTERVAL_MS="${WORKER_INTERVAL_MS:-1000}"
PORT1="${PORT1:-8001}"
PORT2="${PORT2:-8002}"

start_one() {
  local port="$1"
  DB_PATH="$DB_PATH" INSTANCE_ID="local-$port" \
  COMPARE_DELAY_MS="$COMPARE_DELAY_MS" WORKER_INTERVAL_MS="$WORKER_INTERVAL_MS" \
  EXIT_WHEN_FILE_EXISTS="$DB_DIR/exit-$port" \
  nohup python3 -m uvicorn app.main:app --host 127.0.0.1 --port "$port" \
    >>"$DB_DIR/api-$port.log" 2>&1 &
  echo $! >"$DB_DIR/api-$port.pid"
}

wait_ready() {
  local port="$1"
  for _ in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$port/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "server on port $port did not become ready; see $DB_DIR/api-$port.log" >&2
  return 1
}

cmd="${1:-}"
case "$cmd" in
  start)
    mkdir -p "$DB_DIR"
    rm -f "$DB_DIR"/exit-*
    start_one "$PORT1"
    start_one "$PORT2"
    wait_ready "$PORT1"
    wait_ready "$PORT2"
    echo "api1: http://127.0.0.1:$PORT1/docs"
    echo "api2: http://127.0.0.1:$PORT2/docs"
    echo "db:   $DB_PATH"
    ;;
  stop)
    touch "$DB_DIR/exit-$PORT1" "$DB_DIR/exit-$PORT2" 2>/dev/null || true
    sleep 1
    rm -f "$DB_DIR"/exit-*
    ;;
  restart)
    "$0" stop || true
    "$0" start
    ;;
  *)
    echo "usage: $0 {start|stop|restart}" >&2
    exit 1
    ;;
esac
