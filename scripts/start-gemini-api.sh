#!/bin/bash
# start-gemini-api.sh — launch/stop the local Gemini-FastAPI server for opencode2.
# Usage: ./start-gemini-api.sh start | stop | status
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${GEMINI_API_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
PID_FILE="${TMPDIR:-/tmp}/gemini-api.pid"
LOG_FILE="${TMPDIR:-/tmp}/gemini-api.log"

start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "gemini-api already running (pid $(cat "$PID_FILE"), $HEALTH_URL)"
    exit 0
  fi
  echo "Starting gemini-api from $REPO ..."
  ( cd "$REPO" && exec uv run python run.py ) >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 "${GEMINI_START_TIMEOUT_LOOPS:-40}"); do
    code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null)
    if [ "$code" = "200" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "gemini-api healthy at $HEALTH_URL (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    sleep 3
  done
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
  fi
  echo "ERROR: gemini-api did not become healthy within the timeout (${GEMINI_START_TIMEOUT_LOOPS:-40} x 3s). Log: $LOG_FILE" >&2
  tail -5 "$LOG_FILE" >&2
  exit 1
}

stop() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    for _ in $(seq 1 10); do
      kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
      sleep 1
    done
    rm -f "$PID_FILE"
    echo "gemini-api stopped"
  else
    echo "gemini-api not running"
  fi
}

status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    curl -s -m 3 "$HEALTH_URL" && echo && echo "pid $(cat "$PID_FILE")"
  else
    echo "gemini-api not running"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
