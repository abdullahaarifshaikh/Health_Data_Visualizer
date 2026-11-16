#!/usr/bin/env bash
set -euo pipefail
cd /vercel/healthlens
exec 9>/tmp/healthlens-start.lock
flock -x 9
if curl -fsS -H "X-HealthLens-Proxy: ${APP_PROXY_SECRET}" http://127.0.0.1:5000/api/status >/dev/null 2>&1; then
  exit 0
fi
if ! docker info >/dev/null 2>&1; then
  nohup dockerd >/tmp/healthlens-docker.log 2>&1 9>&- &
  timeout 60 bash -c 'until docker info >/dev/null 2>&1; do sleep 1; done'
fi
nohup .venv/bin/python app.py >/tmp/healthlens-server.log 2>&1 9>&- &
for i in $(seq 1 40); do
  if curl -fsS -H "X-HealthLens-Proxy: ${APP_PROXY_SECRET}" http://127.0.0.1:5000/api/status >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done
exit 1
