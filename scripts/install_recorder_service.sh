#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fish/fishbot_pi_control}"
SERVICE_USER="${SERVICE_USER:-fish}"
RECORD_DURATION="${RECORD_DURATION:-300}"
SERVICE_PATH="/etc/systemd/system/fishbot-recorder.service"

sudo mkdir -p "${PROJECT_DIR}/logs"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=Fishbot offline sensor recorder
After=multi-user.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=FISHBOT_RECORD_DURATION=${RECORD_DURATION}
ExecStart=/bin/bash -lc 'mkdir -p logs; if [ -x .venv/bin/python ]; then exec .venv/bin/python scripts/record_sensors.py --duration "\$FISHBOT_RECORD_DURATION"; else exec python3 scripts/record_sensors.py --duration "\$FISHBOT_RECORD_DURATION"; fi'
Restart=no
StandardOutput=append:${PROJECT_DIR}/logs/fishbot-recorder.service.out
StandardError=append:${PROJECT_DIR}/logs/fishbot-recorder.service.err

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo "Installed ${SERVICE_PATH}"
echo "Start with:"
echo "  sudo systemctl start fishbot-recorder.service"
echo "Check with:"
echo "  sudo systemctl status fishbot-recorder.service"
echo "Logs are written under ${PROJECT_DIR}/logs"

