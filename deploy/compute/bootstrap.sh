#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/opt/medgpt
APP_DIR="${APP_ROOT}/app"
VENV_DIR="${APP_ROOT}/venv"
DATA_DIR=/srv/medgpt/index
LOG_DIR=/var/log/medgpt
ENV_FILE=/etc/default/medgpt

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  libgomp1 \
  nginx \
  rsync \
  curl

if ! id -u medgpt >/dev/null 2>&1; then
  sudo useradd --system --create-home --home-dir "${APP_ROOT}" \
    --shell /usr/sbin/nologin medgpt
fi

sudo mkdir -p "${APP_ROOT}" "${DATA_DIR}" "${LOG_DIR}"
sudo chown -R medgpt:medgpt "${APP_ROOT}" "${DATA_DIR}" "${LOG_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  sudo -u medgpt python3 -m venv "${VENV_DIR}"
fi

sudo -u medgpt "${VENV_DIR}/bin/pip" install --upgrade pip
sudo -u medgpt "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements-serve.txt"

sudo install -m 0644 "${APP_DIR}/deploy/compute/medgpt.service" \
  /etc/systemd/system/medgpt.service
sudo install -m 0644 "${APP_DIR}/deploy/compute/nginx-medgpt.conf" \
  /etc/nginx/sites-available/medgpt
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/medgpt /etc/nginx/sites-enabled/medgpt

sudo tee "${ENV_FILE}" >/dev/null <<'EOF'
INDEX_DIR=/srv/medgpt/index
LOCAL_CACHE_DIR=
EMBED_MODEL=BAAI/bge-m3
NPROBE=32
DEFAULT_K=50
MAX_K=200
OPENALEX_MAILTO=dan@danbroz.com
LOG_LEVEL=INFO
PORT=8080
EOF

sudo chown root:root "${ENV_FILE}"
sudo chmod 0644 "${ENV_FILE}"

sudo systemctl daemon-reload
sudo nginx -t
sudo systemctl enable medgpt nginx
sudo systemctl restart medgpt
sudo systemctl restart nginx

echo "bootstrap complete"
