#!/usr/bin/env bash
# PaperScope on a free Oracle Cloud server (Ubuntu), with free HTTPS from Caddy.
# Step by step, from creating the server: docs/FREE_HOSTING.md
#
# First time, on the server:
#   sudo apt-get update && sudo apt-get install -y git
#   git clone https://github.com/vaibhav-c-mane/paper-scope.git
#   bash paper-scope/deploy/oracle-setup.sh
# It asks for your API keys, an admin username and password, and your site address
# (for example paperscope.duckdns.org).
#
# Every time after that (new code pushed to GitHub, or to repair the setup):
#   bash ~/paper-scope/deploy/oracle-setup.sh
# It gets the newest code, updates the libraries and restarts the site (a few seconds).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$APP_DIR/.env"
PY="$APP_DIR/.venv/bin/python"
SERVICE=paperscope
PORT=8000

say() { printf '\n==> %s\n' "$*"; }

# KEY=value lines in .env, written by the same code as scripts/set_admin.py. The values travel
# in the environment, not on the command line, so other processes cannot read them.
set_env() {
    ENV_KEY="$1" ENV_VALUE="$2" "$PY" - "$ENV_FILE" <<'PYEOF'
import os, sys
sys.path.insert(0, "scripts")
from set_admin import update_env
update_env(sys.argv[1], {os.environ["ENV_KEY"]: os.environ["ENV_VALUE"]})
PYEOF
}
get_env() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -n 1 | cut -d= -f2- || true; }

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as your normal user (ubuntu), without sudo. It asks for sudo itself when needed." >&2
    exit 1
fi
cd "$APP_DIR"

say "Installing Python, git and Caddy (the web server that gives free HTTPS)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git curl
if ! command -v caddy >/dev/null 2>&1; then
    if ! sudo apt-get install -y caddy; then
        # Ubuntu older than 24.04 has no caddy package: use Caddy's own package list
        sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https gnupg
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
            | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
            | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
        sudo apt-get update -y
        sudo apt-get install -y caddy
    fi
fi

if [ -z "$(swapon --show --noheadings)" ]; then
    say "Adding 2 GB of swap, so a busy moment cannot run the server out of memory"
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

say "Opening the web ports 80 and 443 in the server's own firewall"
# Oracle's Ubuntu images reject every new connection except SSH; the rules go before that REJECT.
for port in 80 443; do
    if ! sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null; then
        line=$(sudo iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')
        if [ -n "$line" ]; then
            sudo iptables -I INPUT "$line" -p tcp --dport "$port" -m state --state NEW -j ACCEPT
        else
            sudo iptables -A INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT
        fi
    fi
done
if command -v netfilter-persistent >/dev/null 2>&1; then
    sudo netfilter-persistent save >/dev/null
fi

say "Getting the newest code from GitHub"
git pull --ff-only

say "Installing the Python libraries (the first time takes a few minutes)"
[ -x "$PY" ] || python3 -m venv .venv
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

if [ ! -f "$ENV_FILE" ]; then
    cp .env.example "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

# The questions come until the site's address is saved, the last of them, so a first run
# stopped half way asks them all again.
SITE="$(get_env PAPERSCOPE_SITE)"
if [ -z "$SITE" ]; then
    say "Your API keys. They are kept in $ENV_FILE on this server only."
    read -rp "OpenAlex API key (press Enter to skip): " value; set_env OPENALEX_API_KEY "$value"
    read -rp "Your email, sent with OpenAlex and Crossref requests (Enter to skip): " value; set_env OPENALEX_EMAIL "$value"
    read -rp "Semantic Scholar API key (Enter to skip): " value; set_env S2_API_KEY "$value"

    say "Your site's address"
    read -rp "Address, for example paperscope.duckdns.org: " SITE
    SITE="${SITE#https://}"; SITE="${SITE#http://}"; SITE="${SITE%%/*}"
    if ! [[ "$SITE" =~ ^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        echo "\"$SITE\" is not a web address like paperscope.duckdns.org. Run the script again." >&2
        exit 1
    fi
    set_env PAPERSCOPE_SITE "$SITE"
fi

if [ -z "$(get_env PAPERSCOPE_ADMIN_PASSWORD_HASH)" ]; then
    say "Your admin login (only for you)"
    "$PY" scripts/set_admin.py --site "https://$SITE"
fi

say "Running PaperScope as a service (it starts again by itself after a crash or a reboot)"
# Listens only inside the server: visitors reach it through Caddy, never directly.
sudo tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=PaperScope research paper search
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$APP_DIR
Environment=PORT=$PORT
Environment=PAPERSCOPE_HOST=127.0.0.1
Environment=PAPERSCOPE_PROXIES=1
ExecStart=$PY wsgi.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet $SERVICE
sudo systemctl restart $SERVICE

say "Setting up HTTPS for $SITE"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
# PaperScope, written by deploy/oracle-setup.sh. Caddy gets and renews the HTTPS certificate itself.
$SITE {
	encode gzip
	reverse_proxy 127.0.0.1:$PORT
}
EOF
sudo systemctl enable --quiet caddy
sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy

say "Waiting for PaperScope to start"
for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1; then
    echo "PaperScope did not start. Its last log lines:" >&2
    sudo journalctl -u $SERVICE -n 30 --no-pager >&2
    exit 1
fi

server_ip="$(curl -fsS --max-time 5 https://api.ipify.org || true)"
site_ip="$(getent ahostsv4 "$SITE" | awk 'NR == 1 {print $1}' || true)"
if [ -n "$server_ip" ] && [ "$server_ip" != "$site_ip" ]; then
    echo
    echo "WARNING: $SITE points to ${site_ip:-nothing}, but this server is $server_ip."
    echo "Set the IP of $SITE to $server_ip on duckdns.org; HTTPS starts working a minute after that."
fi

echo
echo "PaperScope is running."
echo "  Site:         https://$SITE"
echo "  Admin login:  https://$SITE/$(get_env PAPERSCOPE_ADMIN_PATH)/   (keep this to yourself)"
echo "  Its log:      sudo journalctl -u $SERVICE -f"
echo "  Update later: bash $APP_DIR/deploy/oracle-setup.sh"
