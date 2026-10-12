#!/usr/bin/env bash
# Keep the pursuit server and its public HTTPS tunnel running (restarts either if it exits).
#
#   REDEPLOY_ON_URL_CHANGE=1 setsid nohup demo/run_public.sh > demo/logs/supervisor.log 2>&1 < /dev/null &
#
# - API server: uvicorn on 127.0.0.1:8600, CORS limited to PURSUIT_CORS (optional token: REQUIRE_TOKEN=1).
# - Tunnel: a Cloudflare quick tunnel (no account). Its https://*.trycloudflare.com address changes
#   whenever cloudflared restarts; the current one is written to demo/logs/tunnel_url.
# - REDEPLOY_ON_URL_CHANGE=1 rebuilds the Vercel production deployment with the new address
#   (VITE_API_BASE) using the logged-in Vercel CLI, from $DEPLOY_DIR.
set -u
ROOT=/workspace/cluster
LOGS=$ROOT/demo/logs
mkdir -p "$LOGS"
CLOUDFLARED=${CLOUDFLARED:-/tmp/cloudflared}
DEPLOY_DIR=${DEPLOY_DIR:-/tmp/pursuit-dashboard-deploy/demo/web}
VERCEL_PROJECT=${VERCEL_PROJECT:-express-krypton-ciz5965}
export PURSUIT_CORS=${PURSUIT_CORS:-https://temporary-express-krypton-ciz5965.vercel.app,https://express-krypton-ciz5965.vercel.app,http://127.0.0.1:4173,http://localhost:5173}
# REQUIRE_TOKEN=1: visitors must enter the token from demo/logs/demo_token in the dashboard.
# Default off so the public link works on its own; PURSUIT_MAX_RUNS still caps concurrent simulations.
if [ "${REQUIRE_TOKEN:-0}" = 1 ]; then
  [ -s "$LOGS/demo_token" ] || (umask 077; head -c 18 /dev/urandom | base64 | tr -d '/+=' > "$LOGS/demo_token")
  export PURSUIT_DEMO_TOKEN=$(cat "$LOGS/demo_token")
else
  unset PURSUIT_DEMO_TOKEN
fi

serve() {
  while true; do
    (cd "$ROOT/demo/server" && "$ROOT/.venv/bin/uvicorn" app:app --host 127.0.0.1 --port 8600) >> "$LOGS/server.log" 2>&1
    echo "$(date -Is) server exited; restarting in 3 s" >> "$LOGS/supervisor.log"; sleep 3
  done
}

tunnel() {
  while true; do
    : > "$LOGS/tunnel.log"
    "$CLOUDFLARED" tunnel --no-autoupdate --url http://127.0.0.1:8600 >> "$LOGS/tunnel.log" 2>&1 &
    pid=$!
    url=""
    for _ in $(seq 60); do
      url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1)
      [ -n "$url" ] && break; sleep 1
    done
    if [ -n "$url" ]; then
      old=$(cat "$LOGS/tunnel_url" 2>/dev/null)
      echo "$url" > "$LOGS/tunnel_url"
      echo "$(date -Is) tunnel up: $url" >> "$LOGS/supervisor.log"
      if [ "${REDEPLOY_ON_URL_CHANGE:-0}" = 1 ] && [ "$url" != "$old" ]; then
        (cd "$DEPLOY_DIR" && npx -y vercel@59.26.0 deploy --prod --yes --build-env VITE_API_BASE="$url") >> "$LOGS/deploy.log" 2>&1 \
          && echo "$(date -Is) redeployed with $url" >> "$LOGS/supervisor.log"
      fi
    fi
    wait "$pid"
    echo "$(date -Is) tunnel exited; restarting in 5 s" >> "$LOGS/supervisor.log"; sleep 5
  done
}

serve &
sleep 15
tunnel
