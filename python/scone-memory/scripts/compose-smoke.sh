#!/bin/sh
# Bring the compose stack up, prove it serves: /healthz answers, an
# episode round-trips through Mongo and Qdrant, and the key is enforced.
# Tears the stack down (volumes included) unless KEEP=1.
set -eu
cd "$(dirname "$0")/.."
export SCONE_API_KEY="${SCONE_API_KEY:-smoke-$(date +%s)}"
export SCONE_PORT="${SCONE_PORT:-7437}"
BASE="http://127.0.0.1:${SCONE_PORT}"
cleanup() { [ "${KEEP:-0}" = "1" ] || docker compose down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker compose up -d --build --quiet-pull
for i in $(seq 1 60); do
  curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break
  [ "$i" = 60 ] && { echo "FAIL: /healthz never answered"; docker compose logs scone-memory | tail -20; exit 1; }
  sleep 2
done
echo "1. healthz answers"

code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/status")
[ "$code" = "401" ] || { echo "FAIL: no key gave $code, expected 401"; exit 1; }
echo "2. a request without the key is refused (401)"

auth="Authorization: Bearer $SCONE_API_KEY"
added=$(curl -fsS -H "$auth" -H 'Content-Type: application/json' -d '{"content":"the smoke test moved to Lisbon in March","tags":["smoke"]}' "$BASE/v1/episodes")
id=$(printf '%s' "$added" | sed -n 's/.*"episode_id": *\([0-9]*\).*/\1/p')
[ -n "$id" ] || { echo "FAIL: remember returned $added"; exit 1; }
echo "3. remembered episode $id"

recalled=$(curl -fsS -H "$auth" "$BASE/v1/recall?q=who%20moved%20to%20Lisbon")
printf '%s' "$recalled" | grep -q "\"episode_id\": *$id" || { echo "FAIL: recall did not return episode $id: $recalled"; exit 1; }
echo "4. recalled it through Mongo and Qdrant"

status=$(curl -fsS -H "$auth" "$BASE/v1/status")
printf '%s' "$status" | grep -q '"document_store": *"mongo"' || { echo "FAIL: status says $status"; exit 1; }
printf '%s' "$status" | grep -q '"vector_index": *"qdrant"' || { echo "FAIL: status says $status"; exit 1; }
echo "5. status names mongo and qdrant"
echo "PASS"
