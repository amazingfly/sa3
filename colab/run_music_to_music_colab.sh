#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-sa3-m2m-2nndpass}"
GPU="${COLAB_GPU:-T4}"
RUN_TIMEOUT="${SA3_M2M_COLAB_TIMEOUT:-86400}"
CONFIG_PATH="${SA3_M2M_CONFIG:-${ROOT}/colab/music_to_music_config.json}"
PLAN_ROOT="${SA3_M2M_PLAN_DIR:-${ROOT}/outputs/colab_music_to_music/2nndPass}"
GUARDIAN_START_TIMEOUT="${SA3_M2M_GUARDIAN_START_TIMEOUT:-60}"
GUARDIAN_STALE_AFTER="${SA3_M2M_GUARDIAN_STALE_AFTER:-75}"
PROGRESS_STALE_AFTER="${SA3_M2M_PROGRESS_STALE_AFTER:-1800}"
TUNNEL_KEEPALIVE_INTERVAL="${COLAB_TUNNEL_KEEPALIVE_INTERVAL:-45}"
TUNNEL_KEEPALIVE_TIMEOUT="${COLAB_TUNNEL_KEEPALIVE_TIMEOUT:-10}"
FRONTEND_KEEPALIVE_START_TIMEOUT="${COLAB_FRONTEND_KEEPALIVE_START_TIMEOUT:-60}"
REMOTE_ROOT="/content/sa3"
REMOTE_COLAB_DIR="${REMOTE_ROOT}/colab"
REMOTE_CHUNK_ROOT="${REMOTE_ROOT}/upload_chunks"
REMOTE_SOURCE_BUNDLE="${REMOTE_ROOT}/source_flacs.tar"
REMOTE_SOURCE_BUNDLE_MANIFEST="${REMOTE_ROOT}/source_bundle_manifest.json"
REMOTE_SOURCE_EXTRACT_REPORT="${REMOTE_ROOT}/source_bundle_extract.json"
REMOTE_DRIVE_SOURCE_MANIFEST="${REMOTE_ROOT}/drive_source_manifest.json"
REMOTE_DRIVE_SOURCE_REPORT="${REMOTE_ROOT}/drive_source_download.json"
UPLOAD_DIRECT_MAX_BYTES="${SA3_M2M_UPLOAD_DIRECT_MAX_BYTES:-40000000}"
UPLOAD_CHUNK_BYTES="${SA3_M2M_UPLOAD_CHUNK_BYTES:-33554432}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Music-to-music config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${PLAN_ROOT}"
PLAN_JSON="${PLAN_ROOT}/plan.json"
SOURCE_BUNDLE_CHUNK_DIR="${PLAN_ROOT}/source_bundle_chunks"
SOURCE_BUNDLE_MANIFEST="${PLAN_ROOT}/source_bundle_manifest.json"
DRIVE_SOURCE_MANIFEST="${PLAN_ROOT}/drive_source_manifest.json"

python3 - "${CONFIG_PATH}" "${ROOT}" "${PLAN_JSON}" <<'PY'
import copy
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
plan_path = Path(sys.argv[3]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
local = config.get("local", {})
paths = config.get("paths", {})
source = config.get("source") or {}
if not isinstance(source, dict):
    raise SystemExit("source must be an object when provided")
source_mode = str(source.get("mode") or "local_bundle")
if source_mode not in {"local_bundle", "google_drive_folder"}:
    raise SystemExit("source.mode must be 'local_bundle' or 'google_drive_folder'")
drive_folder_url = str(source.get("folder_url") or source.get("drive_folder_url") or "")
drive_folder_id = str(source.get("folder_id") or "")
drive_verify_with_local_manifest = bool(source.get("verify_with_local_manifest", True))
if source_mode == "google_drive_folder" and not (drive_folder_url or drive_folder_id):
    raise SystemExit("source.folder_url or source.folder_id is required for google_drive_folder")


def resolve_local(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else root / path


input_dir = resolve_local(local.get("input_dir"), "musicLibrary/flac")
flac_dir = resolve_local(local.get("flac_dir"), "musicLibrary/2nndPass/flac")
ogg_dir = resolve_local(local.get("ogg_dir"), "musicLibrary/2nndPass/ogg")
report_dir = resolve_local(local.get("report_dir"), "outputs/colab_music_to_music/2nndPass/reports")
log_dir = resolve_local(local.get("log_dir"), "outputs/colab_music_to_music/2nndPass")
pattern = str(config.get("source_pattern") or "*.flac")
skip_existing = bool(config.get("skip_existing", True))
output_suffix = str(local.get("output_suffix") or "")
max_tracks = local.get("max_tracks")

if not input_dir.is_dir():
    raise SystemExit(f"Local source FLAC directory not found: {input_dir}")

source_paths = sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".flac")
if max_tracks not in (None, "", 0):
    source_paths = source_paths[: int(max_tracks)]

tracks = []
for source in source_paths:
    output_name = f"{source.stem}{output_suffix}.flac"
    local_flac = flac_dir / output_name
    local_ogg = ogg_dir / f"{Path(output_name).stem}.ogg"
    local_report = report_dir / f"{Path(output_name).stem}.json"
    if skip_existing and local_flac.exists() and local_ogg.exists():
        continue
    if local_flac.exists() or local_ogg.exists():
        raise SystemExit(
            "Partial or conflicting second-pass output exists; move it before retrying: "
            f"{local_flac if local_flac.exists() else local_ogg}"
        )
    tracks.append(
        {
            "source_path": str(source),
            "source_name": source.name,
            "output_name": output_name,
            "local_flac": str(local_flac),
            "local_ogg": str(local_ogg),
            "local_report": str(local_report),
        }
    )

remote_input_dir = str(paths.get("input_dir") or "/content/sa3/musicLibrary/flac")
remote_flac_dir = str(paths.get("flac_dir") or "/content/outputs/2nndPass/flac")
remote_ogg_dir = str(paths.get("ogg_dir") or "/content/outputs/2nndPass/ogg")
remote_report_dir = str(paths.get("report_dir") or "/content/outputs/2nndPass/reports")
remote_output_root = str(Path(remote_flac_dir).parent)

attempt_config = copy.deepcopy(config)
attempt_config.pop("local", None)
attempt_config["paths"] = {
    "input_dir": remote_input_dir,
    "flac_dir": remote_flac_dir,
    "ogg_dir": remote_ogg_dir,
    "report_dir": remote_report_dir,
}
attempt_config["tracks"] = [
    {
        "source_name": item["source_name"],
        "output_name": item["output_name"],
    }
    for item in tracks
]

log_dir.mkdir(parents=True, exist_ok=True)
flac_dir.mkdir(parents=True, exist_ok=True)
ogg_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
attempt_config_path = log_dir / "music-to-music-attempt-config.json"
attempt_config_path.write_text(json.dumps(attempt_config, indent=2), encoding="utf-8")

plan = {
    "config_path": str(config_path),
    "attempt_config_path": str(attempt_config_path),
    "local_input_dir": str(input_dir),
    "local_flac_dir": str(flac_dir),
    "local_ogg_dir": str(ogg_dir),
    "local_report_dir": str(report_dir),
    "local_log_dir": str(log_dir),
    "remote_input_dir": remote_input_dir,
    "remote_flac_dir": remote_flac_dir,
    "remote_ogg_dir": remote_ogg_dir,
    "remote_report_dir": remote_report_dir,
    "remote_output_root": remote_output_root,
    "source_mode": source_mode,
    "drive_folder_url": drive_folder_url,
    "drive_folder_id": drive_folder_id,
    "drive_verify_with_local_manifest": drive_verify_with_local_manifest,
    "total_source_tracks": len(source_paths),
    "pending_tracks": len(tracks),
    "tracks": tracks,
}
plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(json.dumps(plan, indent=2))
PY

LOCAL_LOG_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_log_dir"])' "${PLAN_JSON}")"
LOCAL_FLAC_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_flac_dir"])' "${PLAN_JSON}")"
LOCAL_OGG_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_ogg_dir"])' "${PLAN_JSON}")"
LOCAL_REPORT_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_report_dir"])' "${PLAN_JSON}")"
ATTEMPT_CONFIG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["attempt_config_path"])' "${PLAN_JSON}")"
REMOTE_INPUT_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_input_dir"])' "${PLAN_JSON}")"
REMOTE_FLAC_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_flac_dir"])' "${PLAN_JSON}")"
REMOTE_OGG_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_ogg_dir"])' "${PLAN_JSON}")"
REMOTE_REPORT_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_report_dir"])' "${PLAN_JSON}")"
REMOTE_OUTPUT_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_output_root"])' "${PLAN_JSON}")"
REMOTE_DRIVE_SOURCE_REPORT="${REMOTE_OUTPUT_ROOT}/drive_source_download.json"
SOURCE_MODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_mode"])' "${PLAN_JSON}")"
DRIVE_FOLDER_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["drive_folder_url"])' "${PLAN_JSON}")"
DRIVE_FOLDER_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["drive_folder_id"])' "${PLAN_JSON}")"
DRIVE_VERIFY_WITH_LOCAL_MANIFEST="$(python3 -c 'import json,sys; print("1" if json.load(open(sys.argv[1])).get("drive_verify_with_local_manifest", True) else "0")' "${PLAN_JSON}")"
PENDING_TRACKS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pending_tracks"])' "${PLAN_JSON}")"

if [[ "${PENDING_TRACKS}" -eq 0 ]]; then
  echo "No pending music-to-music tracks. Existing second-pass outputs were left unchanged."
  exit 0
fi

mapfile -t SOURCE_PATHS < <(python3 -c 'import json,sys; print(*[t["source_path"] for t in json.load(open(sys.argv[1]))["tracks"]], sep="\n")' "${PLAN_JSON}")
mapfile -t SOURCE_NAMES < <(python3 -c 'import json,sys; print(*[t["source_name"] for t in json.load(open(sys.argv[1]))["tracks"]], sep="\n")' "${PLAN_JSON}")
mapfile -t OUTPUT_NAMES < <(python3 -c 'import json,sys; print(*[t["output_name"] for t in json.load(open(sys.argv[1]))["tracks"]], sep="\n")' "${PLAN_JSON}")

mkdir -p "${LOCAL_LOG_DIR}" "${LOCAL_FLAC_DIR}" "${LOCAL_OGG_DIR}" "${LOCAL_REPORT_DIR}"

session_created=0
guardian_pid=""
guardian_log="${LOCAL_LOG_DIR}/colab_guardian.log"
frontend_pid=""
frontend_ready="${LOCAL_LOG_DIR}/colab_frontend_keepalive.ready.json"
frontend_log="${LOCAL_LOG_DIR}/colab_frontend_keepalive.log"
tunnel_keepalive_log="${LOCAL_LOG_DIR}/colab_tunnel_keepalive.log"
last_tunnel_keepalive_at=0

resolve_colab_python() {
  local colab_path shebang
  colab_path="$(command -v colab)"
  shebang="$(head -n 1 "${colab_path}" 2>/dev/null || true)"
  if [[ "${shebang}" == "#!"* ]] && [[ "${shebang}" != *"/usr/bin/env "* ]]; then
    printf '%s\n' "${shebang#\#!}"
  else
    printf '%s\n' "python3"
  fi
}

COLAB_PYTHON="${COLAB_PYTHON:-$(resolve_colab_python)}"

sync_completed_outputs() {
  local output_name stem report_name temporary_report temporary_flac temporary_ogg
  local local_report local_flac local_ogg
  for output_name in "${OUTPUT_NAMES[@]}"; do
    stem="${output_name%.flac}"
    report_name="${stem}.json"
    local_report="${LOCAL_REPORT_DIR}/${report_name}"
    local_flac="${LOCAL_FLAC_DIR}/${output_name}"
    local_ogg="${LOCAL_OGG_DIR}/${stem}.ogg"
    if [[ -s "${local_report}" && -s "${local_flac}" && -s "${local_ogg}" ]]; then
      continue
    fi
    temporary_report="${LOCAL_REPORT_DIR}/.${report_name}.partial"
    temporary_flac="${LOCAL_FLAC_DIR}/.${output_name}.partial"
    temporary_ogg="${LOCAL_OGG_DIR}/.${stem}.ogg.partial"
    rm -f "${temporary_report}" "${temporary_flac}" "${temporary_ogg}"
    if ! timeout 30s colab download -s "${SESSION}" \
      "${REMOTE_REPORT_DIR}/${report_name}" \
      "${temporary_report}" >/dev/null 2>&1; then
      rm -f "${temporary_report}"
      break
    fi
    if ! python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r.get("ok") else 1)' "${temporary_report}"; then
      echo "Remote item report is not successful yet: ${report_name}" >&2
      rm -f "${temporary_report}"
      break
    fi
    if timeout 240s colab download -s "${SESSION}" \
      "${REMOTE_FLAC_DIR}/${output_name}" \
      "${temporary_flac}" >/dev/null 2>&1 \
      && timeout 180s colab download -s "${SESSION}" \
      "${REMOTE_OGG_DIR}/${stem}.ogg" \
      "${temporary_ogg}" >/dev/null 2>&1 \
      && python3 -c \
        'import hashlib,json,os,sys
report,flac,ogg=sys.argv[1:]
r=json.load(open(report))
for key,path in (("flac",flac),("ogg",ogg)):
    expected=r[key]
    with open(path,"rb") as handle:
        actual=hashlib.file_digest(handle,"sha256").hexdigest()
    assert os.path.getsize(path)==int(expected["bytes"]), (key, path)
    assert actual==expected["sha256"], (key, path)' \
        "${temporary_report}" "${temporary_flac}" "${temporary_ogg}"; then
      mv "${temporary_report}" "${local_report}"
      mv "${temporary_flac}" "${local_flac}"
      mv "${temporary_ogg}" "${local_ogg}"
      downloaded_any=1
      echo "Checkpointed second-pass FLAC: ${local_flac}"
      echo "Checkpointed second-pass OGG: ${local_ogg}"
      continue
    fi
    echo "Discarded incomplete second-pass checkpoint: ${output_name}" >&2
    rm -f "${temporary_report}" "${temporary_flac}" "${temporary_ogg}"
    break
  done
}

run_colab_exec_watched() {
  local label="$1"
  local wall_timeout="$2"
  shift 2

  local command_log command_pid return_code
  command_log="$(mktemp "${LOCAL_LOG_DIR}/.colab-exec.XXXXXX.log")"
  timeout --foreground "${wall_timeout}s" \
    colab exec -s "${SESSION}" "$@" > >(tee "${command_log}") 2>&1 &
  command_pid=$!

  while kill -0 "${command_pid}" 2>/dev/null; do
    if grep -Fq "Connection was lost." "${command_log}"; then
      echo "${label}: Colab CLI connection was lost; aborting this attempt." >&2
      kill "${command_pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${command_pid}" 2>/dev/null || true
      wait "${command_pid}" 2>/dev/null || true
      rm -f "${command_log}"
      return 1
    fi
    sleep 2
  done

  if wait "${command_pid}"; then
    return_code=0
  else
    return_code=$?
  fi
  rm -f "${command_log}"
  return "${return_code}"
}

run_colab_file_with_retries() {
  local label="$1"
  local file="$2"
  local remote_timeout="$3"
  local wall_timeout="$4"
  local attempt

  for attempt in 1 2 3; do
    if run_colab_exec_watched \
      "${label}" \
      "${wall_timeout}" \
      -f "${file}" \
      --timeout "${remote_timeout}"; then
      return 0
    fi
    echo "${label} attempt ${attempt}/3 failed; reconnecting." >&2
    sleep 5
  done
  return 1
}

run_remote_code_with_retries() {
  local label="$1"
  local remote_timeout="$2"
  local wall_timeout="$3"
  local code="$4"
  local attempt

  for attempt in 1 2 3; do
    send_tunnel_keepalive || true
    if printf '%s\n' "${code}" \
      | timeout --foreground "${wall_timeout}s" \
        colab exec -s "${SESSION}" --timeout "${remote_timeout}"; then
      return 0
    fi
    echo "${label} attempt ${attempt}/3 failed; retrying." >&2
    sleep 5
  done
  return 1
}

colab_upload_with_retries() {
  local source_path="$1"
  local remote_path="$2"
  local attempt

  for attempt in 1 2 3; do
    send_tunnel_keepalive || true
    if timeout --foreground 600s \
      colab upload -s "${SESSION}" "${source_path}" "${remote_path}"; then
      return 0
    fi
    echo "Upload attempt ${attempt}/3 failed for ${source_path}; retrying." >&2
    sleep 5
  done
  return 1
}

send_tunnel_keepalive() {
  [[ "${COLAB_TUNNEL_KEEPALIVE:-1}" == "1" ]] || return 0
  local now config_args=()
  now="$(date +%s)"
  if (( now - last_tunnel_keepalive_at < TUNNEL_KEEPALIVE_INTERVAL )); then
    return 0
  fi
  last_tunnel_keepalive_at="${now}"
  if [[ -n "${COLAB_CONFIG:-}" ]]; then
    config_args=(--config "${COLAB_CONFIG}")
  fi
  if ! "${COLAB_PYTHON}" "${ROOT}/scripts/colab_tunnel_keepalive.py" \
    --session "${SESSION}" \
    --authuser "${COLAB_AUTHUSER:-0}" \
    --auth-provider "${COLAB_AUTH_PROVIDER:-oauth2}" \
    --request-timeout "${TUNNEL_KEEPALIVE_TIMEOUT}" \
    "${config_args[@]}" \
    >>"${tunnel_keepalive_log}" 2>&1; then
    echo "Colab tunnel keep-alive ping failed; continuing. See ${tunnel_keepalive_log}" >&2
    return 1
  fi
}

open_frontend_url() {
  local frontend_url="$1"
  echo "Opening the exact CLI runtime in the Colab frontend for keep-alive"
  python3 - "${frontend_url}" "${COLAB_AUTHUSER:-0}" <<'PY'
import sys
import urllib.parse
import webbrowser

url, authuser = sys.argv[1:]
parts = urllib.parse.urlsplit(url)
query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
query = [(key, value) for key, value in query if key != "authuser"]
query.append(("authuser", authuser))
webbrowser.open(
    urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )
)
PY
}

start_frontend_keepalive() {
  [[ "${COLAB_OPEN_FRONTEND:-1}" == "1" ]] || return 0
  local frontend_url started_at
  frontend_url="$(colab url -s "${SESSION}")"
  if [[ "${COLAB_FRONTEND_KEEPALIVE:-1}" != "1" ]]; then
    open_frontend_url "${frontend_url}"
    return 0
  fi

  rm -f "${frontend_ready}" "${frontend_log}"
  echo "Starting authenticated Colab frontend keep-alive"
  python3 "${ROOT}/scripts/colab_frontend_keepalive.py" \
    --url "${frontend_url}" \
    --authuser "${COLAB_AUTHUSER:-0}" \
    --ready-file "${frontend_ready}" \
    >"${frontend_log}" 2>&1 &
  frontend_pid=$!
  started_at="$(date +%s)"
  while [[ ! -s "${frontend_ready}" ]]; do
    if ! kill -0 "${frontend_pid}" 2>/dev/null; then
      wait "${frontend_pid}" >/dev/null 2>&1 || true
      frontend_pid=""
      echo "Colab frontend keep-alive failed to start; falling back to browser open. See ${frontend_log}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    if (( $(date +%s) - started_at > FRONTEND_KEEPALIVE_START_TIMEOUT )); then
      kill "${frontend_pid}" >/dev/null 2>&1 || true
      wait "${frontend_pid}" >/dev/null 2>&1 || true
      frontend_pid=""
      echo "Timed out starting Colab frontend keep-alive; falling back to browser open. See ${frontend_log}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    sleep 2
  done
  echo "Colab frontend keep-alive is active"
}

if [[ ! -s "${HOME}/.config/colab-cli/token.json" ]]; then
  OAUTHLIB_RELAX_TOKEN_SCOPE=1 "${COLAB_PYTHON}" "${ROOT}/colab/auth_colab.py"
fi

cleanup() {
  if [[ -n "${guardian_pid}" ]]; then
    kill "${guardian_pid}" >/dev/null 2>&1 || true
    wait "${guardian_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${frontend_pid}" ]]; then
    kill "${frontend_pid}" >/dev/null 2>&1 || true
    wait "${frontend_pid}" >/dev/null 2>&1 || true
  fi
  if [[ "${session_created}" -eq 1 ]]; then
    echo "Stopping Colab session ${SESSION}"
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  fi
  if [[ -f "${HOME}/.config/colab-cli/token.json" ]]; then
    chmod 600 "${HOME}/.config/colab-cli/token.json"
  fi
}
trap cleanup EXIT

echo "Creating Colab ${GPU} session: ${SESSION}"
existing_status="$(colab status -s "${SESSION}" 2>&1 || true)"
if [[ "${existing_status}" == *"Status:"* ]] \
  && [[ "${existing_status}" != *"not found"* ]] \
  && [[ "${existing_status}" != *"No active sessions"* ]]; then
  echo "Reusing existing Colab session: ${SESSION}"
else
  session_ready=0
  for attempt in 1 2 3 4 5; do
    if colab new -s "${SESSION}" --gpu "${GPU}"; then
      session_ready=1
      break
    fi
    echo "Colab allocation attempt ${attempt}/5 failed; retrying in 30s." >&2
    sleep 30
  done
  if [[ "${session_ready}" -ne 1 ]]; then
    echo "Could not allocate a Colab ${GPU} runtime." >&2
    exit 75
  fi
fi
session_created=1
colab status -s "${SESSION}"
send_tunnel_keepalive || true
start_frontend_keepalive

runtime_ready=0
for attempt in 1 2 3; do
  if run_remote_code_with_retries \
    "Colab runtime directory setup" \
    60 \
    75 \
    "$(printf '%s\n' \
      "from pathlib import Path" \
      "for value in ['${REMOTE_COLAB_DIR}', '${REMOTE_INPUT_DIR}', '${REMOTE_FLAC_DIR}', '${REMOTE_OGG_DIR}', '${REMOTE_REPORT_DIR}', '${REMOTE_OUTPUT_ROOT}', '${REMOTE_CHUNK_ROOT}']:" \
      "    Path(value).mkdir(parents=True, exist_ok=True)" \
      "for part in Path('${REMOTE_CHUNK_ROOT}').glob('source_flacs.tar.part-*'):" \
      "    part.unlink(missing_ok=True)" \
      "for stale in ['${REMOTE_SOURCE_BUNDLE}', '${REMOTE_SOURCE_BUNDLE_MANIFEST}', '${REMOTE_SOURCE_EXTRACT_REPORT}', '${REMOTE_DRIVE_SOURCE_MANIFEST}', '${REMOTE_DRIVE_SOURCE_REPORT}']:" \
      "    Path(stale).unlink(missing_ok=True)")"; then
    runtime_ready=1
    break
  fi
  echo "Colab kernel connection attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
if [[ "${runtime_ready}" -ne 1 ]]; then
  echo "Could not establish a stable Colab kernel connection." >&2
  exit 1
fi

for name in requirements.txt setup_colab.py probe_colab.py run_generate.py run_music_to_music.py launch_music_to_music.py; do
  colab_upload_with_retries \
    "${ROOT}/colab/${name}" \
    "${REMOTE_COLAB_DIR}/${name}"
done
colab_upload_with_retries \
  "${ATTEMPT_CONFIG}" \
  "${REMOTE_COLAB_DIR}/music_to_music_config.json"
colab_upload_with_retries \
  "${ATTEMPT_CONFIG}" \
  "${REMOTE_COLAB_DIR}/generation_config.json"

LOCAL_HF_TOKEN="${HOME}/.cache/huggingface/token"
if [[ -s "${LOCAL_HF_TOKEN}" ]]; then
  colab_upload_with_retries \
    "${LOCAL_HF_TOKEN}" \
    "${REMOTE_ROOT}/hf_token" >/dev/null
  echo "Uploaded Hugging Face credentials for authenticated model download"
fi

run_colab_file_with_retries \
  "Colab setup" \
  "${ROOT}/colab/setup_colab.py" \
  1800 \
  1860

kernel_restarted=0
for attempt in 1 2 3; do
  if timeout --foreground 120s colab restart-kernel -s "${SESSION}"; then
    kernel_restarted=1
    break
  fi
  echo "Colab kernel restart attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
if [[ "${kernel_restarted}" -ne 1 ]]; then
  echo "Could not restart the Colab kernel after setup." >&2
  exit 1
fi

run_colab_file_with_retries \
  "Colab probe" \
  "${ROOT}/colab/probe_colab.py" \
  300 \
  360

create_drive_source_manifest() {
  python3 - "${PLAN_JSON}" "${DRIVE_SOURCE_MANIFEST}" "${DRIVE_FOLDER_URL}" "${DRIVE_FOLDER_ID}" "${DRIVE_VERIFY_WITH_LOCAL_MANIFEST}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

plan_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
folder_url = sys.argv[3]
folder_id = sys.argv[4]
verify_with_local_manifest = sys.argv[5] == "1"
plan = json.loads(plan_path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = []
for track in plan["tracks"]:
    source = Path(track["source_path"])
    name = track["source_name"]
    if Path(name).name != name or not name.lower().endswith(".flac"):
        raise SystemExit(f"unsafe source name: {name}")
    if not source.is_file():
        raise SystemExit(f"local source missing for Drive verification manifest: {source}")
    item = {"name": name}
    if verify_with_local_manifest:
        item["bytes"] = source.stat().st_size
        item["sha256"] = sha256_file(source)
    files.append(item)

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_mode": "google_drive_folder",
    "folder_url": folder_url,
    "folder_id": folder_id,
    "verify_with_local_manifest": verify_with_local_manifest,
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
verification = "SHA-256" if verify_with_local_manifest else "filename-only"
print(f"Created Drive source verification manifest: {len(files)} files ({verification})")
PY
}

download_drive_sources_remote() {
  local script remote_status local_report temporary_report
  colab_upload_with_retries \
    "${DRIVE_SOURCE_MANIFEST}" \
    "${REMOTE_DRIVE_SOURCE_MANIFEST}"
  script="$(
    python3 - "${REMOTE_DRIVE_SOURCE_MANIFEST}" "${REMOTE_DRIVE_SOURCE_REPORT}" "${REMOTE_INPUT_DIR}" <<'PY'
import json
import sys

manifest_path, report_path, input_dir = sys.argv[1:]
print(
    "\n".join(
        [
            "from pathlib import Path",
            "import hashlib, json, shutil, time",
            "import gdown",
            f"manifest_path = Path({json.dumps(manifest_path)})",
            f"report_path = Path({json.dumps(report_path)})",
            f"input_dir = Path({json.dumps(input_dir)})",
            "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))",
            "folder_url = manifest.get('folder_url') or None",
            "folder_id = manifest.get('folder_id') or None",
            "expected_files = {item['name']: item for item in manifest['files']}",
            "def sha256_file(path):",
            "    digest = hashlib.sha256()",
            "    with path.open('rb') as handle:",
            "        for chunk in iter(lambda: handle.read(1024 * 1024), b''):",
            "            digest.update(chunk)",
            "    return digest.hexdigest()",
            "def normalize_downloaded_flacs():",
            "    input_dir.mkdir(parents=True, exist_ok=True)",
            "    for path in list(input_dir.rglob('*.flac')):",
            "        target = input_dir / path.name",
            "        if path == target:",
            "            continue",
            "        if path.name not in expected_files:",
            "            continue",
            "        if target.exists():",
            "            if target.stat().st_size == path.stat().st_size:",
            "                path.unlink(missing_ok=True)",
            "                continue",
            "            raise RuntimeError(f'Conflicting downloaded FLAC: {target}')",
            "        shutil.move(str(path), str(target))",
            "    for directory in sorted([p for p in input_dir.rglob('*') if p.is_dir()], key=lambda p: len(p.parts), reverse=True):",
            "        try:",
            "            directory.rmdir()",
            "        except OSError:",
            "            pass",
            "def verify_sources():",
            "    normalize_downloaded_flacs()",
            "    missing = []",
            "    mismatched = []",
            "    for name, item in expected_files.items():",
            "        path = input_dir / name",
            "        if not path.is_file():",
            "            missing.append(name)",
            "            continue",
            "        size = path.stat().st_size",
            "        if 'bytes' in item and size != int(item['bytes']):",
            "            mismatched.append({'name': name, 'reason': 'bytes', 'actual': size, 'expected': int(item['bytes'])})",
            "            continue",
            "        if 'sha256' not in item:",
            "            continue",
            "        digest = sha256_file(path)",
            "        if digest != item['sha256']:",
            "            mismatched.append({'name': name, 'reason': 'sha256', 'actual': digest, 'expected': item['sha256']})",
            "    return missing, mismatched",
            "input_dir.mkdir(parents=True, exist_ok=True)",
            "missing, mismatched = verify_sources()",
            "if not missing and not mismatched:",
            "    report = {'ok': True, 'reused_existing_files': True, 'files': len(expected_files), 'input_dir': str(input_dir)}",
            "    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')",
            "    print(f'Drive source FLACs already present and verified: {len(expected_files)} files')",
            "    raise SystemExit(0)",
            "print(f'Downloading source FLAC folder from Google Drive: {folder_url or folder_id}', flush=True)",
            "download_kwargs = {",
            "    'output': str(input_dir),",
            "    'quiet': False,",
            "    'use_cookies': False,",
            "    'remaining_ok': True,",
            "    'resume': True,",
            "}",
            "if folder_url:",
            "    download_kwargs['url'] = folder_url",
            "else:",
            "    download_kwargs['id'] = folder_id",
            "downloaded = gdown.download_folder(**download_kwargs)",
            "missing, mismatched = verify_sources()",
            "report = {",
            "    'ok': not missing and not mismatched,",
            "    'created_at': time.time(),",
            "    'folder_url': folder_url,",
            "    'folder_id': folder_id,",
            "    'input_dir': str(input_dir),",
            "    'expected_files': len(expected_files),",
            "    'downloaded_count': len(downloaded or []),",
            "    'missing': missing,",
            "    'mismatched': mismatched,",
            "}",
            "report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')",
            "if missing or mismatched:",
            "    raise RuntimeError(f'Drive source verification failed: missing={len(missing)}, mismatched={len(mismatched)}; report={report_path}')",
            "print(f'Downloaded and verified Drive source FLACs: {len(expected_files)} files')",
        ]
    )
)
PY
  )"
  remote_status=0
  run_remote_code_with_retries \
    "Download and verify source FLACs from Google Drive" \
    3600 \
    3660 \
    "${script}" || remote_status=$?

  local_report="${LOCAL_LOG_DIR}/drive_source_download.json"
  temporary_report="${local_report}.partial"
  rm -f "${temporary_report}"
  if ! timeout 120s colab download -s "${SESSION}" \
    "${REMOTE_DRIVE_SOURCE_REPORT}" \
    "${temporary_report}" >/dev/null 2>&1; then
    echo "Could not download Drive source verification report from Colab: ${REMOTE_DRIVE_SOURCE_REPORT}" >&2
    return 1
  fi
  mv "${temporary_report}" "${local_report}"

  if ! python3 - "${local_report}" <<'PY'; then
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("ok"):
    print(
        "Drive source verification passed: "
        f"{report.get('files') or report.get('expected_files')} files"
    )
    raise SystemExit(0)

missing = report.get("missing") or []
mismatched = report.get("mismatched") or []
print(
    "Drive source verification failed: "
    f"missing={len(missing)}, mismatched={len(mismatched)}; report={report_path}",
    file=sys.stderr,
)
if missing:
    print("Missing source files:", file=sys.stderr)
    for name in missing[:20]:
        print(f"  {name}", file=sys.stderr)
    if len(missing) > 20:
        print(f"  ... {len(missing) - 20} more", file=sys.stderr)
if mismatched:
    print("Mismatched source files:", file=sys.stderr)
    for item in mismatched[:20]:
        print(f"  {item}", file=sys.stderr)
    if len(mismatched) > 20:
        print(f"  ... {len(mismatched) - 20} more", file=sys.stderr)
raise SystemExit(1)
PY
    return 1
  fi

  if [[ "${remote_status}" -ne 0 ]]; then
    echo "Drive source download command failed even though a report was written; not continuing." >&2
    return "${remote_status}"
  fi
}

create_source_bundle_chunks() {
  rm -rf "${SOURCE_BUNDLE_CHUNK_DIR}"
  mkdir -p "${SOURCE_BUNDLE_CHUNK_DIR}"
  python3 - "${PLAN_JSON}" "${SOURCE_BUNDLE_CHUNK_DIR}" "${UPLOAD_CHUNK_BYTES}" "${SOURCE_BUNDLE_MANIFEST}" <<'PY'
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

plan_path = Path(sys.argv[1])
chunk_dir = Path(sys.argv[2])
chunk_size = int(sys.argv[3])
manifest_path = Path(sys.argv[4])
plan = json.loads(plan_path.read_text(encoding="utf-8"))

if chunk_size <= 0:
    raise SystemExit("chunk size must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChunkWriter(io.RawIOBase):
    def __init__(self, target_dir: Path, part_size: int) -> None:
        self.target_dir = target_dir
        self.part_size = part_size
        self.part_index = 0
        self.part_offset = 0
        self.current = None
        self.total_bytes = 0
        self.digest = hashlib.sha256()
        self.parts: list[dict[str, object]] = []

    def writable(self) -> bool:
        return True

    def _open_next(self) -> None:
        self.close_current()
        part_name = f"source_flacs.tar.part-{self.part_index:04d}"
        self.current_path = self.target_dir / part_name
        self.current = self.current_path.open("wb")
        self.part_offset = 0
        self.part_index += 1

    def close_current(self) -> None:
        if self.current is not None:
            self.current.close()
            self.parts.append(
                {
                    "name": self.current_path.name,
                    "bytes": self.current_path.stat().st_size,
                }
            )
            self.current = None

    def write(self, data) -> int:
        view = memoryview(data)
        written = len(view)
        while view:
            if self.current is None:
                self._open_next()
            capacity = self.part_size - self.part_offset
            chunk = view[:capacity]
            self.current.write(chunk)
            self.digest.update(chunk)
            self.total_bytes += len(chunk)
            self.part_offset += len(chunk)
            view = view[len(chunk):]
            if self.part_offset >= self.part_size:
                self.close_current()
        return written

    def close(self) -> None:
        self.close_current()
        super().close()


files = []
for track in plan["tracks"]:
    source = Path(track["source_path"])
    name = track["source_name"]
    if Path(name).name != name or not name.lower().endswith(".flac"):
        raise SystemExit(f"unsafe source name: {name}")
    files.append(
        {
            "path": str(source),
            "name": name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )

writer = ChunkWriter(chunk_dir, chunk_size)
with tarfile.open(fileobj=writer, mode="w|") as archive:
    for item in files:
        path = Path(item["path"])
        info = archive.gettarinfo(str(path), arcname=item["name"])
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        with path.open("rb") as handle:
            archive.addfile(info, handle)
writer.close()

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "chunk_size": chunk_size,
    "bundle": {
        "bytes": writer.total_bytes,
        "sha256": writer.digest.hexdigest(),
        "part_count": len(writer.parts),
        "parts": writer.parts,
    },
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(
    f"Created source bundle chunks: {len(files)} files, "
    f"{writer.total_bytes} tar bytes, {len(writer.parts)} parts"
)
PY
}

reassemble_source_bundle_remote() {
  local script
  script="$(
    python3 - "${REMOTE_CHUNK_ROOT}" "${REMOTE_SOURCE_BUNDLE}" "${REMOTE_SOURCE_BUNDLE_MANIFEST}" "${REMOTE_SOURCE_EXTRACT_REPORT}" "${REMOTE_INPUT_DIR}" <<'PY'
import json
import sys

chunk_root, bundle, manifest_path, report_path, input_dir = sys.argv[1:]
print(
    "\n".join(
        [
            "from pathlib import Path",
            "import hashlib, json, os, shutil, tarfile, time",
            f"chunk_root = Path({json.dumps(chunk_root)})",
            f"bundle = Path({json.dumps(bundle)})",
            f"manifest_path = Path({json.dumps(manifest_path)})",
            f"report_path = Path({json.dumps(report_path)})",
            f"input_dir = Path({json.dumps(input_dir)})",
            "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))",
            "expected_files = {item['name']: item for item in manifest['files']}",
            "def sha256_file(path):",
            "    digest = hashlib.sha256()",
            "    with path.open('rb') as handle:",
            "        for chunk in iter(lambda: handle.read(1024 * 1024), b''):",
            "            digest.update(chunk)",
            "    return digest.hexdigest()",
            "def verify_extracted():",
            "    input_dir.mkdir(parents=True, exist_ok=True)",
            "    for name, item in expected_files.items():",
            "        path = input_dir / name",
            "        if not path.is_file():",
            "            return False",
            "        if path.stat().st_size != int(item['bytes']):",
            "            return False",
            "        if sha256_file(path) != item['sha256']:",
            "            return False",
            "    return True",
            "if report_path.exists():",
            "    try:",
            "        report = json.loads(report_path.read_text(encoding='utf-8'))",
            "    except Exception:",
            "        report = {}",
            "    if report.get('ok') and verify_extracted():",
            "        print(f'Source bundle already extracted and verified: {len(expected_files)} files')",
            "        raise SystemExit(0)",
            "parts = sorted(chunk_root.glob('source_flacs.tar.part-*'))",
            "if not parts:",
            "    if verify_extracted():",
            "        report_path.write_text(json.dumps({'ok': True, 'reused_extracted_files': True, 'files': len(expected_files)}, indent=2), encoding='utf-8')",
            "        print(f'Source files already present and verified: {len(expected_files)} files')",
            "        raise SystemExit(0)",
            "    raise RuntimeError(f'No source bundle parts found in {chunk_root}')",
            "expected_part_count = int(manifest['bundle']['part_count'])",
            "if len(parts) != expected_part_count:",
            "    raise RuntimeError(f'Expected {expected_part_count} source bundle parts, found {len(parts)}')",
            "temporary = bundle.with_name(bundle.name + '.partial')",
            "temporary.unlink(missing_ok=True)",
            "digest = hashlib.sha256()",
            "total = 0",
            "with temporary.open('wb') as output:",
            "    for part in parts:",
            "        with part.open('rb') as handle:",
            "            for chunk in iter(lambda: handle.read(1024 * 1024), b''):",
            "                digest.update(chunk)",
            "                total += len(chunk)",
            "                output.write(chunk)",
            "actual_sha = digest.hexdigest()",
            "expected_bytes = int(manifest['bundle']['bytes'])",
            "expected_sha = manifest['bundle']['sha256']",
            "if total != expected_bytes or actual_sha != expected_sha:",
            "    raise RuntimeError(f'Source bundle verification failed: bytes={total}, sha={actual_sha}')",
            "os.replace(temporary, bundle)",
            "input_dir.mkdir(parents=True, exist_ok=True)",
            "with tarfile.open(bundle, 'r:') as archive:",
            "    members = archive.getmembers()",
            "    member_names = {member.name for member in members}",
            "    if member_names != set(expected_files):",
            "        raise RuntimeError(f'Tar member mismatch: {sorted(member_names ^ set(expected_files))[:5]}')",
            "    for member in members:",
            "        if Path(member.name).name != member.name or member.isdir():",
            "            raise RuntimeError(f'Unsafe tar member: {member.name}')",
            "    archive.extractall(input_dir, members=members, filter='data')",
            "if not verify_extracted():",
            "    raise RuntimeError('Extracted source FLAC verification failed')",
            "report = {'ok': True, 'created_at': time.time(), 'bundle_bytes': total, 'bundle_sha256': actual_sha, 'files': len(expected_files)}",
            "report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')",
            "for part in parts:",
            "    part.unlink(missing_ok=True)",
            "bundle.unlink(missing_ok=True)",
            "print(f'Reassembled, extracted, and verified source bundle: {len(expected_files)} files, {total} bytes')",
        ]
    )
)
PY
  )"
  run_remote_code_with_retries \
    "Reassemble and extract source FLAC bundle" \
    1200 \
    1260 \
    "${script}"
}

if [[ "${SOURCE_MODE}" == "google_drive_folder" ]]; then
  echo "Using Google Drive source folder for ${#SOURCE_PATHS[@]} source FLAC file(s)"
  echo "Drive folder: ${DRIVE_FOLDER_URL:-${DRIVE_FOLDER_ID}}"
  create_drive_source_manifest
  download_drive_sources_remote
else
  echo "Bundling ${#SOURCE_PATHS[@]} source FLAC file(s) for Colab upload"
  create_source_bundle_chunks
  colab_upload_with_retries \
    "${SOURCE_BUNDLE_MANIFEST}" \
    "${REMOTE_SOURCE_BUNDLE_MANIFEST}"

  mapfile -t SOURCE_BUNDLE_PARTS < <(find "${SOURCE_BUNDLE_CHUNK_DIR}" -maxdepth 1 -type f -name 'source_flacs.tar.part-*' | sort)
  echo "Uploading source bundle in ${#SOURCE_BUNDLE_PARTS[@]} chunk(s)"
  for index in "${!SOURCE_BUNDLE_PARTS[@]}"; do
    send_tunnel_keepalive || true
    part="${SOURCE_BUNDLE_PARTS[$index]}"
    echo "Uploading source bundle chunk $((index + 1))/${#SOURCE_BUNDLE_PARTS[@]}: $(basename "${part}")"
    colab_upload_with_retries \
      "${part}" \
      "${REMOTE_CHUNK_ROOT}/$(basename "${part}")"
  done
  reassemble_source_bundle_remote
  rm -rf "${SOURCE_BUNDLE_CHUNK_DIR}"
fi

worker_report="${LOCAL_LOG_DIR}/.music_to_music_worker.json"
rm -f "${worker_report}"
worker_launched=0
for attempt in 1 2 3; do
  run_colab_file_with_retries \
    "Colab music-to-music launch" \
    "${ROOT}/colab/launch_music_to_music.py" \
    60 \
    90 || true
  rm -f "${worker_report}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_ROOT}/music_to_music_worker.json" \
    "${worker_report}" >/dev/null 2>&1; then
    echo "Verified detached Colab music-to-music worker launch."
    worker_launched=1
    break
  fi
  echo "Colab music-to-music launch attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
rm -f "${worker_report}"
if [[ "${worker_launched}" -ne 1 ]]; then
  echo "Could not launch the detached Colab music-to-music worker." >&2
  exit 1
fi

start_guardian() {
  local guardian_started_at
  rm -f "${guardian_log}"
  {
    printf '%s\n' \
      "import time" \
      "from pathlib import Path" \
      "exit_report = Path('${REMOTE_OUTPUT_ROOT}/music_to_music_exit.json')" \
      "while not exit_report.exists():" \
      "    print('SA3 music-to-music allocation guardian active', flush=True)" \
      "    time.sleep(20)"
  } | timeout --foreground "${RUN_TIMEOUT}s" \
    colab exec -s "${SESSION}" --timeout "${RUN_TIMEOUT}" \
    >"${guardian_log}" 2>&1 &
  guardian_pid=$!
  guardian_started_at="$(date +%s)"
  while ! grep -q "SA3 music-to-music allocation guardian active" "${guardian_log}" 2>/dev/null; do
    if ! kill -0 "${guardian_pid}" 2>/dev/null; then
      wait "${guardian_pid}" || true
      cat "${guardian_log}" >&2 || true
      echo "Colab allocation guardian failed to start." >&2
      return 1
    fi
    if (( $(date +%s) - guardian_started_at > GUARDIAN_START_TIMEOUT )); then
      kill "${guardian_pid}" >/dev/null 2>&1 || true
      wait "${guardian_pid}" >/dev/null 2>&1 || true
      cat "${guardian_log}" >&2 || true
      echo "Colab allocation guardian did not produce a heartbeat." >&2
      return 1
    fi
    sleep 2
  done
  echo "Colab allocation guardian is active"
}

start_guardian

console_offset=0
console_progress=0
sync_music_console() {
  local temporary size
  console_progress=0
  temporary="${LOCAL_LOG_DIR}/.music_to_music_console.log.partial"
  rm -f "${temporary}"
  if ! timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_ROOT}/music_to_music_console.log" \
    "${temporary}" >/dev/null 2>&1; then
    rm -f "${temporary}"
    return
  fi
  size="$(stat -c '%s' "${temporary}")"
  if (( size > console_offset )); then
    tail -c "+$((console_offset + 1))" "${temporary}"
    console_offset="${size}"
    console_progress=1
  fi
  mv "${temporary}" "${LOCAL_LOG_DIR}/music_to_music_console.log"
}

return_code=""
started_at="$(date +%s)"
last_progress_at="${started_at}"
while [[ -z "${return_code}" ]]; do
  sleep 15
  send_tunnel_keepalive || true
  sync_music_console
  if [[ "${console_progress}" -eq 1 ]]; then
    last_progress_at="$(date +%s)"
  fi
  downloaded_any=0
  sync_completed_outputs
  if [[ "${downloaded_any}" -eq 1 ]]; then
    last_progress_at="$(date +%s)"
  fi
  status_output="$(colab status -s "${SESSION}" 2>&1 || true)"
  if [[ "${status_output}" != *"Status:"* ]] \
    || [[ "${status_output}" == *"not found"* ]] \
    || [[ "${status_output}" == *"No active sessions"* ]]; then
    failure_log="${LOCAL_LOG_DIR}/colab_runtime_error.log"
    {
      echo "Colab runtime disappeared during music-to-music generation."
      echo "Detected at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "Last Colab status: ${status_output}"
    } | tee "${failure_log}" >&2
    exit 1
  fi
  exit_temporary="${LOCAL_LOG_DIR}/.music_to_music_exit.json.partial"
  rm -f "${exit_temporary}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_ROOT}/music_to_music_exit.json" \
    "${exit_temporary}" >/dev/null 2>&1; then
    return_code="$(
      python3 -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["return_code"]))' \
        "${exit_temporary}"
    )"
    mv "${exit_temporary}" "${LOCAL_LOG_DIR}/music_to_music_exit.json"
  fi
  if [[ -z "${return_code}" ]] \
    && { ! kill -0 "${guardian_pid}" 2>/dev/null \
      || (( $(date +%s) - $(stat -c '%Y' "${guardian_log}" 2>/dev/null || echo 0) \
        > GUARDIAN_STALE_AFTER )); }; then
    kill "${guardian_pid}" >/dev/null 2>&1 || true
    wait "${guardian_pid}" >/dev/null 2>&1 || true
    echo "Colab allocation guardian is missing or stale; reconnecting." >&2
    if ! start_guardian; then
      echo "The Colab assignment expired before music-to-music generation completed." >&2
      exit 75
    fi
  fi
  if (( $(date +%s) - started_at > RUN_TIMEOUT )); then
    echo "Detached Colab music-to-music worker exceeded ${RUN_TIMEOUT} seconds." >&2
    exit 124
  fi
  if (( $(date +%s) - last_progress_at > PROGRESS_STALE_AFTER )); then
    echo \
      "SA3 music-to-music made no console or output progress for " \
      "${PROGRESS_STALE_AFTER} seconds; restarting this Colab attempt." >&2
    exit 76
  fi
  echo "Colab music-to-music heartbeat: guardian and artifact sync active; ${status_output}"
done

sync_music_console
sync_completed_outputs

for name in music_to_music.json music_to_music_error.log music_to_music_worker.json drive_source_download.json setup.json probe.json; do
  temporary="${LOCAL_LOG_DIR}/.${name}.partial"
  rm -f "${temporary}"
  remote_path="${REMOTE_OUTPUT_ROOT}/${name}"
  if [[ "${name}" == "setup.json" || "${name}" == "probe.json" ]]; then
    remote_path="/content/outputs/${name}"
  fi
  if timeout 60s colab download -s "${SESSION}" \
    "${remote_path}" \
    "${temporary}" >/dev/null 2>&1; then
    mv "${temporary}" "${LOCAL_LOG_DIR}/${name}"
  else
    rm -f "${temporary}"
  fi
done

if [[ "${return_code}" -ne 0 ]]; then
  echo "Detached Colab music-to-music worker exited with code ${return_code}." >&2
  exit "${return_code}"
fi

missing=0
for output_name in "${OUTPUT_NAMES[@]}"; do
  stem="${output_name%.flac}"
  if [[ -s "${LOCAL_FLAC_DIR}/${output_name}" && -s "${LOCAL_OGG_DIR}/${stem}.ogg" ]]; then
    echo "Downloaded second-pass pair: ${output_name}"
  else
    echo "Second-pass output missing or failed: ${output_name}" >&2
    missing=1
  fi
done

exit "${missing}"
