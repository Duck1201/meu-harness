#!/usr/bin/env bash
# Sobe o llama-server do RuntimeProfile da rota, quando ele é llama_cpp (ADR 0013).
#
# O harness não sobe o servidor: ele conversa com um que já está no ar e confere,
# pelo /props, que o arquivo carregado é o GGUF cujo digest o contrato declara e
# que a janela é o num_ctx do perfil. Este script monta a linha de comando com o
# que vem de cada fonte — binário, URL e caminho do GGUF de host.json, argumentos
# do servidor de config/model-profiles.json — para que ninguém a digite à mão e
# erre um -c.
#
# Uso:
#   scripts/llama-server.sh                    # perfil da rota padrão
#   scripts/llama-server.sh <runtime_profile>  # outro perfil (ex.: um Challenger)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOST_JSON=${HOST_JSON:-${XDG_CONFIG_HOME:-$HOME/.config}/harness-2/host.json}

LINES=$(
  python3 - "$ROOT" "$HOST_JSON" "${1:-}" <<'PY'
import json, sys
from pathlib import Path
from urllib.parse import urlsplit

root, host_path, requested = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
harness = json.loads((root / "config/harness.json").read_text())
profiles = json.loads((root / "config/model-profiles.json").read_text())["runtime_profiles"]
route = next(r for r in harness["execution_routes"] if r["id"] == harness["default_execution_route"])
profile_id = requested or route["runtime_profile"]
profile = next((p for p in profiles if p["id"] == profile_id), None)
if profile is None:
    sys.exit(f"RuntimeProfile inexistente: {profile_id}")
if profile["runtime"]["backend"] != "llama_cpp":
    sys.exit(f"{profile_id} é servido pelo {profile['runtime']['backend']}, não pelo llama.cpp")
host = json.loads(host_path.read_text())
executable = host.get("llama_server_executable")
gguf = host.get("gguf_paths", {}).get(profile_id)
if not executable or not gguf:
    sys.exit(f"host.json precisa de llama_server_executable e gguf_paths.{profile_id}")
url = urlsplit(host.get("llama_server_url", "http://127.0.0.1:8081"))
for item in [executable, "-m", gguf, "--host", url.hostname or "127.0.0.1",
             "--port", str(url.port or 8081), "--alias", profile["model"]["id"],
             *profile["installation"].get("server_args", [])]:
    print(item)
PY
)
mapfile -t COMMAND <<<"$LINES"

echo "==> ${COMMAND[*]}"
exec "${COMMAND[@]}"
