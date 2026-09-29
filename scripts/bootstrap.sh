#!/usr/bin/env bash
# Prepara um host novo para rodar o harness: deps Python/web, perfil Ollama e
# tokenizer.json. Idempotente — pode rodar de novo sem duplicar trabalho.
# Depois dele, só falta abrir http://127.0.0.1:8765/setup com `uv run harness`.
#
# O perfil vem do contrato: tag, Modelfile, modelo base, digest e tokenizer são os
# do `active_runtime_profile` de config/model-profiles.json. Trocar o perfil
# funcional não pede mudança aqui.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> uv sync"
uv sync

if command -v corepack >/dev/null; then
  echo "==> build do frontend (web/dist)"
  corepack enable
  (cd web && pnpm install && pnpm build)
else
  echo "==> corepack não encontrado, pulando build do frontend" >&2
fi

read -r profile_id tag modelfile base expected_digest tokenizer_url < <(python3 -c '
import json
profiles = json.load(open("config/model-profiles.json"))
active = next(p for p in profiles["runtime_profiles"] if p["id"] == profiles["active_runtime_profile"])
i = active["installation"]
print(active["id"], i["installed_tag"], i["repository_modelfile"], active["model"]["base_model"],
      i["installed_profile_digest_sha256"], i["tokenizer_url"])
')

if command -v ollama >/dev/null; then
  echo "==> perfil Ollama $profile_id ($tag)"
  if ! ollama show "$tag" >/dev/null 2>&1; then
    ollama pull "$base"
    ollama create "${tag%:latest}" -f "$modelfile"
  fi
  # O digest do manifesto, a mesma coisa que OllamaRuntime.verify_profile compara
  # com /api/tags. Hash do texto do Modelfile é outro valor e não serve aqui.
  installed_digest=$(curl -s http://127.0.0.1:11434/api/tags | TAG="$tag" python3 -c '
import json, os, sys
for model in json.load(sys.stdin)["models"]:
    if model["name"] == os.environ["TAG"]:
        print(model["digest"])
        break
')
  if [ "$installed_digest" != "$expected_digest" ]; then
    echo "aviso: digest do perfil Ollama instalado não bate com config/model-profiles.json" >&2
    echo "  esperado: $expected_digest" >&2
    echo "  instalado: $installed_digest" >&2
  fi
else
  echo "==> ollama não encontrado, pulando criação do perfil" >&2
fi

state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/meu-harness"
tokenizer_path="$state_dir/tokenizer-$profile_id.json"
if [ ! -f "$tokenizer_path" ]; then
  echo "==> baixando tokenizer.json de $profile_id para $tokenizer_path"
  mkdir -p "$state_dir"
  curl -fsSL "$tokenizer_url" -o "$tokenizer_path"
else
  echo "==> tokenizer já existe em $tokenizer_path"
fi
echo "==> no setup, aponte o tokenizer para $tokenizer_path"

echo "==> pronto. Suba o servidor com: uv run harness"
