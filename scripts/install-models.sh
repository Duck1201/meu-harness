#!/usr/bin/env bash
# Reinstala no Ollama todos os modelos que o Meu Harness usa.
#
#   tag derivada               Modelfile                               base oficial
#   harness-gemma4-e4b-qat     modelfiles/gemma4_e4b_qat_ollama        gemma4:e4b-it-qat   (perfil core)
#   harness-bge-m3             modelfiles/bge_m3_embedding             bge-m3:latest       (embedding do Corpus)
#   harness-vision-qwen35-2b   modelfiles/qwen35_2b_vision             qwen3.5:2b-q4_K_M   (visão, ADR 0016)
#   harness-judge-ptbr-v2      modelfiles/harness_judge_ptbr_v2        GGUF do ~/judge-train (ADR 0015)
#
# Idempotente: pull de modelo já presente só confere o manifest.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

for tag in gemma4:e4b-it-qat bge-m3:latest qwen3.5:2b-q4_K_M; do
  ollama pull "$tag"
done

ollama create harness-gemma4-e4b-qat -f modelfiles/gemma4_e4b_qat_ollama.Modelfile
ollama create harness-bge-m3 -f modelfiles/bge_m3_embedding.Modelfile
ollama create harness-vision-qwen35-2b -f modelfiles/qwen35_2b_vision.Modelfile

judge_gguf=$(sed -n 's/^FROM //p' modelfiles/harness_judge_ptbr_v2.Modelfile)
if [[ -f "$judge_gguf" ]]; then
  ollama create harness-judge-ptbr-v2 -f modelfiles/harness_judge_ptbr_v2.Modelfile
else
  echo "aviso: GGUF do juiz ausente ($judge_gguf); harness-judge-ptbr-v2 não foi criado" >&2
fi

ollama list
