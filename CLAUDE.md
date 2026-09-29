# CLAUDE.md

Orientações para agentes que trabalham neste repositório. O README cobre
instalação e operação; aqui fica o que não é óbvio ao editar o código.

## Comandos

```bash
uv sync                                        # dependências + .venv
uv run pytest                                  # suíte Python (tests/)
uv run pytest tests/test_agent_engine.py -k nome   # teste único
uv run ruff check . && uv run ruff format --check .
uv run pyright                                 # strict, cobre src/ e tests/
node scripts/validate-contracts.mjs            # coerência docs <-> contratos JSON
node scripts/validate-contracts.mjs --write    # resela digests derivados
uv run python scripts/show-system-prompt.py    # imprime o system prompt de hoje
cd web && pnpm test && pnpm exec tsc -b && pnpm build
uv run python scripts/run-experiment.py --tier model_smoke --profile <id>  # bancada num Challenger
harness                                        # sobe o servidor (uvicorn)
harness --setup --port 8899                    # reabre o setup em outra porta
scripts/searxng.sh                             # sobe/configura o SearXNG opcional
scripts/llama-server.sh [profile]              # sobe o llama-server de um perfil llama_cpp
```

CI (`.github/workflows/ci.yml`) roda três jobs: backend (ruff + pyright +
pytest), frontend (tsc + vitest + build) e contracts (validador, sem
dependências).

## Contratos são código

`config/*.json` e `evals/*.json` não são configuração ilustrativa: o código lê
esses arquivos e o validador exige coerência entre eles e os documentos.

- `config/tool-registry.json` é a **única** fonte de nomes de tool, schemas,
  efeitos, grants e envelope de resultado. Não existe tool declarada em Python.
- `config/harness.json` define ExecutionRoutes, limites de loop, policy, stores,
  rede e UX.
- `config/model-profiles.json` define RuntimeProfiles e a evidência de cada
  capacidade.
- `evals/fixtures/regressions.json` e `evals/experiments.json` carregam digests
  derivados do próprio conteúdo (`dataset_digest_sha256`, `manifest_digest_sha256`,
  `*_sha256` dos contratos).

**Depois de editar qualquer contrato ou fixture, rode
`node scripts/validate-contracts.mjs --write` e commite o digest junto.** Sem
isso o job `contracts` do CI falha.

O validador também checa links e âncoras de todo Markdown em `docs/` e `evals/`
(mais `CONTEXT.md`), então renomear um heading quebra o CI.

Ele **não** cobre `README.md` nem este `CLAUDE.md`: a lista de Markdown do
validador é só `CONTEXT.md` + `docs/` + `evals/`. Drift nesses dois arquivos não
derruba o CI e por isso passa despercebido — ao mexer no código, confira os dois
à mão.

## Invariantes que não se negociam

Vêm de `docs/DECISOES-2.0.md` — se uma mudança colide com um destes itens, ela
exige alterar o documento, os contratos e as fixtures no mesmo commit.

- **Autorização é por efeito, nunca por nome de tool**: `workspace_read` exige
  WorkspaceRootGrant; `workspace_write` exige também WriteGrant; `data_egress`
  exige WebAccessGrant; `corpus_read` exige CorpusGrant, que traz no escopo o
  `corpus_id` que a Conversation pode ler; `pure_compute` e `local_inference`
  não exigem nada (o segundo marca inferência de outro modelo local, ADR 0016).
  Adicionar um `if tool_name == ...` em caminho de policy está errado por
  construção — o mesmo vale para o roteamento tool -> executor, que também lê o
  efeito. O gate vive em quatro executores independentes (`local_tools.py`,
  `web_tools.py`, `corpus_tools.py` e `vision_tools.py`): grant novo exige tocar
  os quatro. A política de caminho do Workspace é uma só, em `workspace_paths.py`.
- **O modelo é não confiável**: seleção, argumentos e resultados passam por
  validação, autorização, confirmação e sandbox do harness. `blocked` só pode ser
  emitido pelo harness; recusa de provedor é `failed`; sucesso sem itens é `empty`.
- **CanonicalHistory é a fonte autoritativa**; ModelView e AG-UI são projeções
  reconstruídas. Reasoning transita ao vivo para a UI mas **nunca** é persistido
  em estado canônico, telemetria ou replay.
- **O conteúdo das mensagens do modelo é JSON minificado**, montado só em
  `context_builder._json_document`; schemas de tool seguem em JSON Schema no
  tool-calling nativo do Ollama. O XML entrou por ADR 0010 sem medição e saiu
  com ela na promoção de `model_view_serialization`; o render XML continua no
  módulo porque agora é o braço candidato do experimento.
- **`host.json` é a única fonte de configuração do host**: não há variável de
  ambiente equivalente, e a aba Configurações grava o mesmo arquivo do setup.
- **Dois stores separados**: estado conversacional e telemetria. A telemetria
  recebe só IDs, digests, classes, tamanhos, contagens e tempos — nunca conteúdo.
  Falha de telemetria é não fatal.
- **Todo ToolResult** tem `status`, `retryable`, `data`, `error`, `meta`.
- **Sem senha de Operator, só loopback direto é atendido**; com senha, toda rota
  exige sessão. Não há terceira opção nem flag que abra a porta sem autenticação.
- **Modo yolo é decisão do Operator, não default**: global, desligado de fábrica,
  com opt-out por Conversation; ligado, aprova toda confirmação (inclusive sob
  taint) e registra cada chamada como `waived` (ADR 0008).
- **Medição inventada é proibida**: experimento sem execução fica com
  `result: null`.
- **O runner ao vivo se compõe num lugar só**: `build_live_runner`
  (`src/harness/evals/model_runner.py`), chamado pelo script de experimento e
  pelo teste que prova que todo tipo de fixture tem runner. Duplicar essa
  composição já deixou o script quebrado por meses sem ninguém notar. O
  `ModelCaseRunner` com embedding, visão e juiz reais também tem um lugar só,
  `build_live_model_runner`, usado pelo script de experimento e pelo coletor de
  traces (`scripts/collect-model-traces.py`, que alimenta `datasets/model-traces/`).

## Arquitetura

Composition root em `api._default_service`, chamado por `create_app` quando nada
é injetado — `__init__.py` só reexporta o pacote. Entrypoint em `__main__.py`
(uvicorn sobre `api.create_app`).

| Camada | Módulos |
|---|---|
| HTTP/SSE | `api.py`, `ag_ui.py`, `auth.py`, `setup.py` |
| Orquestração | `application_service.py`, `agent_engine.py`, `workspace_coordinator.py` |
| Portas | `ports.py` (Protocols), `domain.py` (tipos) |
| Tools | `local_tools.py`, `web_tools.py`, `composite_tools.py`, `brave_browser.py`, `page_verification.py`, `vision_tools.py` + `vision_runtime.py` (ADR 0016), `workspace_paths.py` |
| Estado | `conversation_store.py` (SQLite canônico), `observability_store.py` |
| Contexto/modelo | `context_builder.py`, `system_prompt.py`, `token_estimator.py`, `ollama_runtime.py`, `llamacpp_runtime.py` (ADR 0013) |
| Corpus | `corpus_tools.py` (executor do efeito), `corpus_service.py`, `corpus_store.py` (um SQLite por acervo), `corpus_ingestion.py`, `corpus_scraper.py`, `corpus_browser.py` (rota de browser, ADR 0014), `corpus_judge.py` (juiz consultivo Qwen3-Reranker via Ollama, ADR 0015), `corpus_ocr.py` (OCR de PDF escaneado, GLM-OCR via Ollama, ADR 0018) |
| Config | `config.py` (contratos JSON), `host_config.py` (HostConfig do host) |
| Evals | `evals/` (runner, service, model_runner, runtime_switch, oracles, statistics, store, bench, loader, lease, models, language) |

Fluxo de um Turn: `PendingRequest` -> `AgentEngine` itera AgentSteps ->
`ModelView` reconstruída por `context_builder` a cada passo -> o runtime do
perfil (`ollama_runtime` ou `llamacpp_runtime`, por `runtime.backend`) gera -> tool calls passam por preflight/policy -> `ToolResult` volta ao
CanonicalHistory -> exatamente um `TerminalOutcome`.

O `system_prompt` é **derivado dos contratos e de fatos do host** (ADR 0007), não
escrito à mão. A data corrente é o fato do host: `build_system_prompt` a recebe
como argumento obrigatório — produção passa o relógio em UTC, a bancada passa
`BENCH_DATE` — para que o prompt do bench não mude sozinho a cada dia. O texto do
Operator entra pelo mesmo caminho: é o `SYSTEM-PROMPT.md` inteiro, menos os
comentários HTML, lido no composition root e passado por argumento.

## Convenções

- Python 3.13, `ruff` com `line-length = 100`, regras `E,F,I,UP,B,SIM,RUF`.
- `pyright` em `typeCheckingMode = "strict"` — sem `Any` solto, sem ignore novo.
- Dataclasses `frozen=True, slots=True` para tipos de domínio; `Protocol` em
  `ports.py` para tudo que é substituível.
- Testes espelham o módulo (`tests/test_<modulo>.py`), sem framework extra além
  do pytest. Exceção: `tests/evals/` agrupa por assunto, não por módulo.
- Frontend em `web/`: React 19 + Vite + vitest, cliente em `web/src/client/`. O
  app só compõe `FetchHarnessClient`; `MockHarnessClient` é dublê dos testes e
  não entra no bundle.

## Idioma

Documentação, comentários e mensagens de erro voltadas ao Operator em pt-BR.
Identificadores, nomes de tool, campos de contrato e chaves JSON em inglês.
`config/tool-registry.json#model_facing_language` fixa o idioma exposto ao modelo.

## Documentação relevante

`CONTEXT.md` (glossário canônico), `docs/DECISOES-2.0.md` (contrato normativo),
`docs/RELEASE-PENDING.md` (o que ficou de fora e sob quais gates),
`docs/adr/` (fronteiras arquiteturais), `docs/THREAT-MODEL-AUTH.md`,
`evals/README.md` (protocolo de avaliação).
