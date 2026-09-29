# Traces do modelo no harness

Registro do que o modelo **fez**, não só do veredicto. É a matéria-prima do
fine-tuning do Gemma nas tools do harness: cada linha traz as trocas exatas com o
modelo em cada passo, cada chamada emitida, cada recusa que o harness devolveu, a
resposta final e as asserções que o oráculo reprovou.

Este diretório **não é contrato**. Não tem digest, não entra em
`scripts/validate-contracts.mjs` e não é lido por código de produção. O contrato
versionado continua sendo `evals/fixtures/regressions.json` — aqui é saída de
execução, não fonte de verdade.

## Por que existe um coletor separado

Uma execução de eval não guarda isso. `EvalStore` grava veredicto, TerminalOutcome
e contagens, por contrato de privacidade, e cada caso roda num workspace
temporário que é apagado no fim. A evidência existe apenas dentro do
`CaseRunResult`, em memória. O coletor re-executa as fixtures de modelo e
serializa essa evidência.

O runner é o mesmo da bancada, montado por `build_live_model_runner`
(`src/harness/evals/model_runner.py`) com embedding, visão e juiz reais. O
coletor montava o seu à mão e ficou quebrado sem ninguém notar quando as
fixtures de visão entraram; `tests/evals/test_traces.py` importa o script para
que isso não se repita. Entre o runner e o runtime fica o `RecordingModelRuntime`
(`src/harness/evals/traces.py`), que copia cada request e cada saída como
passaram — **sem reasoning**, que não é persistido em lugar nenhum.

## Regenerar

```bash
.venv/bin/python scripts/collect-model-traces.py
.venv/bin/python scripts/collect-model-traces.py --seed 104729 --seed 7 --output outro.jsonl
```

Sem `--seed`, usa as três sementes registradas do protocolo. Repetir a mesma
semente produz cópias, não evidência nova: a semente é o que varia uma execução
local.

## Esquema, uma linha por caso

| Campo | Conteúdo |
|---|---|
| `runtime_profile`, `model`, `profile_digest_sha256` | de qual modelo instalado é o trace |
| `operator_prompt_digest` | digest do `SYSTEM-PROMPT.md` usado |
| `fixture_id`, `fixture_type`, `tags`, `seed` | identidade do caso |
| `verdict` | `pass`, `fail` ou `inconclusive` |
| `prompt.system` | o system prompt do primeiro passo, como o modelo recebeu |
| `prompt.user` | o pedido da fixture |
| `prompt.offered_tools` | os nomes das tools do primeiro passo |
| `prompt.workspace` | os arquivos semeados antes do turno |
| `observed.tool_calls` | toda chamada emitida, com argumentos |
| `observed.tool_results` | status, erro, produtor, taints e um trecho do payload |
| `observed.final_response` | a resposta entregue |
| `observed.terminal_outcome` | por que o turno parou |
| `harness_refusals` | só as chamadas com `status: blocked`, com código e mensagem |
| `expected.typed_assertions` | o comportamento correto, na forma que o oráculo executa |
| `failed_assertions` | o que reprovou, com o detalhe do operador |
| `model_exchanges` | um item por passo: `messages` (a ModelView inteira), `tools` (schemas), `options`, `seed`, `think`, `output` (conteúdo e tool calls) e `usage` |
| `metrics` | inclui `rejected_model_attempts`, `extra_tool_calls`, `steps_to_terminal` |

## Casos que passam também entram

Numa mesma fixture, um trace que passa e um que falha diferem só na escolha do
modelo — é exatamente o par que um dataset de preferência precisa. Guardar só as
falhas jogaria fora a metade positiva.

## Limites honestos

- Amostra pequena: 14 fixtures de modelo (9 `model_task`, 4 `corpus_answer`, 1
  `capability_gate`) × 3
  sementes. Cresce quando o corpus de fixtures cresce; não é volume de
  fine-tuning ainda.
- Um trace vale para o modelo e o prompt que o geraram: trocou o perfil, o
  `SYSTEM-PROMPT.md` ou o registry de tools, regenere. Os traces do `mitos`
  (agosto de 2026) foram descartados por isso.
- Em `observed.tool_results`, `payload` de resultado grande é truncado em 2000 caracteres, com o tamanho
  original registrado. `model_exchanges` não trunca: é o que o modelo leu.
- `expected.typed_assertions` descreve o comportamento correto de forma
  verificável, mas **não** é uma resposta-alvo escrita. Transformar isso em alvo
  de treino é uma decisão separada, e inventar a resposta ideal aqui seria
  fabricar dado.
