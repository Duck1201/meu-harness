# Avaliações do Harness 2.0

Este diretório transforma a herança empírica e o contrato vigente em avaliações
web repetíveis. As fixtures não copiam conteúdo de produção: reproduzem o
formato das falhas observadas e os invariantes atuais.

## Arquivos

- [`fixtures/regressions.json`](fixtures/regressions.json): casos determinísticos
  de seleção, segurança, parsing, loop, contexto, backend e automações;
- [`experiments.json`](experiments.json): braços, fatores fixos, métricas e gates
  para hipóteses ainda abertas.

## Escopo do corpus

O corpus é para regressões de **modelo** e de **executor**: coisas que só um
RuntimeProfile real, um executor real ou um browser real podem decidir. Duas
consequências:

1. **Toda fixture precisa de um runner.** Um `type` sem runner é descartado em
   silêncio por `EvalService._fixtures`, e uma fixture que nunca roda não
   protege nada. Um teste garante que todo `type` do dataset é suportado.
2. **Contrato de arquitetura não é fixture.** Fronteira RuntimeProfile/
   ExecutionRoute, transitoriedade do reasoning, projeção AG-UI e middleware HTTP
   de autenticação são invariantes de código, cobertos por teste de unidade. Foi
   a mesma decisão tomada para autenticação (`tests/test_auth.py`) e aplicada
   depois a `contract_boundary`, `state_projection` e `parser_contract`.

As fixtures `corpus_contract` montam um Corpus a partir do próprio estímulo e
rodam `corpus_search` contra ele com um embedder determinístico de bancada. Elas
provam o gate — grant ausente é `blocked`, passagem coletada carrega taint, nada
acima do piso é `empty` — e não medem resposta nenhuma. O piso do contrato é
calibrado no modelo real, então a bancada declara o seu; o que a fixture verifica
é que existe piso e que ele zera o resultado. Nelas o grant vem do estímulo,
porque o oráculo de cada uma é escrito para o valor que ela declara.

As fixtures `corpus_answer` medem o que o Operator lê. O mesmo oráculo roda nos
dois braços de `corpus_retrieval_vs_baseline` — tem que ser o mesmo, ou não há
comparação — e quem decide se existe acervo é o braço, não o estímulo. Que os
dois braços rodaram experimentos diferentes é a métrica `injected_passages` que
mostra, e o operador de mesmo nome existe para a fixture que prova a injeção
antes do primeiro AgentStep.

Elas indexam com o `bge-m3` e o piso do contrato, não com o embedder da bancada.
Foi medido o motivo: com o embedder determinístico, a pergunta sobre a porta do
servidor de e-mail traz a passagem do proxy, e o modelo a estica até responder
`8899`; com o `bge-m3` e o piso 0.53, essa mesma pergunta volta sem passagem
nenhuma, enquanto a pergunta que a passagem responde volta com 0.76. Medir o
modelo diante de uma passagem que a produção jamais entregaria é medir outro
sistema. O embedder determinístico continua sendo o das fixtures `corpus_contract`,
que só provam o gate e precisam rodar sem GPU.

A verificação de página segue a mesma regra: enquanto a automação não estiver no
caminho de escrita, nenhuma fixture consegue observá-la, e a cobertura vive em
`tests/test_page_verification.py`. O experimento homônimo permanece como portão
registrado e, sem fixture alcançável, bloqueia com `no_deterministic_cases` — que
é o relato honesto de "ainda não mede nada", em vez de um verde emprestado de
fixtures genéricas de `write_file`.

## Unidade de avaliação

O resultado principal é a tarefa completa, não apenas a primeira tool call. Cada
runner deve registrar:

1. RuntimeProfile e digest completos;
2. ExecutionRoute, fixture, dataset_version, digests de contratos e o
   digest do bloco do Operator em `SYSTEM-PROMPT.md`;
3. sequência de AgentSteps, decisões e model_tools;
4. validação de argumentos e efeitos realmente aplicados;
5. TerminalOutcome e TaskVerdict, sem conflar os dois;
6. tokens, latência, repetição e violações de segurança.

Conteúdo integral fica no Workspace/Conversation isolado da execução. O store de
métricas recebe somente IDs, hashes, tamanhos, contagens, classes e tempos.

## Protocolo

1. Crie os dois stores, uma Conversation e um Workspace exclusivos por braço.
2. Use path temporário curto e de mesmo comprimento entre os braços.
3. Fixe RuntimeProfile, ExecutionRoute, dataset_version, os digests de dataset,
   harness e registry, e o digest do bloco do Operator — dois braços com prompts
   de Operator diferentes não são o mesmo sistema.
4. Randomize a ordem, registre seed/ordem e não reutilize cache entre braços.
5. Execute primeiro o piloto de 15 casos por braço.
6. Trate o piloto apenas como direção.
7. Para promoção, execute 50 casos por braço em três ordens/seeds e reporte
   intervalo, salvo falha de segurança, que reprova imediatamente.
8. Par com verdict inconclusivo em qualquer um dos braços sai da comparação, e
   quantos saíram entra no resultado. Acima de `max_inconclusive_pair_rate`
   (0.2) a execução não decide nada e o gate responde
   `inconclusive_above_ceiling`. A regra anterior reprovava a execução inteira
   ao primeiro inconclusivo, o que com um modelo pequeno tornava o gate
   incumprível e escondia a hipótese atrás do modelo travando.
9. Execute a tarefa fim a fim na superfície web e guarde os casos que falharam;
   não substitua o corpus por exemplos fáceis.

## Bake-off de modelos

Um braço que declara `runtime_profile` roda naquele Challenger, e só o modelo
muda: rota, loop, contexto, registry e embedding do acervo são os do braço
controle. Os braços rodam em sequência, e `scripts/run-experiment.py` troca o
modelo entre eles pelo `ProfileRuntimeSwitch`
(`src/harness/evals/runtime_switch.py`), que descarrega o anterior antes de
subir o próximo — dois modelos de chat não cabem juntos em 8 GB, e o segundo
iria parcial para a CPU e seria medido mais lento do que é.

Cada perfil mede o orçamento com o próprio tokenizer, lido de
`.harness/tokenizers/<runtime_profile_id>.json`; o do perfil da rota continua
vindo de `--tokenizer`. O script imprime o digest de cada tokenizer usado, e ele
entra no congelamento do resultado junto de `runtime_profile_id`.

```bash
uv run python scripts/run-experiment.py --tier model_smoke --profile <challenger_id>
uv run python scripts/run-experiment.py runtime_profile_bakeoff --phase pilot
```

## Digests

Fixtures e experimentos usam SHA-256 de JSON canônico UTF-8, sem espaços e com
chaves ordenadas recursivamente. O campo `digest_contract.scope` enumera, em
ordem, os campos cobertos; o próprio campo de digest fica fora do cálculo. O
validador recalcula dataset, manifesto de experimentos e snapshots dos três
contratos JSON, portanto um digest não pode ser atualizado sem os bytes que ele
identifica.

## Estados de um experimento

- `reproduce_inherited_result`: repete primeiro uma observação do 1.0;
- `required_before_default_change`: impede alterar o baseline;
- `required_before_default_enablement`: impede ligar uma hipótese por padrão;
- `blocked_*`: faltam manifesto, parser ou outra pré-condição de experimento;
- `optional_after_baseline`: otimização que não bloqueia o primeiro release;
- `deferred_*`: só deve consumir engenharia quando seu gatilho ocorrer.

## Validação estrutural

Execute:

```bash
node scripts/validate-contracts.mjs
```

O comando valida JSON, RuntimeProfile/ExecutionRoute, as três coleções do
registry, efeitos e grants, ResultPayload, fixtures/oráculos, digests e links
Markdown locais. Resultados de experimento podem permanecer ausentes; digests e
gates contratuais não.

## Juiz de Corpus

Medição que escolheu o juiz de [ADR-0015](../docs/adr/0015-corpus-answer-judge.md): 60 pares (pergunta, passagem) rotulados à mão em pt-BR, metade quase-acertos — mesmo assunto, dado ausente —, em 2026-09-22. O conjunto é pequeno: diferenças de dois ou três pares entre os melhores são ruído, e ele serve para descartar, não para ordenar os de cima.

| Juiz | Onde | Acerto em 0,5 | AUC | Custo por Turn de 6 passagens |
|---|---|---|---|---|
| Jev (TypeSafe) | nuvem | 100% | 1,000 | ~4 s de rede; passagens saem da máquina |
| Kev-4B | CPU | 96,7% | 0,999 | ~21 s |
| **Qwen3-Reranker-0.6B, instrução em pt-BR** | **GPU via Ollama, Q8** | **85%** | **0,948** | **1,2 s, 746 MB de VRAM** |
| Qwen3-Reranker-0.6B, instrução em pt-BR | CPU, transformers | 86,7% | 0,944 | 13 s |
| bge-reranker-v2-m3 | CPU | 86,7% | 0,920 | ~4 s |
| Kev-0.8B | CPU | 63% | 0,817 | ~3 s |
| Laya multilingual | CPU | 70% | 0,703 | 0,6 s |

## Bancada de visão

As imagens de [`evals/vision/`](vision/) medem o modelo de visão de
`describe_image` ([ADR-0016](../docs/adr/0016-local-vision-tool.md)) no digest que
`config/harness.json#vision` fixa, pelo mesmo runtime da produção:

```bash
uv run python scripts/vision-bench.py
```

São nove casos com gabarito exato: cupom com R$ 24,70, tabela com `24.7` ao lado
de `247`, terminal com código e linha, código, diálogo, gráfico, texto miúdo, uma
senha ilegível e um preço que não existe. Os dois últimos passam quando o modelo
diz que não consegue ler. `make_images.py` regenera as imagens e o `cases.json`.

Medição de 2026-09-23 (9 imagens × 3 seeds a temperatura 0,2, depois 9 a 0 pelo
runtime):

| Modelo | Acertos | Não inventa | Mediana |
|---|---|---|---|
| Qwen3.5-2B, temperatura 0 (runtime de produção) | 9/9 | 2/2 | 0,8 s |
| Qwen3.5-2B, temperatura 0,2 | 25/27 | 6/6 nessas seeds, mas 7/10 em dez | 0,7 s |
| GLM-OCR (só transcreve) | 21/21 | — | 4,4 s |
| Gemma 4 E4B QAT (chat ativo) | 21/27 — embaralha dígitos pequenos | 6/6 | 1,0 s |
| mitos | 21/27 | 0/6 — inventa senha e preço | 0,9 s |
| MiniCPM-V 4.6 1B | 18/27 | 0/6 | 0,5 s |

A temperatura decidiu: a 0,2 o Qwen3.5-2B inventou a senha ilegível em 3 de 10
seeds e leu a linha `118` como `18` em 5; a 0 foi honesto e certo 10 de 10.

