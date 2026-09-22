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
