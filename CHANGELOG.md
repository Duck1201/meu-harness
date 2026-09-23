# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## [Não lançado]

### Alterado

- **Perfil funcional: Gemma 4 E4B QAT.** Venceu o `mitos` (Qwen3.5-4B abliterated)
  pelo protocolo de promoção: 41/50 contra 33/50, zero violações, 7,4 s contra
  12,1 s por caso e nenhuma resposta malformada. É o primeiro perfil a passar a
  edição byte a byte e responde sem fingir que vê (9/9). O `mitos` segue como
  Challenger. O tokenizer do estimador passa a ser o do Gemma.

### Adicionado

- **Bancada de modelos.** Braços de experimento que nomeiam um RuntimeProfile
  (Challenger) rodam nele; o `ProfileRuntimeSwitch` troca o modelo na placa entre
  braços. Treze candidatos medidos e registrados em `evals/experiments.json`.
- **llama.cpp como segundo runtime** para Challengers que o Ollama não serve,
  verificado pelo SHA-256 do GGUF carregado ([ADR-0013](docs/adr/0013-second-local-runtime-llama-cpp.md)).
- **Scrapling na coleta de Corpus**: limpeza de conteúdo oculto e rota de browser
  por sintoma com o EgressGuard em toda requisição, ambos desligados até medição
  ([ADR-0014](docs/adr/0014-corpus-browser-route-scrapling.md)).
- **Visão local, desligada até medição**: `describe_image` lê uma imagem do
  Workspace e pergunta ao Qwen3.5-2B local, fixado por digest; só texto volta,
  com aviso de incerteza. 9/9 na bancada de `evals/vision` a temperatura 0
  ([ADR-0016](docs/adr/0016-local-vision-tool.md)).
- **Juiz consultivo do Corpus**: Qwen3-Reranker-0.6B via Ollama anota se cada
  passagem traz o fato pedido, desligado até o experimento
  ([ADR-0015](docs/adr/0015-corpus-answer-judge.md)).

### Corrigido

- O oráculo de eval lê a resposta como o Operator a vê: escape de Markdown
  (`ERR\_ORIGIN\_2049`) reprovava resposta certa e aprovava invenção.
- O oráculo de eval reprovava respostas honestas do Corpus. "Não há menção",
  "não menciona" e "não contém informações" contam como admitir ignorância, e
  `response_contains` ganhou `unless_admits_ignorance`: citar o fato vizinho
  para descartá-lo deixou de contar como tomá-lo emprestado. O Gemma respondia
  certo à fixture de quase-acerto em todas as seeds e saía reprovado.
- Automações internas chegam no papel que o template do perfil lê
  (`internal_automation_role`); o do Gemma descartava a recuperação do Corpus.
- O motor deixou de fixar `think=True` e as marcas de vazamento do Qwen; o runner
  de eval passou a respeitar `thinking` e `max_steps` dos braços.

## [1.0.0] — 2026-08-15

Primeira release. O que ela exclui deliberadamente está em
[`docs/RELEASE-PENDING.md`](docs/RELEASE-PENDING.md).

### Adicionado

- **Recuperação por Corpus (RAG).** O Operator monta acervos por upload ou por
  coleta web, escolhe qual está ativo em cada Conversation e o harness injeta as
  passagens antes do primeiro AgentStep, com `corpus_search` disponível para o
  modelo refinar a busca depois. Cada Corpus é um SQLite próprio em
  `state_dir/corpora/`; apagar o Corpus apaga o arquivo, e a retenção global não
  o alcança. Autorização é efeito novo `corpus_read` com `CorpusGrant` cujo
  escopo nomeia o acervo — selecionar é conceder, "Desligado" é revogar. Chunk
  vindo do scraper carrega `UntrustedWebTaint`, então material da web continua
  custando confirmação de `data_egress`
  ([ADR-0011](docs/adr/0011-corpus-retrieval-and-corpus-grant.md)).
- **Busca híbrida com piso de relevância.** Vizinhança densa (`bge-m3`, 1024
  dimensões, via `sqlite-vec`) e BM25 (FTS5) fundidas por rank recíproco. O piso
  é lido na similaridade cosseno e não no escore de fusão — rank recíproco ordena
  e não mede, e o primeiro colocado pontua igual respondendo ou não à pergunta.
  Nada acima do piso é `empty`, com a instrução de dizer que não sabe. O piso
  vale por passagem, inclusive para quem só a perna lexical trouxe, e está em
  0,53. Varredura de 25 textos contra um livro técnico: comando genérico ("cria
  um arquivo .md", "roda os testes") não passa de 0,521, então 0,53 já zera todos
  eles, enquanto 0,55 devolvia vazia uma pergunta que o livro responde — "explica
  o algoritmo de janela deslizante", topo 0,541. Não existe piso que separe
  comando de pergunta: um comando com anexo pontua 0,535 e um comando sobre o
  assunto do acervo passa de 0,60. O piso mede relevância, não intenção — quem
  impede a passagem irrelevante de sequestrar o Turn é a instrução injetada. O
  número é do `bge-m3` e não se transfere para outro embedder.
- **Tradução só na query.** Documento é indexado como foi extraído; uma geração
  curta produz a versão autônoma em inglês que alimenta só a perna lexical,
  enquanto a densa usa o texto do Operator. Medição local: a pergunta crua em
  pt-BR na perna lexical contra corpus em inglês piora o ranking.
- **Coleta por API quando o site tem uma.** Semente com `/api.php` é coletada
  pela API do MediaWiki — wiki inteira, texto puro, sem seguir link; o resto cai
  num crawl com teto que respeita `robots.txt`. Job em segundo plano com
  progresso, cancelamento e retomada: a página que o Corpus já tem sai pelo
  endereço na listagem, antes de virar requisição, então uma wiki maior que o
  teto é coletada em rodadas sucessivas em vez de recomeçar do alfabeto. O
  extrato de artigo inteiro vem um por requisição porque é o que o MediaWiki
  concede — pedir vinte devolve dezenove páginas vazias. Extrato que volta magro
  cai para `action=parse`: `extracts` não renderiza template, e a wiki que guarda
  o fato dentro de um responde a página de habilidade com o `See also` e mais
  nada. Custa uma requisição a mais nas páginas magras e chega mais sujo, mas o
  número que a pergunta procura mora justamente na tabela que o extrato descarta.
- **PDF extraído com layout.** O modo padrão do pypdf devolve a página como um
  bloco corrido, sem a linha em branco que separa parágrafos, e insere espaço no
  meio de palavra. Medido num livro de 660 páginas: 655 blocos de 550 palavras
  viram 3453 de 104. Hífen suave no fim da linha passa a juntar as metades — era
  apagado antes da regra que trata quebra, e o Corpus indexava "pode mos".
- **Piso de tamanho de Chunk.** Seção mais curta que `chunk_minimum_tokens` não
  é indexada como passagem; o texto continua no Document. Medido numa wiki de
  jogo: abaixo de oito tokens a faixa é lista de links e andaime de citação, que
  vencia a busca lexical por casar com a pergunta ao pé da letra e ocupava uma
  das seis vagas do Turn sem responder nada. É onde começa o fato de uma linha.
- **Modo yolo.** Decisão permanente do Operator, global e desligada de fábrica,
  com opt-out por Conversation: enquanto estiver ligada o gate aprova toda
  confirmação — inclusive escrita sob `UntrustedWebTaint` — e concede WriteGrant
  e WebAccessGrant que faltarem. Cada chamada continua registrada como `waived`,
  nunca como `approved`. O risco aceito está em
  [`docs/adr/0008-operator-yolo-mode.md`](docs/adr/0008-operator-yolo-mode.md).
- **Tools `calculate` e `get_weather`.** A primeira estreia o efeito
  `pure_compute`, que não exige grant algum, e avalia a expressão por AST com
  whitelist — sem `eval`, com teto de expoente. A segunda usa Open-Meteo, que não
  pede credencial, como `data_egress` com `UntrustedWebTaint`.
- **Aba Configurações editável.** `PUT /api/admin/host-config` valida pelo mesmo
  caminho do setup, grava `host.json` atomicamente e devolve `restart_required`.
- **Confirmação de escrita com UntrustedWebTaint.** Um Turn que lê a web e pede
  para escrever agora para numa `ConfirmationGate` e espera a decisão do
  Operator, em vez de terminar em `blocked`. A decisão vale para aquela chamada:
  não cria grant, não persiste e não amplia autoridade. Registrada em
  CanonicalHistory como automação `write_confirmation`.
- **Autenticação do Operator.** A exposição de rede é derivada da credencial:
  sem senha, só loopback direto; com senha, toda rota exige sessão. Senha em
  PBKDF2-HMAC-SHA256 com 600.000 iterações, sessões opacas só em memória.
  Modelo de ameaça em [`docs/THREAT-MODEL-AUTH.md`](docs/THREAT-MODEL-AUTH.md).
- **Navegador Brave real**, dirigido por CDP sobre o `aiohttp` que já era
  dependência. Um processo e um `--user-data-dir` descartável por operação, e o
  destino é fixado com `--host-resolver-rules` a partir do endereço que o
  `EgressGuard` aprovou — o navegador não resolve DNS por conta própria.
- **Tiers de eval que faltavam.** `ModelCaseRunner` roda o AgentEngine real
  contra o RuntimeProfile e `BrowserBenchCaseRunner` replica a escalação contra
  uma bancada loopback determinística. Antes, `EvalService` bloqueava qualquer
  experimento com `runner_not_configured`.
- **Telas de setup e de login** no frontend; a primeira execução e o primeiro
  acesso deixaram de exigir `curl`.
- Ponto de entrada `harness`, README, CI, `scripts/run-experiment.py` e
  `scripts/validate-contracts.mjs --write`.

### Alterado

- **A busca deixou de ser paga.** A Brave Search API saiu inteira, junto com sua
  credencial e a plumbing dela. Entra SearXNG declarado pelo Operator, com
  DuckDuckGo sem chave como fallback; o endpoint declarado é liberado no
  `EgressGuard` como allowlist de um item, só no caminho de `web_search`
  ([ADR-0009](docs/adr/0009-search-provider-searxng-with-fallback.md)). O
  navegador local fica, e passa a aceitar qualquer Chromium.
- **O que vai para o modelo é JSON minificado**, não XML: a troca por XML entrou
  por legibilidade e sem medição, e a promoção `model_view_serialization` mediu
  perda nos dois critérios que o gate nomeia, então
  [ADR-0010](docs/adr/0010-model-facing-messages-are-xml.md) foi revertida e o
  default voltou ao JSON. O render XML continua em `context_builder` como braço
  candidato, e a deduplicação de `data` repetido é decidida sobre o payload,
  antes do render. Schemas de tool seguem em JSON Schema no tool-calling nativo.
- **Grant que falta vira diálogo, não fim de Turn.** `web_access_grant_required`
  passa a ser perguntado no meio do Turn como já acontecia com
  `write_grant_required`: aprovar concede o grant e o mesmo Turn continua pelo
  SSE aberto, sem reenviar o prompt. `workspace_root_grant_required` segue
  terminal, apontando para as Configurações.
- **Roteamento tool → executor por efeito**, não por prefixo de nome: quem
  declara `data_egress` vai para o executor web, o resto para o local.
- **Menu lateral só com navegação**: o logo e o avatar "OP" saíram.

### Corrigido

- `web_fetch` escalava para o navegador apenas em HTTP 401/403. Uma extração
  abaixo do limiar calibrado de 120 caracteres — o sintoma que
  `harness.json#network.web_fetch` declara, e exatamente como uma página
  renderizada por JavaScript se apresenta a um cliente HTTP — devolvia
  `empty_extraction` e desistia.
- O contador de violações de segurança marcava toda recusa do harness como
  violação, o que tornava o gate `zero_violations` impossível de passar por
  construção. Violação agora significa expectativa de segurança não cumprida.
- `config/harness.json` declarava `LangGraph SqliteSaver` e `assistant-ui`;
  nenhum dos dois existe no projeto.
- `CredentialStore` sobrescrevia uma credencial ao gravar a outra.
- O encerramento do navegador matava só o líder do grupo, deixando zygotes e o
  crashpad handler órfãos.
- `[tool.pyright]` não apontava para o `.venv`, e um checkout novo reportava
  1229 erros falsos.

### Removido

- **`.env` e todas as variáveis `HARNESS_*`.** `host.json` é a fonte única, e o
  que precisa existir antes do app virou flag: `--host`, `--port`,
  `--host-config` e `--setup`.

### Evidência

A bateria de 15/08/2026 substitui a de 10/08: aquela foi congelada contra o
dataset 2.0.0 e digests de harness e registry que não existem mais, o que a
torna evidência de outro sistema pelo próprio protocolo. Tudo abaixo foi medido
sob a semântica de verdict de
[ADR-0012](docs/adr/0012-eval-verdict-separates-failure-from-missing-measurement.md)
— limite de loop, resposta malformada, timeout de geração e recusa de policy são
`FAIL`; só infraestrutura e interrupção externa ficam `INCONCLUSIVE` — então
número anterior a 15/08 não se compara por aritmética.

- `guarded_web_brave_escalation`, o único experimento `required_before_release`,
  foi reexecutado contra os contratos vigentes: promoção de 50 casos por braço
  com três seeds, contra o perfil `mitos` e o Brave instalado. Ambos os braços
  passaram todos os casos, com zero violações de segurança, e o gate de promoção
  retornou `promoted`. Cada caso também roda
  `eval_bench_exception_is_not_available_in_production`, que prova que o egress
  guard de produção continua recusando o endereço da bancada mesmo com
  WebAccessGrant.
- `corpus_retrieval_vs_baseline` **promovido**, na primeira execução em que o
  experimento decide: 24/50 no braço com acervo contra 1/50 sem, zero par
  inconclusivo, e a metade qualitativa do gate medida em vez de inferida —
  `unsupported_claims` 30 com acervo contra 55 no baseline. Fundamentar separa
  por completo: 24/24 com acervo, 0/24 sem. O braço com acervo ainda falha 26/26
  a fixture de ignorância, limitação de RuntimeProfile registrada em
  `docs/RELEASE-PENDING.md`.
- `model_smoke` em 207/234 com **zero inconclusivo**, contra 209/234 com dois
  inconclusivos antes: as duas quedas eram o timeout de 120 s do cliente cortando
  geração longa, agora 300 s pelo contrato. Executor, gate, taint e path seguem
  9/9. `unsupported_claims` soma 12 no corpus inteiro, todas na fixture que mede
  invenção com acervo silencioso.
- `model_view_serialization` **reprovou o candidato XML** nos dois critérios
  declarados — PASS 38 contra 42 e `rejected_model_attempts` 36 contra 17 —,
  mantendo o JSON e sustentando por execução própria a reversão de ADR-0010. O
  candidato ganha onde o gate não olha (`tool_noop_rate` 1,167 contra 2,667,
  latência somada 465 s contra 692 s), e é em resposta malformada que a diferença
  mora: 18 casos contra 9.
- Zero violação de segurança em todos os braços de todos os experimentos.
  Resultados e blocos `frozen` congelados em `evals/experiments.json#results`.

### Limitações aceitas

Medidas nesta bateria e aceitas: o modelo inventa quando o acervo se cala
(`corpus_answer_admits_the_acervo_does_not_say` falha 26/26 no braço com acervo,
e a invenção é contada por `unsupported_claims`, 30 contra 55 do baseline); a
edição byte a byte não passa (`edit_preserves_whitespace_round_trip` segue 0/9,
com `tool_write_required_relative_path` em 8/9 sob o JSON); o modelo sai para a
web quando não enxerga (`vision_is_not_silently_available` passa 2 de 9, e os
casos que falham terminam barrados em `web_taint_confirmation_required`);
recuperação sem reranker; chat e embedding disputando a mesma VRAM; e coleta
genérica pior que a rota MediaWiki. Deliberadas por decisão registrada: modo
yolo aceita prompt injection ([ADR-0008](docs/adr/0008-operator-yolo-mode.md)) e
a busca alcança o endpoint SearXNG declarado mesmo em loopback
([ADR-0009](docs/adr/0009-search-provider-searxng-with-fallback.md)). Da
primeira contagem seguem: Ollama local como único runtime, verificação de página
desligada, DNS rebinding entre validação e conexão, telemetria sem conteúdo,
raciocínio não persistido, retenção por política global, um único Operator sem
papéis, sessão sem persistência, exceção de bancada restrita aos tiers de eval,
Python 3.13 em Linux x86_64 e UX web-first. Cada uma com seu gate de saída em
`docs/RELEASE-PENDING.md`.
