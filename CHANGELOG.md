# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## [Não lançado]

### Alterado

- **O projeto se chama Meu Harness.** Nome de exibição, pacotes (`meu-harness`,
  `meu-harness-web`), diretórios de config e estado e o User-Agent do coletor; o
  comando `harness` e o pacote Python `harness` ficam. O repositório no GitHub virou
  `Duck1201/meu-harness`.
- **O prompt manda buscar o que o modelo não sabe.** Fato recente, preço, pessoa,
  produto, lei ou nome desconhecido vai para o `web_search` no mesmo passo, em vez
  de resposta de memória; a frase só entra com o `web_search` no catálogo. Resultado
  de tool é final: o modelo nunca pede para esperar. `describe_image` orienta a pedir
  a transcrição dos valores de um print de problema, não a descrição do layout.
- **Todo modelo sai de um Modelfile versionado.** Embedding (`harness-bge-m3`),
  visão (`harness-vision-qwen35-2b`), juiz e OCR ganharam Modelfile e tag própria;
  `scripts/install-models.sh` reinstala todos.

- **RAG: lista de exercícios nunca é passagem, e atividade vira uma busca por
  pergunta.** No Kurose, as atividades do Operator eram as Questões de revisão do
  livro, e a página que as lista ganhava a busca por conter a pergunta palavra por
  palavra (o juiz dava 1,0 a ela) em 3 de 5 perguntas; agora em 0 de 5. Uma
  mensagem com várias perguntas faz uma busca para cada uma, cada passagem diz a
  qual responde, e o modelo recebe as que ficaram sem passagem para buscar com
  `corpus_search`. O prompt proíbe dizer que buscou sem ter chamado a tool.

- **Tools refeitas a partir das conversas reais do Operator.** No `model_smoke` do
  Gemma 4 E4B QAT (34 fixtures × 3 sementes), o código novo fez 101/102 contra
  94/102 do anterior. Os ganhos vieram das fixtures tiradas de Turns reais:
  trocar um nome no `index.html` (0/3 → 3/3) e achar um arquivo por um caminho
  aproximado (0/3 → 3/3).
  - `edit` troca um trecho exato (`old_string`/`new_string`, `replace_all`) em vez
    de um intervalo de linhas com SHA-256. Falha dizendo onde o texto divergiu ou
    em que linhas ele se repete, e devolve o trecho editado.
  - `write_file` substitui sem SHA; `read_file` informa `total_lines`; "não
    encontrado" sugere os caminhos reais parecidos; ler uma imagem aponta para o
    `describe_image`; `grep_search` ganhou `ignore_case` e `include`;
    `get_weather` traduz o código WMO em `conditions`. Descrições reescritas.
  - A ModelView parou de ensinar formato errado: passo só de tool calls não vira
    mais JSON no conteúdo; tentativa rejeitada vira aviso do harness, e não fala
    do modelo com o corpo cru do Ollama; o resultado vai sem id, produtor e listas
    vazias (eram 52% dos caracteres); e só marcadores de template são escapados,
    não todo `<`, que fazia o modelo gravar `<` em HTML.
  - O system prompt diz quando agir ("nunca responda que vai fazer") e não trata
    mais "a busca achou o arquivo" como "não mexa nele".

- **Perfil funcional: Gemma 4 E4B QAT.** Venceu o `mitos` (Qwen3.5-4B abliterated)
  pelo protocolo de promoção: 41/50 contra 33/50, zero violações, 7,4 s contra
  12,1 s por caso e nenhuma resposta malformada. É o primeiro perfil a passar a
  edição byte a byte e responde sem fingir que vê (9/9). O `mitos` segue como
  Challenger. O tokenizer do estimador passa a ser o do Gemma.
- **`SYSTEM-PROMPT.md` é só o texto do Operator.** Saíram o espelho selado do
  prompt derivado, a marca `<!-- OPERATOR -->`, o `seal-system-prompt.py` e o
  teste que falhava quando o espelho envelhecia. Comentários HTML no arquivo são
  ignorados. `scripts/show-system-prompt.py` imprime o prompt exato de hoje.
- **O egress guard aceita destinos locais e privados.** `web_fetch`, a coleta de
  Corpus e o navegador alcançam `localhost` e a rede local. O guard valida só a
  forma da URL (`http`/`https`, sem usuário e senha)
  ([ADR-0017](docs/adr/0017-egress-allows-private-destinations.md)).

### Adicionado

- **Medidor de tokens por segundo no chat.** O runtime mede cada geração, o evento
  `generation_stats` leva tokens e tempos, e a mensagem ao vivo mostra tok/s do
  passo, leitura do prompt e média do Turn; ao terminar, fica a última medição.
- **Três Challengers medidos contra o E4B QAT:** E4B Q8_0 e 12B QAT no Ollama, 26B-A4B
  QAT no llama.cpp com os especialistas de 24 das 30 camadas na RAM. O 12B e o 26B-A4B
  acertaram 24/24 casos reais contra 20/24 do E4B, a 67 e 70 s por resposta contra 19 s;
  nenhum perfil mudou de status. Detalhe em `evals/README.md`.
- **Coletor de traces para fine-tuning.** `scripts/collect-model-traces.py` grava em
  `datasets/model-traces/` as trocas exatas com o modelo em cada passo (ModelView,
  schemas, saída, sem reasoning), montado por `build_live_model_runner`.

- **OCR de PDF escaneado no RAG** ([ADR-0018](docs/adr/0018-ocr-for-scanned-pdfs.md)).
  O PDF sem camada de texto, antes recusado, vira um job em segundo plano:
  `pdftoppm` renderiza cada página e o GLM-OCR lê, ~10 s por página. O texto entra
  com `OcrTranscribedTaint` e a passagem avisa o modelo. Só para PDF sem camada:
  refazer por OCR a camada ruim do Kurose trocava erros e perdia acentos.
- **Limpar todas as conversas**, na barra lateral, com confirmação. Apaga as
  arquivadas também e cancela o Turn que estiver rodando antes; acervos de RAG e
  arquivos do Workspace ficam (`DELETE /api/conversations`).
- **Anexar imagem no chat.** Botão ou Ctrl+V no campo de mensagem grava a imagem
  em `anexos/` no Workspace e manda o caminho para o modelo chamar o
  `describe_image`. Só PNG, JPEG e WebP, conferidos pelos bytes.
- **Seis fixtures de modelo** tiradas de falhas reais e das tools que nenhuma
  fixture exercitava (`calculate`, `get_weather`, `list_directory`), e a bancada
  responde a Open-Meteo, para o `get_weather` ser medido sem internet.
- **Bancada de modelos.** Braços de experimento que nomeiam um RuntimeProfile
  (Challenger) rodam nele; o `ProfileRuntimeSwitch` troca o modelo na placa entre
  braços. Treze candidatos medidos e registrados em `evals/experiments.json`.
- **llama.cpp como segundo runtime** para Challengers que o Ollama não serve,
  verificado pelo SHA-256 do GGUF carregado ([ADR-0013](docs/adr/0013-second-local-runtime-llama-cpp.md)).
- **Scrapling na coleta de Corpus**: a rota de browser por sintoma, com o
  EgressGuard em toda requisição, está ligada. A página montada em JavaScript
  que o HTTP traz vazia é renderizada num Chromium efêmero. O extrator Scrapling
  existe e fica desligado: na Wikipédia ele gasta de 20% a 44% mais tokens com
  resíduo de Markdown ([ADR-0014](docs/adr/0014-corpus-browser-route-scrapling.md)).
- **Visão local, ligada**: `describe_image` lê uma imagem do Workspace e pergunta
  ao Qwen3.5-2B local, fixado por digest; só texto volta, com aviso de incerteza.
  9/9 na bancada de `evals/vision` a temperatura 0, 4/4 em prints reais do
  Operator e 50/50 na promoção ([ADR-0016](docs/adr/0016-local-vision-tool.md)).
  `vision-bench.py --cases` mede casos guardados fora do repositório.
- **Juiz consultivo do Corpus, ligado**: Qwen3-Reranker-0.6B via Ollama anota se
  cada passagem traz o fato pedido e avisa o modelo quando nenhuma traz. Promovido
  com 49/50 contra 45/50 e `unsupported_claims` 0 contra 1; exige
  `OLLAMA_MAX_LOADED_MODELS=3` ([ADR-0015](docs/adr/0015-corpus-answer-judge.md)).
  O juiz é o `harness-judge-ptbr-v2`, LoRA pt-BR treinado em `~/judge-train`: AUC 0,956
  contra 0,828 do original na bancada de 168 pares, e na promoção `unsupported_claims`
  0 contra 3.

### Corrigido

- **O chat mostrava "rodando" depois de o Turn acabar,** quando a página era
  recarregada ou a conversa trocada no meio do Turn: o chat passou a consultar o
  servidor enquanto há Turn ativo sem stream próprio.
- **Uma conversa aparecia dentro de outra.** Runs ao vivo agora pertencem à sua
  conversa, e o refresh de uma conversa deixada nunca troca a tela.
- **A confirmação dispensada pelo yolo ficava entre a tool call e o resultado,** e o
  Gemma respondia "estou buscando, aguarde" com os resultados no contexto; 2 de 3
  rodadas antes, 4 de 4 certas depois.
- **O desafio anti-bot do DuckDuckGo virava busca vazia.** Agora é `failed` com
  `provider_blocked`, e o fallback usa User-Agent de navegador.
- **Apagar conversa com Turn rodando** deixava o worker escrevendo numa conversa
  inexistente; apagar uma ou todas agora cancela o worker antes.
- **Página em branco num PDF** (sem `/Contents`) derrubava a ingestão do arquivo.

- **`edit` repara os dois erros que o modelo mais comete.** O bloco sem a quebra
  final ganha a quebra de volta; antes era recusado, e o modelo dizia ao
  Operator que tinha editado. O bloco que troca várias linhas e chega com `\t` e
  `\n` escritos como texto é decodificado. O preview mostra o bloco reparado. Na
  fixture byte a byte, o Gemma foi de 0/6 para 4/6 em seis seeds, e o arquivo
  saiu certo nas seis.
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
