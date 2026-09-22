# Decisões normativas do Harness 2.0

> Versão: 2  
> Data: 15 de agosto de 2026  
> Estado: vigente

Este documento governa a primeira release.

## Fontes executáveis e precedência

Os contratos consumíveis por código são:

- [`config/model-profiles.json`](../config/model-profiles.json), fonte de RuntimeProfiles e evidência de componentes;
- [`config/harness.json`](../config/harness.json), fonte de ExecutionRoutes, limites, policy, estado, stores, rede e UX;
- [`config/tool-registry.json`](../config/tool-registry.json), fonte única de `model_tools`, `internal_automations`, `prohibited_capabilities`, schemas e ResultPayload;
- [`evals/fixtures/regressions.json`](../evals/fixtures/regressions.json) e [`evals/experiments.json`](../evals/experiments.json), contratos versionados de avaliação;
- [`Modelfile`](../Modelfile), entrada da instalação Ollama local.

Em divergência, prevalecem: invariantes de segurança deste documento, contratos JSON e demais decisões desta página, nessa ordem. Alterar uma decisão exige atualizar contratos, fixtures e digests no mesmo conjunto.

## Vocabulário e fronteiras

O vocabulário canônico está em [`CONTEXT.md`](../CONTEXT.md). Em particular:

- RuntimeProfile identifica modelo, runtime, componentes embutidos, plataforma e capacidades observadas; não contém política de um Turn;
- ExecutionRoute seleciona um RuntimeProfile e fixa amostragem, tools, limites e reasoning para uma classe de execução;
- TerminalOutcome registra por que o Turn parou; TaskVerdict é o julgamento posterior do eval e nunca sobrescreve o desfecho;
- CanonicalHistory é estado autoritativo; ModelView e AG-UI são projeções derivadas.

## Baseline da primeira release

| Área | Decisão vigente |
|---|---|
| Runtime | Ollama local é o único runtime funcional; llama.cpp (`llama-server`) é o runtime dos Challengers que o Ollama não serve, verificado pelo SHA-256 do GGUF carregado ([ADR-0013](adr/0013-second-local-runtime-llama-cpp.md)); vLLM e remoto são candidatos sem rota ativa |
| Challengers | Um RuntimeProfile pode existir com `status: challenger`: instalado, com digest medido e `release_eligible: false`. Nenhuma ExecutionRoute de produção o seleciona; só o braço de experimento que o nomeia em `runtime_profile`, e o braço troca o modelo e nada mais. Continua havendo exatamente um perfil `functional`, e um challenger só o substitui pelo protocolo de promoção |
| Perfil ativo | `local_mitos_ollama_reproduction`, digest `a726cef53a75e7def1272308967d836c4c61092760feea3782a6007882ba5a74` |
| Instalação | O perfil instalado foi recriado e está coerente com o `Modelfile` de SHA-256 `8f7f9e8e9b6d16539242c3e52ef37adfc3f0d0e00b7ac8a9a533c3c5f4deec7b` |
| Componentes | Digest próprio ou evidência discriminada de componente embutido; ausência nunca é convertida em hash inventado |
| Tokenizer do estimador | `HuggingFaceTokenEstimator` lê um `tokenizer.json` local ao host, fixado por `host.json#tokenizer_digest` e registrado como componente `token_estimator_tokenizer_file`. Vocabulário e merges são idênticos aos do repositório upstream; os sete tokens de áudio/TTS a mais lá não ocorrem em texto, então a contagem é equivalente e os digests não. Como o arquivo não é versionado, esse digest é registro de procedência, não gate de CI |
| Capacidades | Cada capacidade declara `support`, `evidence` e `gate_status`; declaração de runtime não equivale a gate aprovado |
| Plataforma | Python 3.13, Linux x86_64 |
| ExecutionRoute | `local_web_tools`, web-first, sampling local `temperature=0.3`, `presence_penalty=0`, `think=true`. `sampling.thinking` é o que chega ao runtime em cada AgentStep, e só vale `true` com a capacidade `reasoning` suportada e aprovada no perfil |
| Loop | 15 AgentSteps, até 4 calls por AgentStep, 20 calls por Turn, 40 calls de leitura por Turn, 15 minutos de Turn, 300 segundos por geração do modelo, 2 tentativas malformadas e 8.192 tokens de saída |
| Execução de calls | Tool loop sem streaming; calls de um mesmo AgentStep são executadas em ordem, sem paralelismo |
| Último passo | Nenhuma tool é oferecida e o TerminalOutcome é persistido uma única vez |
| Tools | Somente `model_tools` são model-selectable; automações internas e capacidades proibidas são coleções separadas. Qual executor recebe a call vem do efeito declarado, nunca do nome |
| Resultados | Todo ToolResult contém `status`, `retryable`, `data`, `error` e `meta`; `blocked` só pode ser emitido pelo harness |
| Grants | Workspace requer WorkspaceRootGrant; escrita também exige WriteGrant; rede também exige WebAccessGrant; Corpus exige CorpusGrant do corpus escolhido |
| Rede | Toda operação web é efeito `data_egress`, negado por padrão e autorizado mecanicamente |
| Web | Busca por SearXNG declarado pelo Operator, com DuckDuckGo sem chave como fallback; navegador Chromium local; HTTP pode anteceder browser dentro do executor, nunca por escolha do modelo |
| Isolamento | Cada operação usa contexto de navegador efêmero; web e verificação de página não compartilham estado |
| Contexto | Deduplicação, extração única de HTML e corte por orçamento; compressão de código desligada e experimental |
| Formato do ModelView | JSON minificado em vigor; XML segue como braço candidato em `context_builder`. É decisão medida, não estética: `model_view_format` está em `freeze_per_run`, então cada execução congela o formato que usou, e [ADR-0010](adr/0010-model-facing-messages-are-xml.md) foi revertida pelo gate que ela mesma declarou |
| Corpus | Acervo curado pelo Operator, um store por Corpus; recuperação híbrida com piso de relevância, injetada antes do primeiro AgentStep e disponível como `corpus_search` ([ADR-0011](adr/0011-corpus-retrieval-and-corpus-grant.md)) |
| Estado | CanonicalHistory completo em store conversacional; reasoning é transitório e nunca persistido |
| Stores | Estado canônico e telemetria sem conteúdo em bancos separados; cada Corpus em um arquivo próprio, isolado dos dois |
| Retenção | Uma policy global remove Conversation inteira; nunca cria buracos no histórico; Corpus não é estado conversacional e só sai por exclusão explícita |
| UI | AG-UI é projeção do estado, não fonte canônica; UX e evals são web-first |
| Acesso | Sem senha de Operator, só loopback direto é atendido; com senha, toda rota exige sessão. Não há terceira opção |
| Confirmação | Todo efeito `workspace_write` exige decisão do Operator para aquela chamada, e sob UntrustedWebTaint `data_egress` também; o gate lê o efeito no registry, nunca o nome da tool; sob taint, aprovar não cria grant nem amplia acesso |
| Concessão | Falta de WriteGrant ou de WebAccessGrant é perguntada, não fatal: o Operator concede aprovando o diálogo, sem reenviar o prompt, e o grant continua exigido pela policy e revogável. WorkspaceRootGrant não: ele vem do allowlist do servidor e se resolve nas Configurações |
| Dispensa | O Operator pode dispensar a confirmação de `workspace_write` numa Conversation; é ato próprio e revogável, nunca efeito colateral de aprovar, e não cobre chamada sob taint |
| Modo yolo | Decisão permanente do Operator, global, desligada por padrão, com opt-out por Conversation: enquanto ligada o gate aprova toda confirmação — inclusive sob UntrustedWebTaint — e concede os grants que faltarem. Cada chamada é registrada como `waived`, nunca como `approved` ([ADR-0008](adr/0008-operator-yolo-mode.md)) |
| Roadmap | Qwen2.5 está fora do roadmap e não é challenger de nenhum experimento |

## Estado, projeções e persistência

O store conversacional persiste PendingRequest, Turn, AgentStep, tool calls, ToolResult, resposta final e TerminalOutcome. Reasoning pode transitar ao vivo para a projeção AG-UI, mas não entra no CanonicalHistory, telemetria ou replay. ModelView é reconstruída antes de cada passo e pode deduplicar, extrair HTML e cortar blocos sem alterar o registro.

A telemetria usa outro store e recebe somente IDs, digests, classes, tamanhos, contagens e tempos. Falha de telemetria é não fatal. Retenção é uniforme e global; quando aplicada ao estado conversacional, sua unidade mínima é uma Conversation completa.

Cada Corpus é um terceiro tipo de store, com um arquivo por acervo. Ele guarda conteúdo, como o canônico, mas não é histórico de ninguém: não participa da retenção, não recebe telemetria e some por exclusão explícita do Operator.

## Policy por efeitos

Nomes de tools não autorizam nada. A policy resolve os efeitos declarados no registry:

- `workspace_read` exige WorkspaceRootGrant;
- `workspace_write` exige WorkspaceRootGrant e WriteGrant;
- `data_egress` exige WebAccessGrant e controles de destino, DNS e redirect;
- `corpus_read` exige CorpusGrant, que nomeia em seu escopo o Corpus autorizado;
- `pure_compute` não exige grant algum, porque não lê, não escreve e não sai do host.

WebAccessGrant não é consentimento para backend remoto. Conteúdo obtido da web recebe UntrustedWebTaint, que acompanha derivações e nunca cria grant, confirmação ou permissão. Uma página hostil pode instruir o modelo tanto a alterar arquivos quanto a levá-los embora numa consulta ou URL, então as duas pernas passam pela mesma confirmação enquanto o taint estiver no contexto, e um efeito desconhecido é tratado como se precisasse dela. Paths continuam relativos, canonicalizados, com symlinks resolvidos e confinados ao Workspace.

Toda `workspace_write` para no Operator, que decide vendo o diff que o executor produziu — aprovar argumentos que ninguém leu não é aprovar nada. A pergunta chega depois do preflight: um diálogo sobre chamada que jamais rodaria não ensina nada, e o preview precisa de um path já canonicalizado.

Grant e confirmação já foram duas perguntas: ligar WriteGrant antes de qualquer coisa acontecer e depois aprovar a escrita. É uma decisão cobrada duas vezes, e a primeira é feita às cegas. Então a falta de WriteGrant não encerra mais o Turn: ela é perguntada, com o diff à vista, e aprovar concede o grant. A policy não mudou — `workspace_write` continua exigindo WorkspaceRootGrant e WriteGrant, o grant continua visível e revogável, e revogar durante o Turn continua sendo revalidado antes do efeito. Mudou onde o Operator concede. Aprovar sob UntrustedWebTaint segue sem criar grant algum: ali a pergunta é outra e a resposta vale para uma chamada só.

Quem não quer ser perguntado a cada escrita registra uma dispensa para a Conversation, que é revogável, aparece no estado como qualquer autorização e vale só para a escrita sem taint. Quem quer um agente que não pare nunca liga o modo yolo, que é a decisão oposta e igualmente explícita: enquanto estiver ligada, toda confirmação é respondida com sim e os grants que faltarem são concedidos, inclusive quando a escrita vem de conteúdo da web. O risco aceito ao ligar está em [ADR-0008](adr/0008-operator-yolo-mode.md), o padrão é desligado, e o histórico continua distinguindo `waived` de `approved`. Nenhum caminho de decisão lê nome de tool: quem decide é o efeito declarado em [`config/tool-registry.json`](../config/tool-registry.json), configurado em [`config/harness.json#policy`](../config/harness.json).

## Tools, automações e resultados

[`config/tool-registry.json`](../config/tool-registry.json) é a única fonte de nomes, schemas, efeitos, grants e envelopes. O registry contém exatamente três coleções conceituais:

1. `model_tools`: operações que podem aparecer para o modelo;
2. `internal_automations`: comportamentos controlados pelo harness;
3. `prohibited_capabilities`: efeitos que a primeira release não executa.

`blocked` significa que o harness recusou a operação por policy, grant, validação ou limite. Indisponibilidade ou recusa de provedor é `failed`, com classe em `error`; `empty` é sucesso sem itens. Nenhum deles pode virar string vazia ambígua.

A mesma separação vale no TerminalOutcome do Turn: provedor indisponível, provedor que recusa e geração cortada por timeout do harness têm cada um seu reason code, e o `detail` carrega a classe que o runtime reportou. O timeout é cláusula própria e vem antes da de provedor de propósito: quem cortou a geração foi o harness, e chamar isso de provedor indisponível mandava o Operator reiniciar um Ollama que estava no ar. Erro interno do harness continua sendo `engine_error` e não se disfarça de problema do provedor — quem lê o outcome precisa saber se reinicia o runtime ou abre um bug.

## Corpus e recuperação

Um Corpus é acervo do Operator, não estado de conversa: vive em um arquivo SQLite
próprio sob `state_dir/corpora/`, guarda seu próprio meta — e por isso o diretório
é o índice, sem registro paralelo para discordar do disco. Apagar um Corpus apaga
o arquivo. A retenção global, cuja unidade é a Conversation inteira, não o alcança.

A recuperação entra por duas portas sobre um pipeline só. Havendo CorpusGrant, uma
InternalAutomation recupera antes do primeiro AgentStep e injeta o resultado no
CanonicalHistory; `corpus_search` fica exposta para o modelo refinar a busca nos
passos seguintes. A busca funde vizinhança densa e BM25 por rank recíproco, corta
pelo piso de relevância e devolve cada Chunk com Document, endereço e escore —
nada acima do piso é `empty`, e a instrução de responder que não sabe viaja no
próprio bloco injetado, não no prompt base. O piso vale para cada passagem
citada, não para a busca como um todo: quem entrou só pela perna lexical também
é medido, senão o BM25 injeta o que ninguém conferiu. A perna lexical recebe as
duas formas da pergunta, a do Operator e a reescrita em inglês, porque o idioma
do acervo é do acervo — e o que a forma errada trouxer não passa do piso.

Nenhum fato ingerido passa por paráfrase: o Document é armazenado como foi
extraído e limpo por regras determinísticas, e o que o Chunk acrescenta é um
prefixo de contexto tirado da estrutura do próprio documento. Um Chunk não
começa no meio de um parágrafo; quando o extrator não consegue dizer onde o
parágrafo termina — o texto de uma página de PDF chega sem essa marca —, o bloco
que sozinho estoura o orçamento é cortado em fim de frase, que é a promessa que
a regra fazia. Uma seção mais
curta que `chunk_minimum_tokens` não vira Chunk — o texto continua no Document,
mas para de disputar as poucas passagens que cabem num Turn, porque lista de
links casa com a pergunta ao pé da letra sem responder nada. Tradução existe só
na query. Chunk vindo do scraper carrega UntrustedWebTaint e o ToolResult declara
a união dos taints que devolveu, de modo que material coletado da web continua
custando confirmação de `data_egress` enquanto estiver no contexto. Coletar é ato
do Operator e não pede grant — grants contêm o modelo —, mas passa pelo mesmo
EgressGuard de qualquer saída para a rede ([ADR-0011](adr/0011-corpus-retrieval-and-corpus-grant.md)).

## Web e browser

`web_search` consulta a instância SearXNG que o Operator declarar e cai para o DuckDuckGo quando ela não responde ou não foi configurada — nenhum dos dois exige credencial ([ADR-0009](adr/0009-search-provider-searxng-with-fallback.md)). `web_fetch` tenta HTTP guardado e pode escalar internamente para um Chromium local por sintoma. Ambos são `data_egress`, revalidam SSRF em redirects e produzem dados com UntrustedWebTaint.

Cada execução de browser recebe contexto efêmero. A automação de PageRevision e o acesso web usam contextos distintos, sem cookies, cache, storage ou service workers compartilhados. O risco residual de DNS rebinding está documentado em [`RELEASE-PENDING.md`](RELEASE-PENDING.md).

## Verificação de página

Verificação de página é InternalAutomation pós-escrita e nunca tool do modelo. Sua chave de idempotência é PageRevision, composta por `workspace_id`, `relative_path`, `html_sha256`, `workspace_revision` e `verifier_digest`. O modo candidato `automatic_once_per_page_revision` começa desligado; promoção exige o experimento homônimo e atualização explícita deste contrato.

## Release e avaliações

A primeira superfície de produto é web e toda promoção exige tarefa fim a fim nessa superfície. Fixtures e experimentos declaram `dataset_version`, digest calculável e vínculo com RuntimeProfile e ExecutionRoute. Resultado de experimento pode ser ausente; inventar medição para preencher contrato é proibido.

O que não entra, seus motivos e gates está em [`RELEASE-PENDING.md`](RELEASE-PENDING.md). Isso inclui vLLM, remoto, visão, shell, streaming, paralelismo, compressão de código, branching, approve-with-edits, dark mode e a retirada de Qwen2.5 do roadmap.

## Evidência preservada

A pesquisa e a herança empírica do Harness 1.0 que originaram estas decisões
foram retiradas do repositório; o que sobreviveu delas está aqui, nos ADRs e na
proveniência declarada em `config/model-profiles.json` e
`evals/fixtures/regressions.json`. Histórico completo permanece no git.
