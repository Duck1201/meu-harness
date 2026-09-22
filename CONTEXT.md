# Harness 2.0

## Language

**Operator**:
A pessoa com autoridade para iniciar trabalho, conceder acesso e decidir se um efeito pode ocorrer.
_Avoid_: User, admin, approver

**WorkspaceRootGrant**:
A autorização que vincula uma raiz de arquivos a uma conversa e estabelece o limite do trabalho local.
_Avoid_: Workspace permission, path jail

**Workspace**:
O conjunto de arquivos e revisões alcançáveis sob um WorkspaceRootGrant.
_Avoid_: Repository, working directory, sandbox

**Conversation**:
A interação durável entre um Operator e o harness, composta por solicitações e turnos relacionados.
_Avoid_: Session, chat, thread

**PendingRequest**:
Uma solicitação do Operator aceita e enfileirada que ainda não iniciou um Turn.
_Avoid_: Prompt, job, task

**Turn**:
O intervalo de trabalho que tenta resolver um PendingRequest e termina com exatamente um TerminalOutcome.
_Avoid_: Message, run, request

**AgentStep**:
Uma unidade ordenada de decisão do agente dentro de um Turn, com resposta final ou chamadas de tools propostas.
_Avoid_: Iteration, cycle, hop

**ToolResult**:
A conclusão estruturada de uma ToolCall, distinguindo estado, nova tentativa, payload, erro e metadados.
_Avoid_: Tool response, observation

**ResultPayload**:
A parte deduplicável de um ToolResult que contém os dados produzidos sem substituir sua proveniência.
_Avoid_: Raw result, output string, exception

**CanonicalHistory**:
A sequência autoritativa e sem lacunas dos fatos duráveis de uma Conversation.
_Avoid_: Chat history, transcript, context

**ModelView**:
Uma apresentação derivada e descartável do CanonicalHistory, limitada ao que um passo do modelo precisa e pode receber.
_Avoid_: Memory, compressed history, canonical context

**RuntimeProfile**:
A identidade reproduzível de uma combinação de modelo, runtime, componentes embutidos, plataforma e capacidades observadas.
_Avoid_: Backend, model config, route

**Challenger**:
Um RuntimeProfile instalado e medido que nenhuma ExecutionRoute de produção seleciona; só entra em execução como braço de experimento e só vira o perfil funcional por promoção.
_Avoid_: Candidate model, backup profile, alternative route

**ExecutionRoute**:
A política de execução que seleciona um RuntimeProfile e fixa limites, amostragem, tools e tratamento de raciocínio para um tipo de trabalho.
_Avoid_: RuntimeProfile, backend, endpoint

**TerminalOutcome**:
A categoria mecânica e imutável pela qual um Turn terminou, acompanhada de um reason code específico.
_Avoid_: Task result, score, verdict

**TaskVerdict**:
O julgamento de uma avaliação sobre se o comportamento observado satisfez o oráculo da tarefa. Inconclusivo significa que não deu para julgar — infraestrutura caiu, interrupção veio de fora — e nunca que o modelo teve orçamento e não entregou, o que é falha.
_Avoid_: TerminalOutcome, status, exit reason

**WriteGrant**:
A autorização explícita para efeitos que alteram um Workspace dentro de um WorkspaceRootGrant.
_Avoid_: Write mode, file permission, approval

**WebAccessGrant**:
A autorização explícita para efeitos de data egress destinados à web pública.
_Avoid_: Network enabled, internet access, remote consent

**UntrustedWebTaint**:
A marca de proveniência aplicada a dados originados da web e preservada em suas derivações.
_Avoid_: Unsafe text, prompt injection flag, untrusted boolean

**InternalAutomation**:
Um comportamento controlado pelo harness que não pode ser selecionado nem repetido diretamente pelo modelo.
_Avoid_: Hidden tool, system tool, background agent

**PageRevision**:
A identidade imutável de uma página em um Workspace, formada pelo workspace, path relativo, digest do HTML, revisão do workspace e digest do verificador.
_Avoid_: Page SHA, screenshot cache key, file version

**Corpus**:
O conjunto durável de Documents reunidos pelo Operator sob uma identidade própria e um store isolado dos demais.
_Avoid_: RAG, base de conhecimento, índice, coleção

**Document**:
A unidade ingerida em um Corpus, com o texto extraído e a proveniência que diz de onde ele veio e quando.
_Avoid_: Arquivo, página, fonte

**Chunk**:
O trecho recuperável de um Document, unidade do embedding, do ranking e da citação.
_Avoid_: Passage, fragmento, janela

**CorpusGrant**:
A autorização que vincula um Corpus a uma Conversation e habilita a recuperação naquele Turn.
_Avoid_: Corpus selecionado, toggle de RAG, preferência
