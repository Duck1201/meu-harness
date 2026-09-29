import type { HarnessClient } from "./HarnessClient";
import type {
  AgUiEvent,
  HostConfigSnapshot,
  ApiPendingRequest,
  ChatSnapshot,
  CorporaSnapshot,
  Conversation,
  Corpus,
  CorpusDocument,
  IngestionJob,
  EvalReport,
  EvalRun,
  EvalsSnapshot,
  Experiment,
  FeedbackRecord,
  Grant,
  HealthSnapshot,
  PendingConfirmation,
  RegressionDraft,
  RunAgentInput,
  SessionStatus,
  SettingsSnapshot,
  SetupStatus,
  SetupSubmission,
  Workspace,
} from "../types";

const now = "2026-08-10T10:43:00Z";

const initialWorkspaces: Workspace[] = [
  { id: "workspace-harness", root: "/workspaces/meu-harness" },
  { id: "workspace-bench", root: "/workspaces/bench" },
];

const initialConversations: Conversation[] = [
  {
    id: "chat-128",
    workspace_id: "workspace-harness",
    name: "Refinar retenção por conversa",
    created_at: now,
    updated_at: now,
    last_active_at: now,
    archived_at: null,
  },
  {
    id: "chat-127",
    workspace_id: "workspace-harness",
    name: "Verificação de página",
    created_at: now,
    updated_at: now,
    last_active_at: now,
    archived_at: null,
  },
];

const initialGrants: Grant[] = [
  {
    id: "grant-root",
    conversation_id: "chat-128",
    permission: "WorkspaceRootGrant",
    scope: "/workspaces/meu-harness",
    granted_at: now,
    expires_at: null,
  },
  {
    id: "grant-write",
    conversation_id: "chat-128",
    permission: "WriteGrant",
    scope: "workspace",
    granted_at: now,
    expires_at: null,
  },
  {
    id: "grant-web",
    conversation_id: "chat-128",
    permission: "WebAccessGrant",
    scope: "public-network",
    granted_at: now,
    expires_at: null,
  },
];

const health: HealthSnapshot = {
  ready: true,
  reason_code: null,
  capabilities: {
    settings_mutation: false,
    admin: false,
    ag_ui_sse: true,
    eval_runner: true,
    static_spa: true,
  },
};

const experiments: Experiment[] = [
  {
    id: "page_verification",
    status: "active",
    runtime_profile: "gemma4_e4b_qat_ollama",
    execution_route: "local_web_tools",
    fixture_tags: ["page"],
    arms: [{ id: "control" }, { id: "candidate" }],
  },
];

const initialRuns: EvalRun[] = [
  {
    id: "run-page-12",
    experiment_id: "page_verification",
    tier: "experiment",
    phase: "promotion",
    status: "completed",
    seeds: [3],
    reason_code: null,
    created_at: now,
    updated_at: now,
  },
];

const initialReports: EvalReport[] = [
  {
    id: "report-page-12",
    run_id: "run-page-12",
    payload: { status: "completed", task_success: 0.86 },
    created_at: now,
  },
];

const clone = <T,>(value: T): T => structuredClone(value);

export class MockHarnessClient implements HarnessClient {
  private workspaces = clone(initialWorkspaces);
  private conversations = clone(initialConversations);
  private grants = clone(initialGrants);
  private pending: ApiPendingRequest[] = [
    {
      id: "pending-1",
      conversation_id: "chat-128",
      sequence: 1,
      content: "Depois, confira se o relatório distingue expired de deleted.",
      status: "queued",
      created_at: now,
      updated_at: now,
    },
  ];
  private feedback: FeedbackRecord[] = [];
  private confirmations = new Map<string, PendingConfirmation>();
  private waivers = new Set<string>();
  private corpora: Corpus[] = [
    {
      id: "manual-do-servidor",
      name: "Manual do servidor",
      description: "Notas de operação e rede",
      embedding_model: "bge-m3:latest",
      embedding_dimensions: 1024,
      document_count: 3,
      chunk_count: 42,
      created_at: now,
      updated_at: now,
    },
  ];
  private corpusDocuments: CorpusDocument[] = [
    {
      id: "document-1",
      origin_kind: "upload",
      origin_ref: "manual.pdf",
      title: "Manual do servidor",
      source_digest: "a".repeat(64),
      taints: [],
      chunk_count: 42,
      ingested_at: now,
    },
  ];
  private corpusJobs: IngestionJob[] = [];
  private selectedCorpus = new Map<string, string | null>();
  private yolo = false;
  private hostConfig: HostConfigSnapshot = {
    allowed_workspace_roots: ["/workspaces/meu-harness"],
    tokenizer_path: "/home/operator/.local/state/meu-harness/tokenizer.json",
    tokenizer_digest: "0".repeat(64),
    state_dir: "/home/operator/.local/state/meu-harness",
    allowed_origins: ["http://127.0.0.1:8765"],
    searxng_url: null,
    ollama_url: "http://127.0.0.1:11434",
    browser_executable: null,
  };
  private yoloDisabled = new Set<string>();
  private password = "";
  private sessionStatus: SessionStatus = {
    authentication_required: false,
    authenticated: false,
    expires_at: null,
  };
  private setupStatus: SetupStatus = {
    configured: true,
    required: false,
    restart_required: false,
    token_expires_at: null,
    suggested_state_dir: "/home/operator/.local/state/meu-harness",
    suggested_tokenizer_path: "/home/operator/.local/state/meu-harness/tokenizer.json",
  };
  private runs = clone(initialRuns);
  private reports = clone(initialReports);
  private drafts: RegressionDraft[] = [];
  private selectedConversationId = "chat-128";
  private messages: ChatSnapshot["messages"] = [
    {
      id: "message-user-1",
      turnId: "turn-1",
      role: "user",
      content:
        "Mapeie onde a retenção parcial ainda aparece e proponha a menor mudança para manter conversas inteiras.",
      createdAt: now,
    },
    {
      id: "message-assistant-1",
      turnId: "turn-1",
      role: "assistant",
      content:
        "Encontrei a configuração de retenção e a decisão normativa. A regra consistente é expirar a conversa inteira por idade, sem remover mensagens de um histórico vivo.",
      createdAt: now,
      tools: [
        {
          id: "tool-1",
          name: "grep_search",
          label: "grep_search",
          status: "success",
          arguments: { pattern: "retention|retenção", path: "." },
          summary: "success",
          result: "12 ocorrências em 4 arquivos",
        },
      ],
      terminalOutcome: {
        kind: "completed",
        reason_code: "final_response",
        recorded_at: now,
        detail: null,
      },
    },
  ];

  async getHealth() {
    return clone(health);
  }

  async listWorkspaces() {
    return clone(this.workspaces);
  }

  async listConversations(includeArchived = false) {
    return clone(
      this.conversations.filter(
        (conversation) => includeArchived || conversation.archived_at === null,
      ),
    );
  }

  async getConversation(conversationId: string) {
    const conversation = this.requireConversation(conversationId);
    return clone(conversation);
  }

  async createConversation(workspaceRoot: string, name = "New conversation") {
    const workspace = this.workspaces.find((item) => item.root === workspaceRoot);
    if (!workspace) throw new Error("workspace_root_not_allowed");
    const conversation: Conversation = {
      id: `chat-${Date.now()}`,
      workspace_id: workspace.id,
      name,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      last_active_at: new Date().toISOString(),
      archived_at: null,
    };
    this.conversations.unshift(conversation);
    this.grants.push({
      id: `grant-root-${conversation.id}`,
      conversation_id: conversation.id,
      permission: "WorkspaceRootGrant",
      scope: workspace.root,
      granted_at: conversation.created_at,
      expires_at: null,
    });
    return clone(conversation);
  }

  async selectConversation(conversationId: string) {
    this.requireConversation(conversationId);
    this.selectedConversationId = conversationId;
    return this.getChatSnapshot(conversationId);
  }

  async renameConversation(conversationId: string, name: string) {
    const conversation = this.requireConversation(conversationId);
    conversation.name = name;
    conversation.updated_at = new Date().toISOString();
    return clone(conversation);
  }

  async archiveConversation(conversationId: string, archived = true) {
    const conversation = this.requireConversation(conversationId);
    conversation.archived_at = archived ? new Date().toISOString() : null;
    return clone(conversation);
  }

  async deleteAllConversations() {
    const deleted = this.conversations.length;
    this.conversations = [];
    return deleted;
  }

  async deleteConversation(conversationId: string) {
    this.requireConversation(conversationId);
    this.conversations = this.conversations.filter((item) => item.id !== conversationId);
  }

  async getChatSnapshot(conversationId?: string | null): Promise<ChatSnapshot> {
    const visible = this.conversations.filter((item) => item.archived_at === null);
    const selected = conversationId
      ? this.requireConversation(conversationId)
      : visible.find((item) => item.id === this.selectedConversationId) ?? visible[0];
    if (selected) this.selectedConversationId = selected.id;
    const workspace = selected
      ? this.workspaces.find((item) => item.id === selected.workspace_id)
      : undefined;
    return clone({
      conversationId: selected?.id ?? null,
      conversationTitle: selected?.name ?? null,
      workspaceName: workspace?.root.split("/").at(-1) ?? null,
      workspaces: this.workspaces.map((item) => ({
        id: item.id,
        root: item.root,
        name: item.root.split("/").at(-1) ?? item.root,
        conversations: visible
          .filter((conversation) => conversation.workspace_id === item.id)
          .map((conversation) => ({
            id: conversation.id,
            title: conversation.name,
            updatedAt: conversation.last_active_at,
            active: conversation.id === selected?.id,
          })),
      })),
      grants: selected
        ? this.grants.filter((grant) => grant.conversation_id === selected.id)
        : [],
      messages: selected?.id === "chat-128" ? this.messages : [],
      pendingRequests: selected
        ? this.pending
            .filter(
              (request) =>
                request.conversation_id === selected.id && request.status === "queued",
            )
            .map((request) => ({
              id: request.id,
              prompt: request.content,
              createdAt: request.created_at,
              sequence: request.sequence,
              status: request.status,
              mode: "queued" as const,
            }))
        : [],
      activeTurn: null,
      turns: [
        {
          id: "turn-1",
          conversation_id: "chat-128",
          request_id: "request-1",
          status: "finished",
          started_at: now,
          ended_at: now,
          terminal_outcome: {
            kind: "completed",
            reason_code: "final_response",
            recorded_at: now,
            detail: null,
          },
        },
      ],
      feedback: selected
        ? this.feedback.filter((item) => item.conversation_id === selected.id)
        : [],
      pendingConfirmation: selected ? this.confirmations.get(selected.id) ?? null : null,
      confirmationWaivers:
        selected && this.waivers.has(selected.id) ? ["workspace_write"] : [],
      yolo: this.yolo && !(selected ? this.yoloDisabled.has(selected.id) : false),
      corpusId: selected ? this.selectedCorpus.get(selected.id) ?? null : null,
      corpora: this.corpora,
      execution: {
        defaultExecutionRoute: "local_web_tools",
        runtimeProfile: "gemma4_e4b_qat_ollama",
        loop: {
          max_steps: 15,
          max_tool_calls_per_step: 4,
          max_tool_calls_per_turn: 12,
          max_turn_duration_seconds: 300,
          max_output_tokens: 4096,
          offer_tools_on_final_step: false,
        },
      },
    });
  }

  /** Troca a conversa de exemplo; é dublê de teste, não entra no bundle. */
  setMessages(messages: ChatSnapshot["messages"]): void {
    this.messages = messages;
  }

  async getCorporaSnapshot(): Promise<CorporaSnapshot> {
    return clone({
      available: true,
      embedding_model: "bge-m3:latest",
      accepted_extensions: [".txt", ".md", ".html", ".pdf"],
      corpora: this.corpora,
      jobs: this.corpusJobs,
    });
  }

  async createCorpus(name: string, description = "") {
    const corpus: Corpus = {
      id: name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""),
      name,
      description,
      embedding_model: "bge-m3:latest",
      embedding_dimensions: 1024,
      document_count: 0,
      chunk_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    this.corpora = [...this.corpora, corpus];
    return clone(corpus);
  }

  async renameCorpus(corpusId: string, changes: { name?: string; description?: string }) {
    const corpus = this.requireCorpus(corpusId);
    Object.assign(corpus, changes);
    return clone(corpus);
  }

  async deleteCorpus(corpusId: string) {
    this.requireCorpus(corpusId);
    this.corpora = this.corpora.filter((item) => item.id !== corpusId);
  }

  async listCorpusDocuments(corpusId: string) {
    this.requireCorpus(corpusId);
    return clone(this.corpusDocuments);
  }

  async deleteCorpusDocument(corpusId: string, documentId: string) {
    this.requireCorpus(corpusId);
    this.corpusDocuments = this.corpusDocuments.filter((item) => item.id !== documentId);
  }

  async attachImage(conversationId: string, file: File) {
    if (!this.conversations.some((item) => item.id === conversationId)) {
      throw new Error("conversation_not_found");
    }
    return `anexos/${file.name}`;
  }

  async uploadCorpusDocument(corpusId: string, file: File) {
    const corpus = this.requireCorpus(corpusId);
    corpus.document_count += 1;
    const job: IngestionJob = {
      id: `job-${this.corpusJobs.length + 1}`,
      corpus_id: corpusId,
      kind: "upload",
      origin: file.name,
      status: "completed",
      seen: 1,
      indexed: 1,
      skipped: 0,
      chunks: 4,
      current: null,
      reason_code: null,
      detail: null,
    };
    this.corpusJobs = [...this.corpusJobs, job];
    return clone(job);
  }

  async startCorpusScrape(corpusId: string, seed: string) {
    this.requireCorpus(corpusId);
    const job: IngestionJob = {
      id: `job-${this.corpusJobs.length + 1}`,
      corpus_id: corpusId,
      kind: "scrape",
      origin: seed,
      status: "running",
      seen: 3,
      indexed: 2,
      skipped: 1,
      chunks: 18,
      current: seed,
      reason_code: null,
      detail: "mediawiki",
    };
    this.corpusJobs = [...this.corpusJobs, job];
    return clone(job);
  }

  async listCorpusJobs(corpusId: string) {
    return clone(this.corpusJobs.filter((job) => job.corpus_id === corpusId));
  }

  async cancelCorpusJob(corpusId: string, jobId: string) {
    const job = this.corpusJobs.find(
      (item) => item.id === jobId && item.corpus_id === corpusId,
    );
    if (!job) throw new Error(`Job inexistente: ${jobId}`);
    job.status = "canceled";
    job.current = null;
    return clone(job);
  }

  async selectCorpus(conversationId: string, corpusId: string | null) {
    this.requireConversation(conversationId);
    this.selectedCorpus.set(conversationId, corpusId);
  }

  private requireCorpus(corpusId: string): Corpus {
    const corpus = this.corpora.find((item) => item.id === corpusId);
    if (!corpus) throw new Error(`Corpus inexistente: ${corpusId}`);
    return corpus;
  }

  async enqueueRequest(conversationId: string, content: string) {
    this.requireConversation(conversationId);
    const request: ApiPendingRequest = {
      id: `pending-${Date.now()}`,
      conversation_id: conversationId,
      sequence: this.pending.length + 1,
      content,
      status: "queued",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    this.pending.push(request);
    return clone(request);
  }

  async streamAgent(
    input: RunAgentInput,
    onEvent: (event: AgUiEvent) => void,
    signal?: AbortSignal,
  ) {
    const runId = input.runId ?? `run-${Date.now()}`;
    const userMessage = [...input.messages]
      .reverse()
      .find((message) => message.role === "user");
    const events: AgUiEvent[] = [
      { type: "RUN_STARTED", threadId: input.threadId, runId },
      { type: "STEP_STARTED", stepName: "step-1" },
      {
        type: "TEXT_MESSAGE_START",
        messageId: `${runId}-assistant`,
        role: "assistant",
      },
      {
        type: "TEXT_MESSAGE_CONTENT",
        messageId: `${runId}-assistant`,
        delta: "Resposta simulada pelo MockHarnessClient.",
      },
      { type: "TEXT_MESSAGE_END", messageId: `${runId}-assistant` },
      { type: "STEP_FINISHED", stepName: "step-1" },
      { type: "RUN_FINISHED", threadId: input.threadId, runId },
    ];
    for (const event of events) {
      signal?.throwIfAborted();
      onEvent(event);
      await Promise.resolve();
    }
    if (userMessage?.content) {
      const turnId = `turn-${runId}`;
      this.messages.push(
        {
          id: `user-${runId}`,
          turnId,
          role: "user",
          content: userMessage.content,
          createdAt: new Date().toISOString(),
        },
        {
          id: `assistant-${runId}`,
          turnId,
          role: "assistant",
          content: "Resposta simulada pelo MockHarnessClient.",
          createdAt: new Date().toISOString(),
          terminalOutcome: {
            kind: "completed",
            reason_code: "final_response",
            recorded_at: new Date().toISOString(),
            detail: null,
          },
        },
      );
    }
    return events.at(-1) as Extract<
      AgUiEvent,
      { type: "RUN_FINISHED" | "RUN_ERROR" }
    >;
  }

  async editPendingRequest(
    conversationId: string,
    requestId: string,
    content: string,
  ) {
    const request = this.requirePending(conversationId, requestId);
    request.content = content;
    request.updated_at = new Date().toISOString();
    return clone(request);
  }

  async cancelPendingRequest(conversationId: string, requestId: string) {
    const request = this.requirePending(conversationId, requestId);
    request.status = "canceled";
    request.updated_at = new Date().toISOString();
    return clone(request);
  }

  async listGrants(conversationId: string) {
    return clone(this.grants.filter((grant) => grant.conversation_id === conversationId));
  }

  async addGrant(conversationId: string, permission: string) {
    this.requireConversation(conversationId);
    const grant: Grant = {
      id: `grant-${Date.now()}`,
      conversation_id: conversationId,
      permission,
      scope: permission === "WriteGrant" ? "workspace" : "public-network",
      granted_at: new Date().toISOString(),
      expires_at: null,
    };
    this.grants.push(grant);
    return clone(grant);
  }

  async revokeGrant(conversationId: string, grantId: string) {
    this.grants = this.grants.filter(
      (grant) => grant.conversation_id !== conversationId || grant.id !== grantId,
    );
  }

  async getPendingConfirmation(conversationId: string) {
    this.requireConversation(conversationId);
    return clone(this.confirmations.get(conversationId) ?? null);
  }

  async resolveConfirmation(
    conversationId: string,
    confirmationId: string,
    approved: boolean,
    waive = false,
  ) {
    const pending = this.confirmations.get(conversationId);
    if (!pending || pending.id !== confirmationId) {
      throw new Error("confirmation_not_pending");
    }
    if (approved && waive) this.waivers.add(conversationId);
    this.confirmations.delete(conversationId);
  }

  async revokeConfirmationWaiver(conversationId: string, _effect: string) {
    this.waivers.delete(conversationId);
  }

  /** Seeds a pending confirmation so the demo mode can exercise the approval card. */
  seedConfirmation(confirmation: PendingConfirmation) {
    this.confirmations.set(confirmation.conversation_id, confirmation);
  }

  async updateHostConfig(submission: SetupSubmission) {
    this.hostConfig = {
      allowed_workspace_roots: submission.allowed_workspace_roots,
      tokenizer_path: submission.tokenizer_path,
      tokenizer_digest: "0".repeat(64),
      state_dir: submission.state_dir,
      allowed_origins: submission.allowed_origins,
      searxng_url: submission.searxng_url ?? null,
      ollama_url: submission.ollama_url ?? "http://127.0.0.1:11434",
      browser_executable: submission.browser_executable ?? null,
    };
    return { restart_required: true };
  }

  async setYoloEnabled(enabled: boolean) {
    this.yolo = enabled;
    return undefined;
  }

  async setConversationYolo(conversationId: string, disabled: boolean) {
    if (disabled) this.yoloDisabled.add(conversationId);
    else this.yoloDisabled.delete(conversationId);
    return undefined;
  }

  async stop(_conversationId: string) {
    this.confirmations.delete(_conversationId);
    return undefined;
  }

  async addFeedback(
    conversationId: string,
    rating: number,
    turnId?: string,
    comment?: string,
  ) {
    const item: FeedbackRecord = {
      id: `feedback-${Date.now()}`,
      conversation_id: conversationId,
      turn_id: turnId ?? null,
      rating,
      comment: comment ?? null,
      created_at: new Date().toISOString(),
    };
    this.feedback.push(item);
    return clone(item);
  }

  async getEvalsSnapshot(): Promise<EvalsSnapshot> {
    return clone({
      experiments,
      runs: this.runs,
      reports: this.reports,
      capabilities: { eval_runner: true, progress_transport: "polling_json" },
      selectedRunId: this.runs[0]?.id ?? null,
    });
  }

  async listEvalExperiments() {
    return clone(experiments);
  }

  async listEvalRuns() {
    return clone(this.runs);
  }

  async getEvalRun(runId: string) {
    const run = this.requireRun(runId);
    return { run: clone(run), arms: [], cases: [], metrics: [] };
  }

  async createEvalRun(input: {
    experimentId: string;
    tier: "contract" | "model_smoke" | "experiment";
    phase: "pilot" | "promotion";
    seeds?: number[];
  }) {
    const run: EvalRun = {
      id: `run-${Date.now()}`,
      experiment_id: input.experimentId,
      tier: input.tier,
      phase: input.phase,
      status: "created",
      seeds: input.seeds ?? [],
      reason_code: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    this.runs.unshift(run);
    return clone(run);
  }

  async startEvalRun(runId: string) {
    const run = this.requireRun(runId);
    run.status = "queued";
    return clone(run);
  }

  async cancelEvalRun(runId: string) {
    const run = this.requireRun(runId);
    run.status = "canceled";
    return clone(run);
  }

  async listEvalReports() {
    return clone(this.reports);
  }

  async getEvalReport(runId: string) {
    const report = this.reports.find((item) => item.run_id === runId);
    if (!report) throw new Error("eval_report_not_found");
    return clone(report);
  }

  async listRegressionDrafts() {
    return clone(this.drafts);
  }

  async createRegressionDraft(_conversationId: string, feedbackId: string) {
    const feedback = this.feedback.find((item) => item.id === feedbackId);
    if (!feedback) throw new Error("feedback_not_found");
    const draft: RegressionDraft = {
      id: `draft-${Date.now()}`,
      source_feedback_sha256: "mock-feedback-digest",
      source_turn_sha256: "mock-turn-digest",
      rating: feedback.rating,
      comment_present: feedback.comment !== null,
      terminal_outcome_kind: "completed",
      terminal_outcome_reason: "final_response",
      created_at: new Date().toISOString(),
    };
    this.drafts.push(draft);
    return clone(draft);
  }

  async getSettingsSnapshot(): Promise<SettingsSnapshot> {
    return clone({
      health,
      workspaces: this.workspaces,
      mutable: true,
      host_config: this.hostConfig,
      default_execution_route: "local_web_tools",
      runtime_profile: "gemma4_e4b_qat_ollama",
      yolo_enabled: this.yolo,
      loop: {
        max_steps: 15,
        max_tool_calls_per_step: 4,
        max_tool_calls_per_turn: 12,
        max_turn_duration_seconds: 300,
        max_output_tokens: 4096,
        offer_tools_on_final_step: false,
      },
    });
  }

  async getSessionStatus() {
    return clone(this.sessionStatus);
  }

  async login(password: string) {
    if (password !== this.password) throw new Error("invalid_credentials");
    this.sessionStatus = {
      authentication_required: true,
      authenticated: true,
      expires_at: now,
    };
    return clone(this.sessionStatus);
  }

  async logout() {
    this.sessionStatus = { ...this.sessionStatus, authenticated: false, expires_at: null };
  }

  /** Puts the mock behind a login, the way a host with a password behaves. */
  seedAuthenticationRequired(password: string) {
    this.password = password;
    this.sessionStatus = {
      authentication_required: true,
      authenticated: false,
      expires_at: null,
    };
  }

  async getSetupStatus() {
    return clone(this.setupStatus);
  }

  async completeSetup(token: string, _submission: SetupSubmission) {
    if (!token) throw new Error("setup_token_required");
    this.setupStatus = {
      ...this.setupStatus,
      configured: true,
      required: false,
      restart_required: true,
      token_expires_at: null,
    };
  }

  /** Puts the mock back in the state a first boot presents. */
  seedSetupRequired(tokenExpiresAt: string | null = null) {
    this.setupStatus = {
      configured: false,
      required: true,
      restart_required: false,
      token_expires_at: tokenExpiresAt,
      suggested_state_dir: "/home/operator/.local/state/meu-harness",
      suggested_tokenizer_path: "/home/operator/.local/state/meu-harness/tokenizer.json",
    };
  }

  private requireConversation(conversationId: string) {
    const conversation = this.conversations.find((item) => item.id === conversationId);
    if (!conversation) throw new Error("conversation_not_found");
    return conversation;
  }

  private requirePending(conversationId: string, requestId: string) {
    const request = this.pending.find(
      (item) => item.id === requestId && item.conversation_id === conversationId,
    );
    if (!request) throw new Error("pending_request_not_found");
    return request;
  }

  private requireRun(runId: string) {
    const run = this.runs.find((item) => item.id === runId);
    if (!run) throw new Error("eval_run_not_found");
    return run;
  }
}
