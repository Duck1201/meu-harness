import type { HarnessClient } from "./HarnessClient";
import type {
  AgUiEvent,
  AgUiTerminalEvent,
  ApiChatSnapshot,
  ApiPendingRequest,
  CanonicalHistoryEntry,
  ChatMessage,
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
  JsonValue,
  PendingConfirmation,
  RegressionDraft,
  RunAgentInput,
  SessionStatus,
  SettingsSnapshot,
  SetupStatus,
  SetupSubmission,
  Retrieval,
  ToolCall,
  Turn,
  Workspace,
  WorkspaceGroup,
} from "../types";

type FetchImplementation = typeof fetch;

interface ApiErrorBody {
  error?: { code?: string; message?: string };
  detail?: unknown;
}

interface SettingsApiSnapshot {
  mutable: boolean;
  host_config: SettingsSnapshot["host_config"];
  default_execution_route: string;
  runtime_profile: string;
  yolo_enabled: boolean;
  loop: SettingsSnapshot["loop"];
}

export class HarnessApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail?: unknown) {
    super(message);
    this.name = "HarnessApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export class FetchHarnessClient implements HarnessClient {
  private session: string | null = null;
  private readonly baseUrl: string;
  private readonly fetchImplementation: FetchImplementation;

  constructor(
    baseUrl = "/api",
    fetchImplementation: FetchImplementation = globalThis.fetch.bind(globalThis),
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.fetchImplementation = fetchImplementation;
  }

  async getHealth() {
    return this.request<HealthSnapshot>("/health");
  }

  async listWorkspaces() {
    const payload = await this.request<{ workspaces: Workspace[] }>("/workspaces");
    return payload.workspaces;
  }

  async listConversations(includeArchived = false) {
    const query = includeArchived ? "?include_archived=true" : "";
    const payload = await this.request<{ conversations: Conversation[] }>(
      `/conversations${query}`,
    );
    return payload.conversations;
  }

  async getConversation(conversationId: string) {
    const payload = await this.request<{ conversation: Conversation }>(
      `/conversations/${encodeURIComponent(conversationId)}`,
    );
    return payload.conversation;
  }

  async createConversation(workspaceRoot: string, name = "New conversation") {
    const payload = await this.request<{ conversation: Conversation }>(
      "/conversations",
      this.jsonRequest("POST", { workspace_root: workspaceRoot, name }),
    );
    return payload.conversation;
  }

  selectConversation(conversationId: string) {
    return this.getChatSnapshot(conversationId);
  }

  async renameConversation(conversationId: string, name: string) {
    return this.updateConversation(conversationId, { name });
  }

  async archiveConversation(conversationId: string, archived = true) {
    return this.updateConversation(conversationId, { archived });
  }

  async deleteConversation(conversationId: string) {
    await this.request<void>(`/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    });
  }

  async getChatSnapshot(conversationId?: string | null): Promise<ChatSnapshot> {
    const [workspaces, conversations, settings, corpora] = await Promise.all([
      this.listWorkspaces(),
      this.listConversations(),
      this.request<SettingsApiSnapshot>("/ui/settings"),
      this.getCorporaSnapshot(),
    ]);
    const selected = conversationId
      ? conversations.find((conversation) => conversation.id === conversationId) ??
        (await this.getConversation(conversationId))
      : conversations[0];
    const groups = groupConversations(workspaces, conversations, selected?.id ?? null);

    if (!selected) {
      return {
        conversationId: null,
        conversationTitle: null,
        workspaceName: null,
        workspaces: groups,
        grants: [],
        messages: [],
        pendingRequests: [],
        activeTurn: null,
        turns: [],
        feedback: [],
        pendingConfirmation: null,
        confirmationWaivers: [],
        yolo: settings.yolo_enabled ?? false,
        corpusId: null,
        corpora: corpora.corpora,
        execution: toExecutionSnapshot(settings),
      };
    }

    const encodedId = encodeURIComponent(selected.id);
    const [snapshot, grants] = await Promise.all([
      this.request<ApiChatSnapshot>(`/ui/chat?conversation_id=${encodedId}`),
      this.listGrants(selected.id),
    ]);
    const workspace = workspaces.find(
      (item) => item.id === snapshot.conversation.workspace_id,
    );

    return {
      conversationId: snapshot.conversation.id,
      conversationTitle: snapshot.conversation.name,
      workspaceName: workspace ? workspaceName(workspace.root) : null,
      workspaces: groupConversations(
        workspaces,
        conversations,
        snapshot.conversation.id,
      ),
      grants,
      messages: projectTimeline(snapshot.history, snapshot.turns),
      pendingRequests: snapshot.pending_requests.map((request, index) => ({
        id: request.id,
        prompt: request.content,
        createdAt: request.created_at,
        sequence: request.sequence,
        status: request.status,
        mode: index === 0 && snapshot.active_turn ? "waiting" : "queued",
      })),
      activeTurn: snapshot.active_turn,
      turns: snapshot.turns,
      feedback: snapshot.feedback,
      pendingConfirmation: snapshot.pending_confirmation,
      confirmationWaivers: snapshot.confirmation_waivers ?? [],
      yolo: snapshot.yolo ?? false,
      corpusId: snapshot.corpus_id ?? null,
      corpora: corpora.corpora,
      execution: toExecutionSnapshot(settings),
    };
  }

  async getCorporaSnapshot() {
    return this.request<CorporaSnapshot>("/ui/corpora");
  }

  async createCorpus(name: string, description = "") {
    const payload = await this.request<{ corpus: Corpus }>(
      "/corpora",
      this.jsonRequest("POST", { name, description }),
    );
    return payload.corpus;
  }

  async renameCorpus(corpusId: string, changes: { name?: string; description?: string }) {
    const payload = await this.request<{ corpus: Corpus }>(
      `/corpora/${encodeURIComponent(corpusId)}`,
      this.jsonRequest("PATCH", changes),
    );
    return payload.corpus;
  }

  async deleteCorpus(corpusId: string) {
    await this.request<void>(`/corpora/${encodeURIComponent(corpusId)}`, {
      method: "DELETE",
    });
  }

  async listCorpusDocuments(corpusId: string) {
    const payload = await this.request<{ documents: CorpusDocument[] }>(
      `/corpora/${encodeURIComponent(corpusId)}/documents`,
    );
    return payload.documents;
  }

  async deleteCorpusDocument(corpusId: string, documentId: string) {
    await this.request<void>(
      `/corpora/${encodeURIComponent(corpusId)}/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
    );
  }

  async uploadCorpusDocument(corpusId: string, file: File) {
    // multipart e não JSON: um PDF em base64 cresce um terço e ainda teria que
    // passar inteiro pelo parser de JSON antes de virar bytes de novo.
    const body = new FormData();
    body.append("file", file, file.name);
    const payload = await this.request<{ job: IngestionJob }>(
      `/corpora/${encodeURIComponent(corpusId)}/documents`,
      { method: "POST", body },
    );
    return payload.job;
  }

  async attachImage(conversationId: string, file: File) {
    const body = new FormData();
    body.append("file", file, file.name);
    const payload = await this.request<{ path: string }>(
      `/conversations/${encodeURIComponent(conversationId)}/attachments`,
      { method: "POST", body },
    );
    return payload.path;
  }

  async startCorpusScrape(corpusId: string, seed: string) {
    const payload = await this.request<{ job: IngestionJob }>(
      `/corpora/${encodeURIComponent(corpusId)}/jobs`,
      this.jsonRequest("POST", { seed }),
    );
    return payload.job;
  }

  async listCorpusJobs(corpusId: string) {
    const payload = await this.request<{ jobs: IngestionJob[] }>(
      `/corpora/${encodeURIComponent(corpusId)}/jobs`,
    );
    return payload.jobs;
  }

  async cancelCorpusJob(corpusId: string, jobId: string) {
    const payload = await this.request<{ job: IngestionJob }>(
      `/corpora/${encodeURIComponent(corpusId)}/jobs/${encodeURIComponent(jobId)}`,
      { method: "DELETE" },
    );
    return payload.job;
  }

  async selectCorpus(conversationId: string, corpusId: string | null) {
    await this.request<void>(
      `/conversations/${encodeURIComponent(conversationId)}/corpus`,
      this.jsonRequest("PUT", { corpus_id: corpusId }),
    );
  }

  async enqueueRequest(conversationId: string, content: string) {
    const payload = await this.request<{ request: ApiPendingRequest }>(
      `/conversations/${encodeURIComponent(conversationId)}/requests`,
      this.jsonRequest("POST", { content }),
    );
    return payload.request;
  }

  async streamAgent(
    input: RunAgentInput,
    onEvent: (event: AgUiEvent) => void,
    signal?: AbortSignal,
  ): Promise<AgUiTerminalEvent> {
    const request = this.jsonRequest("POST", input);
    const response = await this.fetchImplementation(`${this.baseUrl}/agent`, {
      ...request,
      headers: {
        ...request.headers,
        Accept: "text/event-stream",
      },
      signal,
    });
    if (!response.ok) await this.throwResponseError(response);
    if (!response.body) {
      throw new HarnessApiError(
        response.status,
        "stream_body_missing",
        "A resposta SSE não contém um stream.",
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let terminal: AgUiTerminalEvent | null = null;

    const consumeFrames = (flush: boolean) => {
      while (true) {
        const match = /\r?\n\r?\n/.exec(buffer);
        if (!match) break;
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const event = parseSseFrame(frame);
        if (!event) continue;
        onEvent(event);
        if (event.type === "RUN_FINISHED" || event.type === "RUN_ERROR") {
          terminal = event;
        }
      }
      if (flush && buffer.trim()) {
        const event = parseSseFrame(buffer);
        buffer = "";
        if (event) {
          onEvent(event);
          if (event.type === "RUN_FINISHED" || event.type === "RUN_ERROR") {
            terminal = event;
          }
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      consumeFrames(false);
    }
    buffer += decoder.decode();
    consumeFrames(true);

    if (!terminal) {
      throw new HarnessApiError(
        response.status,
        "stream_ended_without_terminal_event",
        "O stream terminou sem RUN_FINISHED ou RUN_ERROR.",
      );
    }
    return terminal;
  }

  async editPendingRequest(
    conversationId: string,
    requestId: string,
    content: string,
  ) {
    const payload = await this.request<{ request: ApiPendingRequest }>(
      `/conversations/${encodeURIComponent(conversationId)}/requests/${encodeURIComponent(requestId)}`,
      this.jsonRequest("PATCH", { content }),
    );
    return payload.request;
  }

  async cancelPendingRequest(conversationId: string, requestId: string) {
    const payload = await this.request<{ request: ApiPendingRequest }>(
      `/conversations/${encodeURIComponent(conversationId)}/requests/${encodeURIComponent(requestId)}`,
      { method: "DELETE" },
    );
    return payload.request;
  }

  async listGrants(conversationId: string) {
    const payload = await this.request<{ grants: Grant[] }>(
      `/conversations/${encodeURIComponent(conversationId)}/grants`,
    );
    return payload.grants;
  }

  async addGrant(conversationId: string, permission: string) {
    const payload = await this.request<{ grant: Grant }>(
      `/conversations/${encodeURIComponent(conversationId)}/grants`,
      this.jsonRequest("POST", { permission }),
    );
    return payload.grant;
  }

  async revokeGrant(conversationId: string, grantId: string) {
    await this.request<void>(
      `/conversations/${encodeURIComponent(conversationId)}/grants/${encodeURIComponent(grantId)}`,
      { method: "DELETE" },
    );
  }

  async getPendingConfirmation(conversationId: string) {
    const payload = await this.request<{ confirmation: PendingConfirmation | null }>(
      `/conversations/${encodeURIComponent(conversationId)}/confirmation`,
    );
    return payload.confirmation;
  }

  async resolveConfirmation(
    conversationId: string,
    confirmationId: string,
    approved: boolean,
    waive = false,
  ) {
    await this.request(
      `/conversations/${encodeURIComponent(conversationId)}/confirmation/${encodeURIComponent(
        confirmationId,
      )}`,
      this.jsonRequest("POST", { approved, waive }),
    );
  }

  async revokeConfirmationWaiver(conversationId: string, effect: string) {
    await this.request(
      `/conversations/${encodeURIComponent(conversationId)}/confirmation-waivers/${encodeURIComponent(
        effect,
      )}`,
      { method: "DELETE" },
    );
  }

  async updateHostConfig(submission: SetupSubmission) {
    return this.request<{ restart_required: boolean }>(
      "/admin/host-config",
      this.jsonRequest("PUT", submission),
    );
  }

  async setYoloEnabled(enabled: boolean) {
    await this.request("/admin/yolo", this.jsonRequest("PUT", { enabled }));
  }

  async setConversationYolo(conversationId: string, disabled: boolean) {
    await this.request(
      `/conversations/${encodeURIComponent(conversationId)}`,
      this.jsonRequest("PATCH", { yolo_disabled: disabled }),
    );
  }

  async stop(conversationId: string) {
    await this.request(`/conversations/${encodeURIComponent(conversationId)}/stop`, {
      method: "POST",
    });
  }

  async addFeedback(
    conversationId: string,
    rating: number,
    turnId?: string,
    comment?: string,
  ) {
    const payload = await this.request<{ feedback: FeedbackRecord }>(
      `/conversations/${encodeURIComponent(conversationId)}/feedback`,
      this.jsonRequest("POST", {
        rating,
        comment: comment ?? null,
        turn_id: turnId ?? null,
      }),
    );
    return payload.feedback;
  }

  async getEvalsSnapshot() {
    const payload = await this.request<Omit<EvalsSnapshot, "selectedRunId">>(
      "/ui/evals",
    );
    return {
      ...payload,
      selectedRunId: payload.runs[0]?.id ?? null,
    };
  }

  async listEvalExperiments() {
    const payload = await this.request<{ experiments: Experiment[] }>(
      "/evals/experiments",
    );
    return payload.experiments;
  }

  async listEvalRuns() {
    const payload = await this.request<{ runs: EvalRun[] }>("/evals/runs");
    return payload.runs;
  }

  getEvalRun(runId: string) {
    return this.request<Record<string, unknown>>(`/evals/runs/${encodeURIComponent(runId)}`);
  }

  async createEvalRun(input: {
    experimentId: string;
    tier: "contract" | "model_smoke" | "experiment";
    phase: "pilot" | "promotion";
    seeds?: number[];
  }) {
    const payload = await this.request<{ run: EvalRun }>(
      "/evals/runs",
      this.jsonRequest("POST", {
        experiment_id: input.experimentId,
        tier: input.tier,
        phase: input.phase,
        ...(input.seeds ? { seeds: input.seeds } : {}),
      }),
    );
    return payload.run;
  }

  async startEvalRun(runId: string) {
    const payload = await this.request<{ run: EvalRun }>(
      `/evals/runs/${encodeURIComponent(runId)}/start`,
      { method: "POST" },
    );
    return payload.run;
  }

  async cancelEvalRun(runId: string) {
    const payload = await this.request<{ run: EvalRun }>(
      `/evals/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    );
    return payload.run;
  }

  async listEvalReports() {
    const payload = await this.request<{ reports: EvalReport[] }>("/evals/reports");
    return payload.reports;
  }

  async getEvalReport(runId: string) {
    const payload = await this.request<{ report: EvalReport }>(
      `/evals/reports/${encodeURIComponent(runId)}`,
    );
    return payload.report;
  }

  async listRegressionDrafts() {
    const payload = await this.request<{ drafts: RegressionDraft[] }>(
      "/evals/regression-drafts",
    );
    return payload.drafts;
  }

  async createRegressionDraft(conversationId: string, feedbackId: string) {
    const payload = await this.request<{ draft: RegressionDraft }>(
      "/evals/regression-drafts",
      this.jsonRequest("POST", {
        conversation_id: conversationId,
        feedback_id: feedbackId,
      }),
    );
    return payload.draft;
  }

  async getSettingsSnapshot() {
    const [health, workspaces, settings] = await Promise.all([
      this.getHealth(),
      this.listWorkspaces(),
      this.request<SettingsApiSnapshot>("/ui/settings"),
    ]);
    return { health, workspaces, ...settings };
  }

  private async updateConversation(
    conversationId: string,
    update: { name?: string; archived?: boolean },
  ) {
    const payload = await this.request<{ conversation: Conversation }>(
      `/conversations/${encodeURIComponent(conversationId)}`,
      this.jsonRequest("PATCH", update),
    );
    return payload.conversation;
  }

  async getSessionStatus() {
    return this.request<SessionStatus>("/session");
  }

  async login(password: string) {
    const payload = await this.request<{ session: string; expires_at: string }>(
      "/session",
      this.jsonRequest("POST", { password }),
    );
    // The token lives in memory only: a reload asks for the password again, and
    // nothing writes it to storage where a script could read it.
    this.session = payload.session;
    return {
      authentication_required: true,
      authenticated: true,
      expires_at: payload.expires_at,
    };
  }

  async logout() {
    try {
      await this.request("/session", { method: "DELETE" });
    } finally {
      this.session = null;
    }
  }

  async getSetupStatus() {
    return this.request<SetupStatus>("/setup/status");
  }

  async completeSetup(token: string, submission: SetupSubmission) {
    // The setup token is ephemeral and travels in its own header, never in the body.
    await this.request("/setup", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Harness-Setup-Token": token,
      },
      body: JSON.stringify(submission),
    });
  }

  private withSession(init?: RequestInit): RequestInit | undefined {
    if (this.session === null) return init;
    return {
      ...init,
      headers: { ...(init?.headers ?? {}), "X-Harness-Session": this.session },
    };
  }

  private jsonRequest(method: string, body: unknown): RequestInit {
    return {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImplementation(
      `${this.baseUrl}${path}`,
      this.withSession(init),
    );
    if (!response.ok) await this.throwResponseError(response);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  private async throwResponseError(response: Response): Promise<never> {
    let payload: ApiErrorBody = {};
    try {
      payload = (await response.json()) as ApiErrorBody;
    } catch {
      // A non-JSON proxy response still remains an explicit live-client error.
    }
    const code = payload.error?.code ?? `http_${response.status}`;
    const detail = payload.detail;
    const message =
      payload.error?.message ??
      (typeof detail === "string" ? detail : response.statusText || code);
    throw new HarnessApiError(response.status, code, message, detail);
  }
}

export function parseSseFrame(frame: string): AgUiEvent | null {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch (error) {
    throw new HarnessApiError(200, "invalid_sse_json", "Evento SSE inválido.", error);
  }
  if (!isAgUiEvent(parsed)) {
    throw new HarnessApiError(200, "invalid_ag_ui_event", "Evento AG-UI inválido.", parsed);
  }
  return parsed;
}

export function projectTimeline(
  history: CanonicalHistoryEntry[],
  turns: Turn[],
): ChatMessage[] {
  const turnById = new Map(turns.map((turn) => [turn.id, turn]));
  const grouped = new Map<string, CanonicalHistoryEntry[]>();
  for (const entry of [...history].sort((left, right) => left.sequence - right.sequence)) {
    const entries = grouped.get(entry.turn_id) ?? [];
    entries.push(entry);
    grouped.set(entry.turn_id, entries);
  }

  const messages: ChatMessage[] = [];
  for (const [turnId, entries] of grouped) {
    const turn = turnById.get(turnId);
    const tools = new Map<string, ToolCall>();
    const events: NonNullable<ChatMessage["events"]> = [];
    let retrieval: Retrieval | undefined;
    let finalResponse: CanonicalHistoryEntry | undefined;
    let metrics: ChatMessage["metrics"];

    for (const entry of entries) {
      if (entry.kind === "user_message") {
        const content = stringValue(entry.payload.content);
        if (content !== null) {
          messages.push({
            id: entry.id,
            turnId,
            role: "user",
            content,
            createdAt: entry.created_at,
          });
        }
      }
      if (entry.kind === "model_attempt") {
        for (const call of arrayValue(entry.payload.tool_calls)) {
          if (!isRecord(call)) continue;
          const id = stringValue(call.id);
          if (!id) continue;
          const name = stringValue(call.name) ?? "";
          tools.set(id, {
            id,
            name,
            label: name,
            status: "running",
            arguments: isRecord(call.arguments) ? call.arguments : {},
            summary: "",
          });
        }
        metrics ??= metricsValue(entry.payload.metrics);
      }
      if (entry.kind === "tool_result") {
        const id = stringValue(entry.payload.tool_call_id);
        if (!id) continue;
        const status = stringValue(entry.payload.status) ?? "";
        const existing = tools.get(id);
        const error = entry.payload.error;
        const data = entry.payload.data;
        const meta = entry.payload.meta;
        const name = stringValue(entry.payload.tool_name) ?? existing?.name ?? "";
        tools.set(id, {
          id,
          name,
          label: name,
          status: status === "success" || status === "empty" ? "success" : "error",
          arguments: existing?.arguments ?? {},
          summary: status,
          ...(error !== null && error !== undefined
            ? { error: displayJson(error) }
            : {}),
          ...(data !== null && data !== undefined ? { result: displayJson(data) } : {}),
          ...(isRecord(data) && typeof data.diff === "string" ? { diff: data.diff } : {}),
          ...(isRecord(meta) && typeof meta.duration_ms === "number"
            ? { durationMs: meta.duration_ms }
            : {}),
        });
      }
      if (entry.kind === "final_response") {
        finalResponse = entry;
        metrics ??= metricsValue(entry.payload.metrics);
      }
      if (
        entry.kind === "internal_automation" &&
        entry.payload.automation_id === "corpus_retrieval"
      ) {
        // A recuperação tem card próprio: despejar o JSON dela junto das outras
        // automações esconderia as fontes atrás de um <pre>.
        retrieval = toRetrieval(entry.payload);
        continue;
      }
      if (
        entry.kind === "rejected_model_attempt" ||
        entry.kind === "internal_automation"
      ) {
        events.push({ id: entry.id, kind: entry.kind, payload: entry.payload });
      }
    }

    if (finalResponse || tools.size || events.length || retrieval) {
      messages.push({
        id: finalResponse?.id ?? `${turnId}-activity`,
        turnId,
        role: "assistant",
        content: finalResponse ? stringValue(finalResponse.payload.content) ?? "" : "",
        createdAt:
          finalResponse?.created_at ?? entries.at(-1)?.created_at ?? turn?.started_at ?? "",
        ...(tools.size ? { tools: [...tools.values()] } : {}),
        ...(retrieval ? { retrieval } : {}),
        ...(events.length ? { events } : {}),
        ...(metrics ? { metrics } : {}),
        ...(turn?.terminal_outcome ? { terminalOutcome: turn.terminal_outcome } : {}),
      });
    }
  }
  return messages;
}

function toRetrieval(payload: Record<string, JsonValue>): Retrieval {
  const passages = arrayValue(payload.passages).flatMap((item) => {
    if (!isRecord(item)) return [];
    const marker = typeof item.marker === "number" ? item.marker : 0;
    const text = stringValue(item.text);
    if (!marker || text === null) return [];
    return [
      {
        marker,
        document: stringValue(item.document) ?? "",
        location: stringValue(item.location) ?? "",
        origin: item.origin === "scrape" ? ("scrape" as const) : ("upload" as const),
        source: stringValue(item.source) ?? "",
        untrusted: item.untrusted === true,
        text,
      },
    ];
  });
  return {
    corpus: stringValue(payload.corpus) ?? "",
    status: stringValue(payload.status) ?? "completed",
    searchQuery: stringValue(payload.search_query),
    passages,
    detail: stringValue(payload.detail),
  };
}

function groupConversations(
  workspaces: Workspace[],
  conversations: Conversation[],
  selectedId: string | null,
): WorkspaceGroup[] {
  return workspaces.map((workspace) => ({
    id: workspace.id,
    root: workspace.root,
    name: workspaceName(workspace.root),
    conversations: conversations
      .filter((conversation) => conversation.workspace_id === workspace.id)
      .map((conversation) => ({
        id: conversation.id,
        title: conversation.name,
        updatedAt: conversation.last_active_at,
        active: conversation.id === selectedId,
        archived: conversation.archived_at !== null,
      })),
  }));
}

function workspaceName(root: string) {
  const segments = root.replace(/[\\/]+$/, "").split(/[\\/]/);
  return segments.at(-1) || root;
}

function toExecutionSnapshot(settings: SettingsApiSnapshot) {
  return {
    defaultExecutionRoute: settings.default_execution_route,
    runtimeProfile: settings.runtime_profile,
    loop: settings.loop,
  };
}

function isAgUiEvent(value: unknown): value is AgUiEvent {
  return (
    isRecord(value) &&
    typeof value.type === "string" &&
    AG_UI_EVENT_TYPES.has(value.type)
  );
}

const AG_UI_EVENT_TYPES = new Set([
  "RUN_STARTED",
  "RUN_FINISHED",
  "RUN_ERROR",
  "STEP_STARTED",
  "STEP_FINISHED",
  "REASONING_START",
  "REASONING_END",
  "REASONING_MESSAGE_START",
  "REASONING_MESSAGE_CONTENT",
  "REASONING_MESSAGE_END",
  "TOOL_CALL_START",
  "TOOL_CALL_ARGS",
  "TOOL_CALL_END",
  "TOOL_CALL_RESULT",
  "TEXT_MESSAGE_START",
  "TEXT_MESSAGE_CONTENT",
  "TEXT_MESSAGE_END",
  "CUSTOM",
]);

function isRecord(value: unknown): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: JsonValue | undefined): string | null {
  return typeof value === "string" ? value : null;
}

function arrayValue(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function metricsValue(value: JsonValue | undefined) {
  if (!isRecord(value)) return undefined;
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string | number] =>
      typeof entry[1] === "string" || typeof entry[1] === "number",
  );
  return entries.length ? Object.fromEntries(entries) : undefined;
}

function displayJson(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
