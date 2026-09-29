export type AppArea = "chat" | "corpus" | "evals" | "settings";
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface HealthSnapshot {
  ready: boolean;
  reason_code: string | null;
  capabilities: Record<string, boolean>;
}

export interface Workspace {
  id: string;
  root: string;
}

export interface Conversation {
  id: string;
  workspace_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  last_active_at: string;
  archived_at: string | null;
}

export interface ApiPendingRequest {
  id: string;
  conversation_id: string;
  sequence: number;
  content: string;
  status: "queued" | "canceled" | "dequeued";
  created_at: string;
  updated_at: string;
}

export interface CanonicalHistoryEntry {
  id: string;
  sequence: number;
  conversation_id: string;
  turn_id: string;
  kind:
    | "user_message"
    | "model_attempt"
    | "rejected_model_attempt"
    | "tool_result"
    | "final_response"
    | "internal_automation";
  payload: Record<string, JsonValue>;
  created_at: string;
}

export interface TerminalOutcome {
  kind:
    | "completed"
    | "limit_reached"
    | "cancelled"
    | "blocked"
    | "failed"
    | "abandoned";
  reason_code: string;
  recorded_at: string;
  detail: string | null;
}

export interface Turn {
  id: string;
  conversation_id: string;
  request_id: string;
  status: "active" | "finished";
  started_at: string;
  ended_at: string | null;
  terminal_outcome: TerminalOutcome | null;
}

export interface Grant {
  id: string;
  conversation_id: string;
  permission: string;
  scope: string;
  granted_at: string;
  expires_at: string | null;
}

export interface FeedbackRecord {
  id: string;
  conversation_id: string;
  turn_id: string | null;
  rating: number;
  comment: string | null;
  created_at: string;
}

export interface ConfirmationPreview {
  tool_call_id: string;
  path: string;
  kind: "create" | "replace" | "edit";
  diff: string;
  truncated: boolean;
}

export interface PendingConfirmation {
  id: string;
  conversation_id: string;
  turn_id: string;
  step_sequence: number;
  reason_code: string;
  tool_calls: Array<{
    id: string;
    name: string;
    arguments: Record<string, JsonValue>;
    // Efeitos declarados no registry: é o que diz se esta decisão é sobre
    // escrever no Workspace ou sobre sair de novo para a rede.
    effects?: string[];
  }>;
  previews: ConfirmationPreview[];
}

export interface ApiChatSnapshot {
  conversation: Conversation;
  pending_requests: ApiPendingRequest[];
  history: CanonicalHistoryEntry[];
  active_turn: Turn | null;
  turns: Turn[];
  feedback: FeedbackRecord[];
  pending_confirmation: PendingConfirmation | null;
  confirmation_waivers: string[];
  yolo: boolean;
  corpus_id: string | null;
}

export interface WorkspaceGroup {
  id: string;
  name: string;
  root: string;
  conversations: ConversationSummary[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: string;
  active?: boolean;
  archived?: boolean;
}

export interface Corpus {
  id: string;
  name: string;
  description: string;
  embedding_model: string;
  embedding_dimensions: number;
  document_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface CorpusDocument {
  id: string;
  origin_kind: "upload" | "scrape";
  origin_ref: string;
  title: string;
  source_digest: string;
  taints: string[];
  chunk_count: number;
  ingested_at: string;
}

export type IngestionJobStatus =
  | "queued"
  | "running"
  | "canceling"
  | "canceled"
  | "completed"
  | "failed";

export interface IngestionJob {
  id: string;
  corpus_id: string;
  kind: "upload" | "scrape" | "ocr";
  origin: string;
  status: IngestionJobStatus;
  seen: number;
  indexed: number;
  skipped: number;
  chunks: number;
  current: string | null;
  reason_code: string | null;
  detail: string | null;
}

export interface CorporaSnapshot {
  available: boolean;
  embedding_model: string | null;
  accepted_extensions: string[];
  corpora: Corpus[];
  jobs: IngestionJob[];
}

export interface RetrievedPassage {
  marker: number;
  document: string;
  location: string;
  origin: "upload" | "scrape";
  source: string;
  untrusted: boolean;
  text: string;
}

export interface Retrieval {
  corpus: string;
  status: string;
  searchQuery: string | null;
  passages: RetrievedPassage[];
  detail: string | null;
}

export type ToolStatus = "success" | "error" | "running";

export interface ToolCall {
  id: string;
  name: string;
  label: string;
  status: ToolStatus;
  durationMs?: number;
  arguments: Record<string, unknown>;
  summary: string;
  error?: string;
  result?: string;
  diff?: string;
}

export interface ReasoningTrace {
  summary: string;
  content: string;
  transient: true;
}

export interface TimelineEvent {
  id: string;
  kind: "rejected_model_attempt" | "internal_automation";
  payload: Record<string, JsonValue>;
}

/**
 * O que a janela de contexto custou no passo mais recente do turno.
 *
 * Chega por `CUSTOM`/`harness.context_usage`, um evento por passo, e é
 * transitório como o raciocínio: descreve a execução ao vivo, não o histórico.
 */
export interface ContextUsage {
  inputTokens: number;
  contextWindow: number;
  outputBudget: number;
  droppedTurns: number;
}

/**
 * Quão rápido o modelo gerou: o passo mais recente e a soma do turno.
 *
 * Chega por `CUSTOM`/`harness.generation_stats`, um evento por passo, com
 * contagens e tempos medidos pelo runtime — nunca estimados na UI.
 */
export interface GenerationSpeed {
  lastOutputTokens: number;
  lastEvalMs: number;
  lastPromptTokens: number;
  lastPromptEvalMs?: number;
  turnOutputTokens: number;
  turnEvalMs: number;
  steps: number;
}

export interface ChatMessage {
  id: string;
  turnId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  reasoning?: ReasoningTrace;
  tools?: ToolCall[];
  retrieval?: Retrieval;
  contextUsage?: ContextUsage;
  generationSpeed?: GenerationSpeed;
  events?: TimelineEvent[];
  metrics?: Record<string, string | number>;
  terminalOutcome?: TerminalOutcome;
  liveOutcome?: { kind: string; reasonCode: string };
  live?: boolean;
}

export interface PendingRequest {
  id: string;
  prompt: string;
  createdAt: string;
  sequence?: number;
  status?: ApiPendingRequest["status"];
  files?: string[];
  mode: "queued" | "waiting";
}

export interface ExecutionSnapshot {
  defaultExecutionRoute: string;
  runtimeProfile: string;
  loop: Record<string, JsonValue>;
}

export interface ChatSnapshot {
  conversationId: string | null;
  conversationTitle: string | null;
  workspaceName: string | null;
  workspaces: WorkspaceGroup[];
  grants: Grant[];
  messages: ChatMessage[];
  pendingRequests: PendingRequest[];
  activeTurn: Turn | null;
  turns: Turn[];
  feedback: FeedbackRecord[];
  pendingConfirmation: PendingConfirmation | null;
  confirmationWaivers: string[];
  yolo: boolean;
  corpusId: string | null;
  corpora: Corpus[];
  execution?: ExecutionSnapshot;
}

export type EvalTier = "contract" | "model_smoke" | "experiment";
export type EvalPhase = "pilot" | "promotion";
export type EvalRunStatus =
  | "created"
  | "queued"
  | "running"
  | "completed"
  | "canceling"
  | "canceled"
  | "blocked"
  | "failed";

export interface Experiment {
  id: string;
  status: string;
  runtime_profile: string;
  execution_route: string;
  fixture_tags: string[];
  arms: Array<{ id: string; [key: string]: JsonValue }>;
  [key: string]: JsonValue;
}

export interface EvalRun {
  id: string;
  experiment_id: string;
  tier: EvalTier;
  phase: EvalPhase;
  status: EvalRunStatus;
  seeds: number[];
  reason_code: string | null;
  created_at: string;
  updated_at: string;
}

export interface EvalReport {
  id: string;
  run_id: string;
  payload: Record<string, JsonValue>;
  created_at: string;
}

export interface RegressionDraft {
  id: string;
  source_feedback_sha256: string;
  source_turn_sha256: string;
  rating: number;
  comment_present: boolean;
  terminal_outcome_kind: TerminalOutcome["kind"];
  terminal_outcome_reason: string;
  created_at: string;
}

export interface EvalsSnapshot {
  experiments: Experiment[];
  runs: EvalRun[];
  reports: EvalReport[];
  capabilities: {
    eval_runner: boolean;
    progress_transport: string;
  };
  selectedRunId: string | null;
}

export interface LoopSettings {
  max_steps: number;
  max_tool_calls_per_step: number;
  max_tool_calls_per_turn: number;
  max_turn_duration_seconds: number;
  max_output_tokens: number;
  offer_tools_on_final_step: boolean;
  [key: string]: JsonValue;
}

export interface SettingsSnapshot {
  health: HealthSnapshot;
  workspaces: Workspace[];
  mutable: boolean;
  host_config: HostConfigSnapshot | null;
  default_execution_route: string;
  runtime_profile: string;
  yolo_enabled: boolean;
  loop: LoopSettings;
}

export interface SessionStatus {
  authentication_required: boolean;
  authenticated: boolean;
  expires_at: string | null;
}

export interface SetupStatus {
  configured: boolean;
  required: boolean;
  restart_required: boolean;
  token_expires_at: string | null;
  suggested_state_dir: string;
  suggested_tokenizer_path: string;
}

export interface SetupSubmission {
  allowed_workspace_roots: string[];
  tokenizer_path: string;
  state_dir: string;
  allowed_origins: string[];
  searxng_url?: string | null;
  ollama_url?: string;
  browser_executable?: string | null;
}

export interface HostConfigSnapshot {
  allowed_workspace_roots: string[];
  tokenizer_path: string;
  tokenizer_digest: string;
  state_dir: string;
  allowed_origins: string[];
  searxng_url: string | null;
  ollama_url: string;
  browser_executable: string | null;
}

export interface RunAgentMessage {
  id: string;
  role: "developer" | "system" | "assistant" | "user" | "tool";
  content: string | null;
  toolCallId?: string;
}

export interface RunAgentInput {
  threadId: string;
  runId?: string;
  messages: RunAgentMessage[];
  state?: JsonValue;
  context?: JsonValue;
  tools?: JsonValue;
}

export type AgUiEvent =
  | { type: "RUN_STARTED"; threadId: string; runId: string }
  | { type: "RUN_FINISHED"; threadId: string; runId: string }
  | { type: "RUN_ERROR"; message: string; code?: string }
  | { type: "STEP_STARTED" | "STEP_FINISHED"; stepName: string | null }
  | { type: "REASONING_START" | "REASONING_END" }
  | {
      type: "REASONING_MESSAGE_START";
      messageId: string;
      role: "reasoning";
    }
  | {
      type: "REASONING_MESSAGE_CONTENT";
      messageId: string;
      delta: string;
    }
  | { type: "REASONING_MESSAGE_END"; messageId: string }
  | {
      type: "TOOL_CALL_START";
      toolCallId: string;
      toolCallName: string;
    }
  | { type: "TOOL_CALL_ARGS"; toolCallId: string; delta: string }
  | { type: "TOOL_CALL_END"; toolCallId: string }
  | {
      type: "TOOL_CALL_RESULT";
      messageId: string;
      toolCallId: string;
      role: "tool";
      content: string;
    }
  | {
      type: "TEXT_MESSAGE_START";
      messageId: string;
      role: "assistant";
    }
  | {
      type: "TEXT_MESSAGE_CONTENT";
      messageId: string;
      delta: string;
    }
  | { type: "TEXT_MESSAGE_END"; messageId: string }
  | { type: "CUSTOM"; name: string; value: JsonValue };

export type AgUiTerminalEvent = Extract<
  AgUiEvent,
  { type: "RUN_FINISHED" | "RUN_ERROR" }
>;
