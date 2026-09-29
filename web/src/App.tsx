import {
  Activity,
  Archive,
  ArrowUp,
  Bot,
  Brain,
  Check,
  ChevronDown,
  CircleStop,
  FileCode2,
  FlaskConical,
  Globe2,
  HardDrive,
  ImagePlus,
  Library,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Search,
  Settings as SettingsIcon,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  WifiOff,
  X,
  Zap,
  ZapOff,
} from "lucide-react";
import {
  Children,
  useEffect,
  useRef,
  useState,
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { harnessClient, type HarnessClient } from "./client";
import { ConfirmationDialog } from "./components/ConfirmationDialog";
import { PendingQueue } from "./components/PendingQueue";
import { ContextGauge } from "./components/ContextGauge";
import { RetrievalCard, passageAnchor } from "./components/RetrievalCard";
import { ToolCallCard } from "./components/ToolCallCard";
import type {
  AgUiEvent,
  AppArea,
  ChatMessage,
  ChatSnapshot,
  ContextUsage,
  CorporaSnapshot,
  CorpusDocument,
  IngestionJob,
  EvalPhase,
  EvalReport,
  EvalRun,
  EvalTier,
  EvalsSnapshot,
  FeedbackRecord,
  Grant,
  JsonValue,
  SessionStatus,
  SettingsSnapshot,
  SetupStatus,
  ToolCall,
  WorkspaceGroup,
} from "./types";

interface AppProps {
  client?: HarnessClient;
}

interface AppData {
  chat: ChatSnapshot;
  evals: EvalsSnapshot;
  settings: SettingsSnapshot;
}

interface LiveRun {
  runId: string;
  // A conversa dona do run: sem ela o stream de uma conversa era desenhado na
  // que estivesse aberta.
  conversationId: string;
  status: "queued" | "running" | "finished" | "error";
  reasoning: string;
  content: string;
  tools: ToolCall[];
  toolArguments: Record<string, string>;
  steps: number;
  contextUsage?: ContextUsage;
  outcome?: { kind: string; reasonCode: string };
  error?: string;
}

// Uma origin de loopback tem duas grafias, e o navegador manda exatamente a que
// o Operator digitou na barra: quem autoriza só uma vê a outra virar 403 nos
// próprios assets, ou seja, tela branca sem erro visível. O fallback anterior ao
// setup já autoriza o par; os formulários passam a sugerir o mesmo, e a lista
// continua editável antes de salvar.
const LOOPBACK_COUNTERPART: Record<string, string> = {
  "127.0.0.1": "localhost",
  localhost: "127.0.0.1",
};

export function withLoopbackPair(origins: readonly string[]): string[] {
  const result = [...origins];
  for (const origin of origins) {
    let url: URL;
    try {
      url = new URL(origin);
    } catch {
      continue;
    }
    const counterpart = LOOPBACK_COUNTERPART[url.hostname];
    if (!counterpart) continue;
    url.hostname = counterpart;
    if (!result.includes(url.origin)) result.push(url.origin);
  }
  return result;
}

const areaItems = [
  { id: "chat" as const, label: "Chat", icon: MessageSquare },
  { id: "corpus" as const, label: "RAG", icon: Library },
  { id: "evals" as const, label: "Avaliações", icon: FlaskConical },
  { id: "settings" as const, label: "Configurações", icon: SettingsIcon },
];

const markdownComponents: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  table: ({ node: _node, ...props }) => (
    <div className="assistant-table-wrap">
      <table {...props} />
    </div>
  ),
};

/**
 * Liga `[1]` à passagem 1 do card de recuperação.
 *
 * Um marcador fora da faixa — o modelo citando a quinta passagem de três — sai
 * como texto comum: um link quebrado afirmaria que a fonte existe.
 */
function citationComponents(count: number): Components {
  if (count === 0) return markdownComponents;
  const cite = (children: ReactNode) =>
    Children.map(children, (child) =>
      typeof child === "string" ? linkedMarkers(child, count) : child,
    );
  return {
    ...markdownComponents,
    p: ({ node: _node, children, ...props }) => <p {...props}>{cite(children)}</p>,
    li: ({ node: _node, children, ...props }) => <li {...props}>{cite(children)}</li>,
    td: ({ node: _node, children, ...props }) => <td {...props}>{cite(children)}</td>,
  };
}

function linkedMarkers(text: string, count: number): ReactNode {
  const parts = text.split(/(\[\d{1,2}\])/g);
  if (parts.length === 1) return text;
  return parts.map((part, index) => {
    const match = /^\[(\d{1,2})\]$/.exec(part);
    const marker = match ? Number(match[1]) : 0;
    if (!marker || marker > count) return part;
    return (
      <a className="citation-marker" href={`#${passageAnchor(marker)}`} key={`${index}-${part}`}>
        {part}
      </a>
    );
  });
}

export function App({ client = harnessClient }: AppProps) {
  const [area, setArea] = useState<AppArea>("chat");
  const [data, setData] = useState<AppData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [session, setSession] = useState<SessionStatus | null>(null);

  // Setup comes first: on an unconfigured host every other endpoint is useless,
  // and a host behind a password answers nothing else until the Operator logs in.
  useEffect(() => {
    let active = true;
    setLoadError(null);

    client
      .getSetupStatus()
      .then(async (status) => {
        if (!active) return;
        setSetupStatus(status);
        if (status.required) return;
        const current = await client.getSessionStatus();
        if (active) setSession(current);
      })
      .catch((error: unknown) => {
        if (active) setLoadError(errorMessage(error));
      });

    return () => {
      active = false;
    };
  }, [client]);

  const gateOpen =
    setupStatus !== null &&
    !setupStatus.required &&
    session !== null &&
    (!session.authentication_required || session.authenticated);

  useEffect(() => {
    if (!gateOpen || data !== null) return;
    let active = true;

    Promise.all([
      client.getChatSnapshot(),
      client.getEvalsSnapshot(),
      client.getSettingsSnapshot(),
    ])
      .then(([chat, evals, settings]) => {
        if (active) setData({ chat, evals, settings });
      })
      .catch((error: unknown) => {
        if (active) setLoadError(errorMessage(error));
      });

    return () => {
      active = false;
    };
  }, [client, gateOpen, data]);

  if (setupStatus?.required) {
    return <SetupArea client={client} status={setupStatus} onDone={setSetupStatus} />;
  }

  if (session?.authentication_required && !session.authenticated) {
    return <LoginArea client={client} onAuthenticated={setSession} />;
  }

  if (setupStatus?.restart_required) {
    return (
      <div className="setup-page">
        <div className="setup-card" role="status">
          <header>
            <span className="loading-mark">H2</span>
            <h1>Configuração gravada</h1>
            <p>Reinicie o servidor para que ela entre em vigor e recarregue esta página.</p>
          </header>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="primary-nav" aria-label="Áreas principais">
        <nav>
          {areaItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={area === item.id ? "active" : ""}
                type="button"
                key={item.id}
                onClick={() => setArea(item.id)}
                aria-current={area === item.id ? "page" : undefined}
                aria-label={item.label}
                title={item.label}
              >
                <Icon size={23} strokeWidth={2} />
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="area-stage">
        {loadError ? (
          <LoadError message={loadError} />
        ) : !data ? (
          <LoadingShell />
        ) : (
          <>
            {area === "chat" && (
              <ChatArea
                snapshot={data.chat}
                client={client}
                onSnapshot={(chat) => setData((current) => current && { ...current, chat })}
              />
            )}
            {area === "corpus" && <CorpusArea client={client} />}
            {area === "evals" && (
              <EvalsArea
                snapshot={data.evals}
                client={client}
                onSnapshot={(evals) =>
                  setData((current) => current && { ...current, evals })
                }
              />
            )}
            {area === "settings" && (
              <SettingsArea snapshot={data.settings} client={client} />
            )}
          </>
        )}
      </main>
    </div>
  );
}

function LoginArea({
  client,
  onAuthenticated,
}: {
  client: HarnessClient;
  onAuthenticated: (status: SessionStatus) => void;
}) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    void client
      .login(password)
      .then(onAuthenticated)
      .catch((cause: unknown) => setError(errorMessage(cause)))
      .finally(() => {
        setBusy(false);
        setPassword("");
      });
  };

  return (
    <div className="setup-page">
      <form className="setup-card" onSubmit={submit}>
        <header>
          <span className="loading-mark">H2</span>
          <h1>Entrar no Harness</h1>
          <p>
            Este host exige a senha de Operator. A sessão vive só nesta aba: recarregar
            a página pede a senha de novo.
          </p>
        </header>

        <label className="setup-field">
          <span>Senha de Operator</span>
          <input
            type="password"
            value={password}
            required
            aria-label="Senha de Operator"
            autoComplete="current-password"
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>

        {error && (
          <div className="action-error setup-error" role="alert">
            {error}
          </div>
        )}

        <button className="primary-button" type="submit" disabled={busy}>
          {busy ? "Entrando…" : "Entrar"}
        </button>
      </form>
    </div>
  );
}

const setupFields = [
  {
    name: "allowed_workspace_roots",
    label: "Raízes de Workspace autorizadas",
    hint: "Diretórios que as tools de arquivo do agente podem ler e gravar, um caminho absoluto por linha. Qualquer caminho fora daqui é recusado, mesmo que o modelo peça.",
    multiline: true,
    required: true,
  },
  {
    name: "state_dir",
    label: "Diretório de estado",
    hint: "Onde ficam o CanonicalHistory, a telemetria e o tokenizer.",
    multiline: false,
    required: true,
  },
  {
    name: "tokenizer_path",
    label: "Caminho do tokenizer.json",
    hint: "Arquivo HuggingFace usado para o orçamento de contexto. O digest é calculado automaticamente.",
    multiline: false,
    required: true,
  },
  {
    name: "allowed_origins",
    label: "Origins autorizadas",
    hint: "De onde o navegador pode chamar esta API — protocolo, host e porta (ex.: http://127.0.0.1:8765), um por linha. Requisição com outro Origin é recusada. Já vem preenchido com a origin desta aba.",
    multiline: true,
    required: true,
  },
  {
    name: "searxng_url",
    label: "Instância SearXNG (opcional)",
    hint: "URL de uma instância SearXNG com format=json, ex.: http://127.0.0.1:8080/search. Sem ela, web_search usa o DuckDuckGo, que não pede chave.",
    multiline: false,
    required: false,
  },
] as const;

function SetupArea({
  client,
  status,
  onDone,
}: {
  client: HarnessClient;
  status: SetupStatus;
  onDone: (status: SetupStatus) => void;
}) {
  const [token, setToken] = useState("");
  const [values, setValues] = useState<Record<string, string>>({
    state_dir: status.suggested_state_dir,
    tokenizer_path: status.suggested_tokenizer_path,
    allowed_origins: withLoopbackPair([window.location.origin]).join("\n"),
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const lines = (name: string) =>
    (values[name] ?? "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const searxng = (values.searxng_url ?? "").trim();
    void client
      .completeSetup(token.trim(), {
        allowed_workspace_roots: lines("allowed_workspace_roots"),
        state_dir: (values.state_dir ?? "").trim(),
        tokenizer_path: (values.tokenizer_path ?? "").trim(),
        allowed_origins: lines("allowed_origins"),
        searxng_url: searxng === "" ? null : searxng,
      })
      .then(() => onDone({ ...status, configured: true, required: false, restart_required: true }))
      .catch((cause: unknown) => setError(errorMessage(cause)))
      .finally(() => setBusy(false));
  };

  return (
    <div className="setup-page">
      <form className="setup-card" onSubmit={submit}>
        <header>
          <span className="loading-mark">H2</span>
          <h1>Configurar este host</h1>
          <p>
            O servidor imprimiu um token de setup no stderr ao subir. Ele expira, vale uma
            vez e só é aceito por uma conexão direta de loopback.
          </p>
        </header>

        <label className="setup-field">
          <span>Token de setup</span>
          <input
            type="password"
            value={token}
            required
            aria-label="Token de setup"
            autoComplete="off"
            onChange={(event) => setToken(event.target.value)}
          />
          <small>Copie a linha "Harness setup: /setup token=…" do terminal.</small>
        </label>

        {setupFields.map((field) => (
          <label className="setup-field" key={field.name}>
            <span>{field.label}</span>
            {field.multiline ? (
              <textarea
                rows={3}
                required={field.required}
                aria-label={field.label}
                value={values[field.name] ?? ""}
                onChange={(event) =>
                  setValues((current) => ({ ...current, [field.name]: event.target.value }))
                }
              />
            ) : (
              <input
                type="text"
                required={field.required}
                aria-label={field.label}
                autoComplete="off"
                value={values[field.name] ?? ""}
                onChange={(event) =>
                  setValues((current) => ({ ...current, [field.name]: event.target.value }))
                }
              />
            )}
            <small>{field.hint}</small>
          </label>
        ))}

        {error && (
          <div className="action-error setup-error" role="alert">
            {error}
          </div>
        )}

        <button className="primary-button" type="submit" disabled={busy}>
          {busy ? "Gravando…" : "Concluir setup"}
        </button>
        <p className="setup-footnote">
          Depois de concluir, reinicie o servidor para que a configuração entre em vigor.
        </p>
      </form>
    </div>
  );
}

function LoadingShell() {
  return (
    <div className="loading-shell" role="status" aria-live="polite">
      <span className="loading-mark">H2</span>
      <p>Carregando o Harness…</p>
    </div>
  );
}

function LoadError({ message }: { message: string }) {
  return (
    <div className="load-error" role="alert">
      <WifiOff size={26} />
      <h1>Não foi possível carregar o Harness</h1>
      <p>{message}</p>
    </div>
  );
}

const TURN_POLL_MS = 1500;

function ChatArea({
  snapshot,
  client,
  onSnapshot,
}: {
  snapshot: ChatSnapshot;
  client: HarnessClient;
  onSnapshot: (snapshot: ChatSnapshot) => void;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [liveRuns, setLiveRuns] = useState<Record<string, LiveRun>>({});
  const controllers = useRef(new Set<AbortController>());
  const snapshotRequest = useRef(0);
  // A conversa que o Operator escolheu, atualizada no clique e não quando o
  // snapshot chega: um refresh disparado por outra conversa, que chegue depois,
  // não pode trocar a tela por baixo dele.
  const selectedId = useRef(snapshot.conversationId);

  useEffect(
    () => () => {
      for (const controller of controllers.current) controller.abort();
      controllers.current.clear();
    },
    [],
  );

  const replaceSnapshot = async (load: () => Promise<ChatSnapshot>) => {
    const request = ++snapshotRequest.current;
    const next = await load();
    if (request === snapshotRequest.current) {
      selectedId.current = next.conversationId;
      onSnapshot(next);
    }
    return next;
  };

  const refresh = async (conversationId = selectedId.current) => {
    // O run de uma conversa que o Operator deixou segue vivo, mas não manda
    // mais na tela: o snapshot dela só vale quando ela voltar a ser a aberta.
    if (conversationId !== selectedId.current) return null;
    return replaceSnapshot(() => client.getChatSnapshot(conversationId));
  };

  const runAction = async (name: string, action: () => Promise<void>) => {
    setBusyAction(name);
    setActionError(null);
    try {
      await action();
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    } finally {
      setBusyAction(null);
    }
  };

  const selectConversation = (conversationId: string) => {
    // Os streams das outras conversas continuam: ao voltar, o progresso está lá.
    selectedId.current = conversationId;
    void runAction("select", async () => {
      await replaceSnapshot(() => client.selectConversation(conversationId));
      setSidebarOpen(false);
    });
  };

  const createConversation = async (root: string, name: string) => {
    return runAction("create", async () => {
      const conversation = await client.createConversation(root, name);
      selectedId.current = conversation.id;
      await replaceSnapshot(() => client.selectConversation(conversation.id));
      setSidebarOpen(false);
    });
  };

  const renameConversation = () => {
    if (!snapshot.conversationId || !snapshot.conversationTitle) return;
    const name = window.prompt("Novo nome da conversa", snapshot.conversationTitle)?.trim();
    if (!name || name === snapshot.conversationTitle) return;
    void runAction("rename", async () => {
      await client.renameConversation(snapshot.conversationId!, name);
      await refresh();
    });
  };

  const archiveConversation = () => {
    if (!snapshot.conversationId) return;
    void runAction("archive", async () => {
      await client.archiveConversation(snapshot.conversationId!);
      selectedId.current = null;
      await replaceSnapshot(() => client.getChatSnapshot(null));
    });
  };

  const deleteConversation = () => {
    if (!snapshot.conversationId) return;
    if (!window.confirm("Excluir esta conversa permanentemente?")) return;
    void runAction("delete", async () => {
      await client.deleteConversation(snapshot.conversationId!);
      selectedId.current = null;
      await replaceSnapshot(() => client.getChatSnapshot(null));
    });
  };

  const toggleGrant = (permission: "WriteGrant" | "WebAccessGrant") => {
    if (!snapshot.conversationId) return;
    const current = snapshot.grants.find((grant) => grant.permission === permission);
    void runAction(`grant-${permission}`, async () => {
      if (current) await client.revokeGrant(snapshot.conversationId!, current.id);
      else await client.addGrant(snapshot.conversationId!, permission);
      await refresh();
    });
  };

  const send = (content: string) => {
    if (!snapshot.conversationId) return;
    const conversationId = snapshot.conversationId;
    const runId = makeId();
    const controller = new AbortController();
    controllers.current.add(controller);
    setActionError(null);
    setLiveRuns((current) => ({
      ...current,
      [runId]: {
        runId,
        conversationId,
        status: "queued",
        reasoning: "",
        content: "",
        tools: [],
        toolArguments: {},
        steps: 0,
      },
    }));

    const onEvent = (event: AgUiEvent) => {
      setLiveRuns((current) => {
        const run = current[runId];
        if (!run) return current;
        return { ...current, [runId]: reduceLiveRun(run, event) };
      });
      const isConfirmationEvent =
        event.type === "CUSTOM" &&
        (event.name === "harness.confirmation_required" ||
          event.name === "harness.confirmation_resolved");
      if (
        event.type === "RUN_STARTED" ||
        event.type === "STEP_STARTED" ||
        isConfirmationEvent
      ) {
        void refresh(conversationId).catch((error) => setActionError(errorMessage(error)));
      }
    };

    void client
      .streamAgent(
        {
          threadId: conversationId,
          runId,
          messages: [{ id: `user-${runId}`, role: "user", content }],
        },
        onEvent,
        controller.signal,
      )
      .then(async (terminal) => {
        if (terminal.type === "RUN_ERROR") setActionError(terminal.message);
        await refresh(conversationId);
        setLiveRuns((current) => {
          const next = { ...current };
          delete next[runId];
          return next;
        });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message = errorMessage(error);
        setActionError(message);
        setLiveRuns((current) => {
          const run = current[runId];
          return run
            ? { ...current, [runId]: { ...run, status: "error", error: message } }
            : current;
        });
      })
      .finally(() => controllers.current.delete(controller));
  };

  const editPending = async (requestId: string, content: string) => {
    if (!snapshot.conversationId) return;
    await client.editPendingRequest(snapshot.conversationId, requestId, content);
    await refresh();
  };

  const cancelPending = async (requestId: string) => {
    if (!snapshot.conversationId) return;
    await client.cancelPendingRequest(snapshot.conversationId, requestId);
    await refresh();
  };

  const stop = () => {
    if (!snapshot.conversationId) return;
    void runAction("stop", async () => {
      await client.stop(snapshot.conversationId!);
      await refresh();
    });
  };

  const resolveConfirmation = (approved: boolean, waive: boolean) => {
    const pending = snapshot.pendingConfirmation;
    if (!snapshot.conversationId || !pending) return;
    void runAction(approved ? "confirm-approve" : "confirm-deny", async () => {
      await client.resolveConfirmation(snapshot.conversationId!, pending.id, approved, waive);
      await refresh();
    });
  };

  const toggleConversationYolo = () => {
    if (!snapshot.conversationId) return;
    void runAction("yolo-conversation", async () => {
      if (snapshot.yolo) {
        // Desligar é decisão local: as outras conversas continuam como estavam.
        await client.setConversationYolo(snapshot.conversationId!, true);
      } else {
        // Ligar em um clique é o ponto do modo: liga no host e tira a exceção daqui.
        await client.setYoloEnabled(true);
        await client.setConversationYolo(snapshot.conversationId!, false);
      }
      await refresh();
    });
  };

  const revokeConfirmationWaiver = () => {
    if (!snapshot.conversationId) return;
    void runAction("waiver-revoke", async () => {
      await client.revokeConfirmationWaiver(snapshot.conversationId!, "workspace_write");
      await refresh();
    });
  };

  const runsHere = Object.values(liveRuns).filter(
    (run) => run.conversationId === snapshot.conversationId,
  );
  const liveMessages = runsHere
    // O consumo da janela também segura a mensagem ao vivo: ele chega antes do
    // primeiro token, que é justamente quando saber quanto sobrou tem valor.
    .filter(
      (run) => run.content || run.reasoning || run.tools.length || run.error || run.contextUsage,
    )
    .map(liveRunMessage);
  const streamingHere = runsHere.some(
    (run) => run.status === "queued" || run.status === "running",
  );
  const isRunning = snapshot.activeTurn !== null || streamingHere;
  // Turn ativo ou fila sem um stream desta aba para avisar do fim: aconteceu
  // depois de recarregar a página ou de trocar de conversa e voltar, e a UI
  // ficava mostrando "rodando" até o Operator recarregar de novo.
  const needsPolling =
    !streamingHere &&
    (snapshot.activeTurn !== null || snapshot.pendingRequests.length > 0);
  useEffect(() => {
    if (!needsPolling || !snapshot.conversationId) return;
    const conversationId = snapshot.conversationId;
    const timer = window.setTimeout(() => {
      void refresh(conversationId).catch(() => undefined);
    }, TURN_POLL_MS);
    return () => window.clearTimeout(timer);
    // Cada snapshot novo reagenda o próximo; refresh confere a conversa pela ref.
  }, [needsPolling, snapshot]);
  const hasWorkspaces = snapshot.workspaces.length > 0;
  const hasConversation = snapshot.conversationId !== null;

  return (
    <div className="chat-layout">
      {sidebarOpen && (
        <button
          className="drawer-scrim"
          type="button"
          onClick={() => setSidebarOpen(false)}
          aria-label="Fechar conversas"
        />
      )}
      <ConversationSidebar
        groups={snapshot.workspaces}
        selectedId={snapshot.conversationId}
        open={sidebarOpen}
        creating={creatingConversation}
        onToggleCreating={setCreatingConversation}
        busy={busyAction === "create" || busyAction === "select"}
        onClose={() => setSidebarOpen(false)}
        onSelect={selectConversation}
        onCreate={createConversation}
      />

      <section className="chat-stage" aria-labelledby="conversation-title">
        <header className="conversation-header">
          <button
            className="icon-button mobile-menu"
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Abrir conversas"
            aria-expanded={sidebarOpen}
          >
            <Menu size={18} />
          </button>
          <div className="conversation-identity">
            <span className="eyebrow">{snapshot.workspaceName ?? "Harness"}</span>
            <h1 id="conversation-title">
              {snapshot.conversationTitle ??
                (hasWorkspaces ? "Nenhuma conversa" : "Workspace não autorizado")}
            </h1>
          </div>
          {hasConversation && (
            <>
              <GrantChips
                grants={snapshot.grants}
                busyAction={busyAction}
                onToggle={toggleGrant}
              />
              <button
                className={snapshot.yolo ? "yolo-chip yolo-chip--on" : "yolo-chip"}
                type="button"
                onClick={toggleConversationYolo}
                disabled={busyAction === "yolo-conversation"}
                aria-pressed={snapshot.yolo}
                title={
                  snapshot.yolo
                    ? "Yolo ligado: nada é confirmado nesta conversa, nem escrita vinda da web. Clique para voltar a confirmar aqui."
                    : "Clique para o modelo agir sem confirmação — inclusive escrita derivada de conteúdo da web."
                }
              >
                {snapshot.yolo ? <Zap size={12} /> : <ZapOff size={12} />}
                Yolo {snapshot.yolo ? "ligado" : "desligado"}
              </button>
              {snapshot.confirmationWaivers.includes("workspace_write") && (
                <button
                  className="waiver-chip"
                  type="button"
                  onClick={revokeConfirmationWaiver}
                  disabled={busyAction === "waiver-revoke"}
                  title="Voltar a confirmar cada escrita nesta conversa"
                >
                  <ShieldAlert size={12} />
                  Escritas sem confirmação
                </button>
              )}
              <div className="conversation-actions" aria-label="Ações da conversa">
                <button
                  className="icon-button"
                  type="button"
                  onClick={renameConversation}
                  disabled={busyAction !== null}
                  aria-label="Renomear conversa"
                >
                  <Pencil size={14} />
                </button>
                <button
                  className="icon-button"
                  type="button"
                  onClick={archiveConversation}
                  disabled={busyAction !== null}
                  aria-label="Arquivar conversa"
                >
                  <Archive size={14} />
                </button>
                <button
                  className="icon-button icon-button--danger"
                  type="button"
                  onClick={deleteConversation}
                  disabled={busyAction !== null}
                  aria-label="Excluir conversa"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </>
          )}
        </header>

        {actionError && (
          <div className="action-error" role="alert">
            {actionError}
          </div>
        )}

        {!hasWorkspaces ? (
          <EmptyState
            title="Nenhuma raiz de Workspace autorizada"
            detail="Configure HARNESS_WORKSPACE_ROOTS no servidor para iniciar uma conversa."
          />
        ) : !hasConversation ? (
          <EmptyState
            title="Crie a primeira conversa"
            detail="Escolha uma das raízes autorizadas no painel de conversas."
            action={() => {
              const root = snapshot.workspaces[0]?.root;
              if (root) void createConversation(root, "Nova conversa");
            }}
          />
        ) : (
          <>
            <div className="timeline" id="chat-timeline">
              {isRunning && (
                <div className="run-strip" role="status">
                  <span className="pulse-dot" />
                  <span>
                    {snapshot.activeTurn ? "Turn em execução" : "Aguardando na fila"}
                  </span>
                  <span>{runsHere.length} stream(s)</span>
                </div>
              )}
              {!snapshot.messages.length && !liveMessages.length && (
                <div className="timeline-empty">
                  <MessageSquare size={21} />
                  <h2>Conversa vazia</h2>
                  <p>Envie uma solicitação para iniciar o CanonicalHistory.</p>
                </div>
              )}
              {[...snapshot.messages, ...liveMessages].map((message) => (
                <TimelineMessage
                  message={message}
                  key={message.id}
                  conversationId={snapshot.conversationId!}
                  client={client}
                  existingFeedback={snapshot.feedback.find(
                    (item) => item.turn_id === message.turnId,
                  )}
                  grants={snapshot.grants}
                  onGrant={toggleGrant}
                />
              ))}
              <PendingQueue
                requests={snapshot.pendingRequests}
                onEdit={editPending}
                onCancel={cancelPending}
              />
            </div>
            <Composer
              isRunning={isRunning}
              isStopping={busyAction === "stop"}
              onStop={stop}
              onSend={send}
              onAttach={
                snapshot.conversationId
                  ? (file) => client.attachImage(snapshot.conversationId as string, file)
                  : undefined
              }
            />
          </>
        )}
      </section>

      <ChatContext
        snapshot={snapshot}
        onSelectCorpus={(corpusId) => {
          const conversationId = snapshot.conversationId;
          if (!conversationId) return;
          void runAction("corpus", async () => {
            await client.selectCorpus(conversationId, corpusId);
            await refresh(conversationId);
          });
        }}
      />
      {snapshot.pendingConfirmation && (
        <ConfirmationDialog
          confirmation={snapshot.pendingConfirmation}
          busy={busyAction === "confirm-approve" || busyAction === "confirm-deny"}
          onResolve={resolveConfirmation}
        />
      )}
    </div>
  );
}

function ConversationSidebar({
  groups,
  selectedId,
  open,
  creating,
  onToggleCreating,
  busy,
  onClose,
  onSelect,
  onCreate,
}: {
  groups: WorkspaceGroup[];
  selectedId: string | null;
  open: boolean;
  creating: boolean;
  onToggleCreating: (creating: boolean | ((current: boolean) => boolean)) => void;
  busy: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onCreate: (root: string, name: string) => Promise<boolean>;
}) {
  const [root, setRoot] = useState(groups[0]?.root ?? "");
  const [name, setName] = useState("Nova conversa");
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!groups.some((group) => group.root === root)) setRoot(groups[0]?.root ?? "");
  }, [groups, root]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!root || !name.trim()) return;
    void onCreate(root, name.trim()).then((created) => {
      if (created) onToggleCreating(false);
    });
  };

  const normalizedSearch = search.trim().toLocaleLowerCase("pt-BR");

  return (
    <aside
      className={`conversation-sidebar ${open ? "open" : ""}`}
      aria-label="Conversas por Workspace"
    >
      <div className="sidebar-topline">
        <div>
          <span className="eyebrow">Harness</span>
          <strong>Conversas</strong>
        </div>
        <button
          className="icon-button close-drawer"
          type="button"
          onClick={onClose}
          aria-label="Fechar conversas"
        >
          <X size={17} />
        </button>
      </div>
      <button
        className="new-chat-button"
        type="button"
        disabled={!groups.length || busy}
        onClick={() => onToggleCreating((current) => !current)}
        aria-expanded={creating}
      >
        <Plus size={16} /> Nova conversa
      </button>
      {creating && (
        <form className="new-chat-form" onSubmit={submit}>
          <label>
            <span>Raiz autorizada</span>
            <select
              value={root}
              onChange={(event) => setRoot(event.target.value)}
              aria-label="Raiz autorizada"
            >
              {groups.map((group) => (
                <option value={group.root} key={group.id}>
                  {group.root}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Nome</span>
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <button className="primary-button" type="submit" disabled={busy || !name.trim()}>
            Criar
          </button>
        </form>
      )}
      <label className="sidebar-search">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Buscar conversas</span>
        <input
          type="search"
          placeholder="Buscar"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </label>
      <div className="workspace-groups">
        {groups.map((group) => {
          const conversations = group.conversations.filter((conversation) =>
            conversation.title.toLocaleLowerCase("pt-BR").includes(normalizedSearch),
          );
          return (
            <section className="workspace-group" key={group.id}>
              <div className="workspace-heading">
                <ChevronDown size={13} />
                <h2>{group.name}</h2>
                <span>{conversations.length}</span>
              </div>
              <div className="conversation-list">
                {conversations.map((conversation) => (
                  <button
                    className={conversation.id === selectedId ? "active" : ""}
                    type="button"
                    key={conversation.id}
                    onClick={() => onSelect(conversation.id)}
                    aria-current={conversation.id === selectedId ? "page" : undefined}
                  >
                    <span className="conversation-name">{conversation.title}</span>
                    <time>{formatTimestamp(conversation.updatedAt)}</time>
                  </button>
                ))}
              </div>
            </section>
          );
        })}
      </div>
      <div className="sidebar-footer-note">
        <ShieldCheck size={14} />
        WorkspaceRootGrant define o limite local
      </div>
    </aside>
  );
}

const OUTCOME_HINTS: Record<string, string> = {
  // O pedido de grant vira diálogo no meio do Turn: estes dois códigos só sobram
  // quando ninguém respondeu por ele (execução sem Operator).
  write_grant_required: "O pedido de grant Write não foi respondido nesta sessão.",
  web_access_grant_required: "O pedido de grant Web não foi respondido nesta sessão.",
  write_grant_denied: "O grant Write foi negado, então a escrita não rodou.",
  web_access_grant_denied: "O grant Web foi negado, então a chamada não rodou.",
  workspace_root_grant_required: "A conversa não tem workspace root. Selecione uma raiz permitida.",
  sensitive_path_denied: "O caminho pedido é sensível e a policy nega o acesso.",
  host_path_denied: "A policy do host nega este caminho.",
  path_outside_workspace: "O caminho resolvido cai fora da workspace root.",
  web_taint_confirmation_denied: "A confirmação da ação sob taint web foi negada.",
  malformed_model_response_limit:
    "O modelo respondeu sem texto e sem tool call. Reenvie ou simplifique o pedido.",
  context_budget_exceeded: "O contexto estourou o orçamento. Abra outra conversa ou reduza o pedido.",
  runtime_verification_failed:
    "O runtime Ollama não confere com o perfil esperado. Verifique o modelo instalado.",
  model_invocation_limit: "O turno atingiu o limite de passos. Divida a tarefa.",
  tool_calls_per_turn_limit: "O turno atingiu o limite de tool calls. Divida a tarefa.",
  tool_calls_per_step_limit: "O passo pediu tool calls demais de uma vez.",
  rejected_model_attempt_limit: "O modelo insistiu em chamadas inválidas e o turno foi encerrado.",
  model_provider_unavailable:
    "O Ollama não respondeu. Verifique se o serviço está no ar e reenvie o pedido.",
  model_provider_error: "O Ollama recusou a chamada. O detalhe abaixo traz a resposta dele.",
  model_generation_timeout:
    "A geração passou do tempo máximo e o harness a encerrou. O Ollama está no ar; o pedido é que é longo demais.",
  malformed_model_response: "A resposta do modelo não pôde ser lida e o turno parou aqui.",
  engine_error: "Falha interna do harness, não do modelo. O detalhe abaixo identifica onde.",
  model_not_installed: "O modelo do perfil ativo não está instalado no Ollama.",
  model_digest_mismatch:
    "O modelo instalado não bate com o digest do perfil. Reinstale a partir do Modelfile.",
  runtime_not_verified: "O runtime ainda não foi verificado nesta inicialização.",
  turn_time_budget_exhausted: "O turno estourou o tempo máximo. Divida a tarefa.",
};

const GRANT_FOR_REASON: Record<string, "WriteGrant" | "WebAccessGrant"> = {
  write_grant_required: "WriteGrant",
  web_access_grant_required: "WebAccessGrant",
};

export function TurnOutcome({
  kind,
  reasonCode,
  detail,
  grants,
  onGrant,
}: {
  kind: string;
  reasonCode: string;
  detail?: string | null;
  grants: Grant[];
  onGrant: (permission: "WriteGrant" | "WebAccessGrant") => void;
}) {
  const hint = OUTCOME_HINTS[reasonCode];
  const permission = GRANT_FOR_REASON[reasonCode];
  const missing = permission && !grants.some((grant) => grant.permission === permission);
  return (
    <div className="turn-outcome-block">
      <div className={`turn-outcome turn-outcome--${kind}`}>
        <span>{kind}</span>
        <code>{reasonCode}</code>
      </div>
      {hint && <p className="turn-outcome-hint">{hint}</p>}
      {/* The provider's own class lives here and nowhere else. */}
      {detail && <p className="turn-outcome-detail">{detail}</p>}
      {missing && permission && (
        <button type="button" className="secondary-button" onClick={() => onGrant(permission)}>
          Ativar grant {permission === "WriteGrant" ? "Write" : "Web"}
        </button>
      )}
    </div>
  );
}

function GrantChips({
  grants,
  busyAction,
  onToggle,
}: {
  grants: Grant[];
  busyAction: string | null;
  onToggle: (permission: "WriteGrant" | "WebAccessGrant") => void;
}) {
  const root = grants.find((grant) => grant.permission === "WorkspaceRootGrant");
  return (
    <div className="permission-chips" aria-label="Grants da conversa">
      {root && (
        <span className="permission permission--active" title={root.scope}>
          <Search size={12} /> Leitura
        </span>
      )}
      {(
        [
          ["WriteGrant", "Write", "Escrita", FileCode2],
          ["WebAccessGrant", "Web", "Web", Globe2],
        ] as const
      ).map(([permission, variant, label, Icon]) => {
        const active = grants.some((grant) => grant.permission === permission);
        return (
          <button
            type="button"
            className={`permission permission--${variant} ${active ? "permission--active" : ""}`}
            key={permission}
            onClick={() => onToggle(permission)}
            disabled={busyAction === `grant-${permission}`}
            aria-pressed={active}
            aria-label={`${active ? "Revogar" : "Ativar"} ${label}`}
          >
            <Icon size={12} /> {label}
          </button>
        );
      })}
    </div>
  );
}


function TimelineMessage({
  message,
  conversationId,
  client,
  existingFeedback,
  grants,
  onGrant,
}: {
  message: ChatMessage;
  conversationId: string;
  client: HarnessClient;
  existingFeedback?: FeedbackRecord;
  grants: Grant[];
  onGrant: (permission: "WriteGrant" | "WebAccessGrant") => void;
}) {
  if (message.role === "user") {
    return (
      <article className="message message--user">
        <div className="message-meta">
          <span>Você</span>
          <time>{formatTimestamp(message.createdAt)}</time>
        </div>
        <p>{message.content}</p>
      </article>
    );
  }

  return (
    <article className="message message--assistant">
      <div className="assistant-marker" aria-hidden="true">
        <Sparkles size={16} />
      </div>
      <div className="assistant-content">
        <div className="message-meta">
          <span>Modelo</span>
          <time>{formatTimestamp(message.createdAt)}</time>
          {message.live && <em className="live-label">ao vivo</em>}
        </div>
        {message.reasoning && (
          <details className="reasoning-block">
            <summary>
              <Brain size={14} />
              <span>{message.reasoning.summary}</span>
              <em>transitório</em>
              <ChevronDown size={14} className="summary-chevron" />
            </summary>
            <p>{message.reasoning.content}</p>
          </details>
        )}
        {message.contextUsage && <ContextGauge usage={message.contextUsage} />}
        {message.retrieval && <RetrievalCard retrieval={message.retrieval} />}
        {message.tools && message.tools.length > 0 && (
          <div className="tool-stack" aria-label="Chamadas de tools">
            {message.tools.map((tool) => (
              <ToolCallCard tool={tool} key={tool.id} />
            ))}
          </div>
        )}
        {message.events?.map((event) => (
          <details className="canonical-event" key={event.id}>
            <summary>
              {event.kind} <ChevronDown size={14} className="summary-chevron" />
            </summary>
            <pre>{JSON.stringify(event.payload, null, 2)}</pre>
          </details>
        ))}
        {message.content && (
          <div className="assistant-answer">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={citationComponents(message.retrieval?.passages.length ?? 0)}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {message.metrics && <Metrics metrics={message.metrics} />}
        {message.terminalOutcome && (
          <TurnOutcome
            kind={message.terminalOutcome.kind}
            reasonCode={message.terminalOutcome.reason_code}
            detail={message.terminalOutcome.detail}
            grants={grants}
            onGrant={onGrant}
          />
        )}
        {message.liveOutcome && (
          <TurnOutcome
            kind={message.liveOutcome.kind}
            reasonCode={message.liveOutcome.reasonCode}
            grants={grants}
            onGrant={onGrant}
          />
        )}
        {!message.live && message.content && (
          <Feedback
            conversationId={conversationId}
            turnId={message.turnId}
            client={client}
            initial={existingFeedback}
          />
        )}
      </div>
    </article>
  );
}

function Metrics({ metrics }: { metrics: Record<string, string | number> }) {
  return (
    <details className="metrics-block">
      <summary>
        <Activity size={14} /> Métricas fornecidas
        <ChevronDown className="summary-chevron" size={14} />
      </summary>
      <div className="metrics-grid">
        {Object.entries(metrics).map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </details>
  );
}

function Feedback({
  conversationId,
  turnId,
  client,
  initial,
}: {
  conversationId: string;
  turnId: string;
  client: HarnessClient;
  initial?: FeedbackRecord;
}) {
  const [feedback, setFeedback] = useState<FeedbackRecord | undefined>(initial);
  const [draftId, setDraftId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const rate = (rating: -1 | 1) => {
    setBusy(true);
    setError(null);
    void client
      .addFeedback(conversationId, rating, turnId)
      .then(setFeedback)
      .catch((caught) => setError(errorMessage(caught)))
      .finally(() => setBusy(false));
  };

  const exportDraft = () => {
    if (!feedback) return;
    setBusy(true);
    setError(null);
    void client
      .createRegressionDraft(conversationId, feedback.id)
      .then((draft) => setDraftId(draft.id))
      .catch((caught) => setError(errorMessage(caught)))
      .finally(() => setBusy(false));
  };

  return (
    <div className="feedback-row" aria-label="Feedback da resposta">
      <span>
        {draftId
          ? `Rascunho ${draftId} criado`
          : feedback
            ? "Feedback registrado"
            : "Esta resposta ajudou?"}
      </span>
      <button
        className={feedback?.rating === 1 ? "active" : ""}
        type="button"
        onClick={() => rate(1)}
        disabled={busy}
        aria-pressed={feedback?.rating === 1}
      >
        <ThumbsUp size={13} /> Funcionou
      </button>
      <button
        className={feedback?.rating === -1 ? "active negative" : ""}
        type="button"
        onClick={() => rate(-1)}
        disabled={busy}
        aria-pressed={feedback?.rating === -1}
      >
        <ThumbsDown size={13} /> Não funcionou
      </button>
      {feedback?.rating === -1 && !draftId && (
        <button type="button" onClick={exportDraft} disabled={busy}>
          Exportar rascunho
        </button>
      )}
      {error && <span className="inline-error" role="alert">{error}</span>}
    </div>
  );
}

type Attachment = { path: string; name: string; preview: string };

// Os formatos que o describe_image lê; o servidor confere os bytes de novo.
const ATTACHABLE_TYPES = ["image/png", "image/jpeg", "image/webp"];
const DEFAULT_IMAGE_QUESTION = "O que tem nesta imagem?";

export function withAttachments(prompt: string, paths: string[]): string {
  if (paths.length === 0) return prompt;
  const lines = paths.map((path) => `Imagem anexada: ${path}`);
  return `${prompt || DEFAULT_IMAGE_QUESTION}\n\n${lines.join("\n")}`;
}

function Composer({
  isRunning,
  isStopping,
  onStop,
  onSend,
  onAttach,
}: {
  isRunning: boolean;
  isStopping: boolean;
  onStop: () => void;
  onSend: (prompt: string) => void;
  onAttach?: (file: File) => Promise<string>;
}) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [uploading, setUploading] = useState(0);
  const [attachError, setAttachError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const insert = (text: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return setValue((current) => `${current}${text}`);
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    setValue((current) => `${current.slice(0, start)}${text}${current.slice(end)}`);
    requestAnimationFrame(() => {
      textarea.focus();
      textarea.setSelectionRange(start + text.length, start + text.length);
    });
  };

  const attach = (files: File[]) => {
    if (!onAttach) return;
    const images = files.filter((file) => ATTACHABLE_TYPES.includes(file.type));
    if (images.length < files.length) {
      setAttachError("Só PNG, JPEG e WebP podem ser anexados.");
    } else {
      setAttachError(null);
    }
    for (const file of images) {
      setUploading((count) => count + 1);
      onAttach(file)
        .then((path) => {
          const preview = URL.createObjectURL(file);
          setAttachments((current) => [...current, { path, name: file.name, preview }]);
        })
        .catch((reason: unknown) => {
          setAttachError(reason instanceof Error ? reason.message : "Falha ao anexar a imagem.");
        })
        .finally(() => setUploading((count) => count - 1));
    }
  };

  const detach = (path: string) => {
    setAttachments((current) => {
      const removed = current.find((item) => item.path === path);
      if (removed) URL.revokeObjectURL(removed.preview);
      return current.filter((item) => item.path !== path);
    });
  };

  const canSend = (value.trim().length > 0 || attachments.length > 0) && uploading === 0;

  const submit = () => {
    if (!canSend) return;
    onSend(
      withAttachments(
        value.trim(),
        attachments.map((item) => item.path),
      ),
    );
    for (const item of attachments) URL.revokeObjectURL(item.preview);
    setAttachments([]);
    setValue("");
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.files);
    if (files.length === 0 || !onAttach) return;
    event.preventDefault();
    attach(files);
  };

  return (
    <div className="composer-dock">
      <form className="composer" onSubmit={onSubmit}>
        <label className="sr-only" htmlFor="prompt-composer">
          Solicitação para o Harness
        </label>
        {(attachments.length > 0 || uploading > 0) && (
          <ul className="composer-attachments" aria-label="Imagens anexadas">
            {attachments.map((item) => (
              <li key={item.path}>
                <img src={item.preview} alt={item.name} />
                <span title={item.path}>{item.name}</span>
                <button
                  type="button"
                  aria-label={`Remover ${item.name}`}
                  onClick={() => detach(item.path)}
                >
                  <X size={12} />
                </button>
              </li>
            ))}
            {uploading > 0 && <li className="uploading">Enviando…</li>}
          </ul>
        )}
        <textarea
          id="prompt-composer"
          ref={textareaRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          rows={2}
          placeholder="Peça uma mudança ou faça uma pergunta…"
        />
        <div className="composer-footer">
          <div className="composer-tools">
            <button type="button" onClick={() => insert("```diff\n\n```")}>
              <FileCode2 size={15} /> diff
            </button>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={!onAttach}
              title="Anexar imagem (também dá para colar com Ctrl+V)"
            >
              <ImagePlus size={15} /> imagem
            </button>
            <input
              ref={fileRef}
              type="file"
              accept={ATTACHABLE_TYPES.join(",")}
              multiple
              hidden
              aria-label="Escolher imagem"
              onChange={(event) => {
                attach(Array.from(event.target.files ?? []));
                event.target.value = "";
              }}
            />
          </div>
          <div className="composer-actions">
            <span>Enter envia · ⇧Enter quebra linha</span>
            <button
              className="stop-button"
              type="button"
              onClick={onStop}
              disabled={!isRunning || isStopping}
            >
              <Square size={13} fill="currentColor" /> Parar
            </button>
            <button
              className="send-button"
              type="submit"
              disabled={!canSend}
              aria-label="Enviar solicitação"
            >
              <ArrowUp size={17} />
            </button>
          </div>
        </div>
      </form>
      {attachError && (
        <span className="inline-error" role="alert">
          {attachError}
        </span>
      )}
      <p>Solicitações concorrentes permanecem na fila desta conversa.</p>
    </div>
  );
}

function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: () => void;
}) {
  return (
    <div className="empty-transcript">
      <MessageSquare size={24} />
      <h2>{title}</h2>
      <p>{detail}</p>
      {action && (
        <button className="secondary-button" type="button" onClick={action}>
          Nova conversa
        </button>
      )}
    </div>
  );
}

function ChatContext({
  snapshot,
  onSelectCorpus,
}: {
  snapshot: ChatSnapshot;
  onSelectCorpus: (corpusId: string | null) => void;
}) {
  if (!snapshot.execution) return <aside className="chat-context" />;
  return (
    <aside className="chat-context" aria-label="ExecutionRoute">
      <div className="context-heading">
        <span className="eyebrow">ExecutionRoute</span>
        <span className="context-live">snapshot</span>
      </div>
      {snapshot.corpora.length > 0 && (
        <label className="context-corpus">
          <span className="eyebrow">
            <Library size={13} /> Corpus
          </span>
          <select
            value={snapshot.corpusId ?? ""}
            disabled={!snapshot.conversationId}
            onChange={(event) => onSelectCorpus(event.target.value || null)}
          >
            <option value="">Desligado</option>
            {snapshot.corpora.map((corpus) => (
              <option key={corpus.id} value={corpus.id}>
                {corpus.name}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="context-model">
        <span className="context-model-icon">
          <Bot size={17} />
        </span>
        <div>
          <strong>{snapshot.execution.runtimeProfile}</strong>
          <span>{snapshot.execution.defaultExecutionRoute}</span>
        </div>
      </div>
      <dl className="context-facts">
        {Object.entries(snapshot.execution.loop).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{displayValue(value)}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

function CorpusArea({ client }: { client: HarnessClient }) {
  const [snapshot, setSnapshot] = useState<CorporaSnapshot | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<CorpusDocument[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [seed, setSeed] = useState("");

  const load = async (keep = selectedId) => {
    const next = await client.getCorporaSnapshot();
    setSnapshot(next);
    const selected = next.corpora.find((item) => item.id === keep) ?? next.corpora[0] ?? null;
    setSelectedId(selected?.id ?? null);
    setDocuments(selected ? await client.listCorpusDocuments(selected.id) : []);
  };

  useEffect(() => {
    let active = true;
    client
      .getCorporaSnapshot()
      .then(async (next) => {
        if (!active) return;
        setSnapshot(next);
        const first = next.corpora[0] ?? null;
        setSelectedId(first?.id ?? null);
        if (first) {
          const items = await client.listCorpusDocuments(first.id);
          if (active) setDocuments(items);
        }
      })
      .catch((cause: unknown) => {
        if (active) setError(errorMessage(cause));
      });
    return () => {
      active = false;
    };
  }, [client]);

  // Um job de coleta leva dezenas de minutos: o progresso chega por polling, o
  // mesmo transporte que a aba de Avaliações já usa.
  const running = (snapshot?.jobs ?? []).some(
    (job) => job.status === "running" || job.status === "queued",
  );
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      void client.getCorporaSnapshot().then(setSnapshot).catch(() => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, [client, running]);

  const act = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (cause: unknown) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };

  if (!snapshot) return <LoadingShell />;

  if (!snapshot.available) {
    return (
      <div className="corpus-page">
        <div className="corpus-empty" role="status">
          <Library size={22} />
          <h1>Nenhum modelo de embedding disponível</h1>
          <p>
            O RuntimeProfile ativo não declara um modelo de embedding, ou ele não está
            instalado no Ollama. Instale-o com <code>scripts/install-models.sh</code> e reinicie o
            servidor.
          </p>
        </div>
      </div>
    );
  }

  const selected = snapshot.corpora.find((item) => item.id === selectedId) ?? null;
  const jobs = snapshot.jobs.filter((job) => job.corpus_id === selectedId);

  return (
    <div className="corpus-page">
      <div className="corpus-layout">
        <aside className="corpus-list" aria-label="Corpora">
          <div className="corpus-list-heading">
            <span className="eyebrow">Corpora</span>
            <span>{snapshot.embedding_model}</span>
          </div>
          <ul>
            {snapshot.corpora.map((corpus) => (
              <li key={corpus.id}>
                <button
                  type="button"
                  className={corpus.id === selectedId ? "active" : ""}
                  onClick={() =>
                    void act(async () => {
                      setSelectedId(corpus.id);
                      setDocuments(await client.listCorpusDocuments(corpus.id));
                    })
                  }
                >
                  <strong>{corpus.name}</strong>
                  <span>
                    {corpus.document_count} documentos · {corpus.chunk_count} chunks
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <form
            className="corpus-create"
            onSubmit={(event: FormEvent) => {
              event.preventDefault();
              if (!name.trim()) return;
              void act(async () => {
                const created = await client.createCorpus(name.trim());
                setName("");
                await load(created.id);
              });
            }}
          >
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Nome do novo Corpus"
              aria-label="Nome do novo Corpus"
            />
            <button className="secondary-button" type="submit" disabled={busy}>
              <Plus size={15} /> Criar
            </button>
          </form>
        </aside>

        <main className="corpus-detail">
          {error && (
            <p className="corpus-error" role="alert">
              {error}
            </p>
          )}
          {!selected ? (
            <div className="corpus-empty" role="status">
              <Library size={22} />
              <h1>Nenhum Corpus ainda</h1>
              <p>Crie um para subir arquivos ou coletar uma wiki.</p>
            </div>
          ) : (
            <>
              <header className="corpus-detail-heading">
                <div>
                  <h1>{selected.name}</h1>
                  <p>
                    {selected.document_count} documentos · {selected.chunk_count} chunks ·{" "}
                    {selected.embedding_model} ({selected.embedding_dimensions} dims)
                  </p>
                </div>
                <button
                  className="danger-button"
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    void act(async () => {
                      await client.deleteCorpus(selected.id);
                      await load(null);
                    })
                  }
                >
                  <Trash2 size={15} /> Apagar Corpus
                </button>
              </header>

              <section className="corpus-card">
                <h2>Subir arquivo</h2>
                <p>Aceitos: {snapshot.accepted_extensions.join(", ")}</p>
                <input
                  type="file"
                  aria-label="Arquivo para ingerir"
                  disabled={busy}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (!file) return;
                    void act(async () => {
                      const job = await client.uploadCorpusDocument(selected.id, file);
                      if (job.status === "failed") setError(job.detail ?? "Falha na ingestão.");
                      await load(selected.id);
                    });
                  }}
                />
              </section>

              <section className="corpus-card">
                <h2>Coletar da web</h2>
                <p>
                  Wiki com <code>/api.php</code> é coletada pela API; qualquer outro site cai
                  num crawl com teto, respeitando <code>robots.txt</code>.
                </p>
                <form
                  onSubmit={(event: FormEvent) => {
                    event.preventDefault();
                    if (!seed.trim()) return;
                    void act(async () => {
                      await client.startCorpusScrape(selected.id, seed.trim());
                      setSeed("");
                      setSnapshot(await client.getCorporaSnapshot());
                    });
                  }}
                >
                  <input
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                    placeholder="https://exemplo.fandom.com/wiki/Inicio"
                    aria-label="URL semente da coleta"
                  />
                  <button className="secondary-button" type="submit" disabled={busy}>
                    <Globe2 size={15} /> Coletar
                  </button>
                </form>
                {jobs.length > 0 && (
                  <ul className="corpus-jobs">
                    {jobs.map((job) => (
                      <li key={job.id}>
                        <CorpusJobRow
                          job={job}
                          onCancel={() =>
                            void act(async () => {
                              await client.cancelCorpusJob(selected.id, job.id);
                              setSnapshot(await client.getCorporaSnapshot());
                            })
                          }
                        />
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="corpus-card">
                <h2>Documentos</h2>
                {documents.length === 0 ? (
                  <p>Nenhum documento ingerido.</p>
                ) : (
                  <ul className="corpus-documents">
                    {documents.map((document) => (
                      <li key={document.id}>
                        <div>
                          <strong>{document.title}</strong>
                          <span>
                            {document.origin_kind === "scrape" ? "web" : "upload"} ·{" "}
                            {document.chunk_count} chunks · {document.origin_ref}
                          </span>
                        </div>
                        {document.taints.length > 0 && (
                          <span className="retrieval-taint" title="Coletado da web">
                            <ShieldAlert size={13} /> web
                          </span>
                        )}
                        <button
                          className="icon-button"
                          type="button"
                          aria-label={`Remover ${document.title}`}
                          disabled={busy}
                          onClick={() =>
                            void act(async () => {
                              await client.deleteCorpusDocument(selected.id, document.id);
                              await load(selected.id);
                            })
                          }
                        >
                          <Trash2 size={15} />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function CorpusJobRow({ job, onCancel }: { job: IngestionJob; onCancel: () => void }) {
  const active = job.status === "running" || job.status === "queued";
  return (
    <div className="corpus-job">
      <div>
        <strong>{job.origin}</strong>
        <span>
          {job.status} · {job.indexed} indexados · {job.skipped} pulados · {job.chunks} chunks
          {job.current ? ` · ${job.current}` : ""}
          {job.detail && !active ? ` · ${job.detail}` : ""}
        </span>
      </div>
      {active && (
        <button className="icon-button" type="button" aria-label="Cancelar coleta" onClick={onCancel}>
          <CircleStop size={15} />
        </button>
      )}
    </div>
  );
}

function EvalsArea({
  snapshot,
  client,
  onSnapshot,
}: {
  snapshot: EvalsSnapshot;
  client: HarnessClient;
  onSnapshot: (snapshot: EvalsSnapshot) => void;
}) {
  const [selectedRunId, setSelectedRunId] = useState(snapshot.selectedRunId);
  const [showCreate, setShowCreate] = useState(false);
  const [experimentId, setExperimentId] = useState(snapshot.experiments[0]?.id ?? "");
  const [tier, setTier] = useState<EvalTier>("experiment");
  const [phase, setPhase] = useState<EvalPhase>("pilot");
  const [exportedReport, setExportedReport] = useState<EvalReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedRunId && snapshot.runs.some((run) => run.id === selectedRunId)) return;
    setSelectedRunId(snapshot.selectedRunId ?? snapshot.runs[0]?.id ?? null);
  }, [selectedRunId, snapshot.runs, snapshot.selectedRunId]);

  const hasActiveRuns = snapshot.runs.some((run) =>
    ["queued", "running", "canceling"].includes(run.status),
  );

  useEffect(() => {
    if (!hasActiveRuns) return;
    let active = true;
    const interval = window.setInterval(() => {
      void client
        .getEvalsSnapshot()
        .then((next) => {
          if (active) onSnapshot({ ...next, selectedRunId: selectedRunId ?? next.selectedRunId });
        })
        .catch((caught) => {
          if (active) setError(errorMessage(caught));
        });
    }, 2_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [client, hasActiveRuns, onSnapshot, selectedRunId]);

  const refresh = async (preferredRunId?: string) => {
    const next = await client.getEvalsSnapshot();
    const selected = preferredRunId ?? selectedRunId ?? next.selectedRunId;
    onSnapshot({ ...next, selectedRunId: selected });
    setSelectedRunId(selected);
  };

  const act = (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    void action()
      .catch((caught) => setError(errorMessage(caught)))
      .finally(() => setBusy(false));
  };

  const createRun = (event: FormEvent) => {
    event.preventDefault();
    if (!experimentId) return;
    act(async () => {
      const created = await client.createEvalRun({ experimentId, tier, phase });
      const started = await client.startEvalRun(created.id);
      await refresh(started.id);
      setShowCreate(false);
    });
  };

  const selectedRun = snapshot.runs.find((run) => run.id === selectedRunId);
  const report =
    exportedReport?.run_id === selectedRunId
      ? exportedReport
      : snapshot.reports.find((item) => item.run_id === selectedRunId);

  return (
    <div className="evals-page">
      <header className="area-header">
        <div>
          <span className="eyebrow">Laboratório de avaliação</span>
          <h1>Avaliações</h1>
          <p>Runs e relatórios retornados pelo serviço de avaliação.</p>
        </div>
        <div className="area-header-actions">
          <button
            className="primary-button"
            type="button"
            onClick={() => setShowCreate((current) => !current)}
            disabled={!snapshot.capabilities.eval_runner || !snapshot.experiments.length}
          >
            <Plus size={15} /> Novo run
          </button>
        </div>
      </header>

      {!snapshot.capabilities.eval_runner && (
        <div className="blocked-banner" role="status">
          <CircleStop size={15} /> blocked: eval runner indisponível
        </div>
      )}
      {error && <div className="page-error" role="alert">{error}</div>}
      {showCreate && (
        <form className="eval-run-form" onSubmit={createRun}>
          <label>
            <span>Experiment</span>
            <select value={experimentId} onChange={(event) => setExperimentId(event.target.value)}>
              {snapshot.experiments.map((experiment) => (
                <option value={experiment.id} key={experiment.id}>{experiment.id}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Tier</span>
            <select value={tier} onChange={(event) => setTier(event.target.value as EvalTier)}>
              <option value="contract">contract</option>
              <option value="model_smoke">model_smoke</option>
              <option value="experiment">experiment</option>
            </select>
          </label>
          <label>
            <span>Phase</span>
            <select value={phase} onChange={(event) => setPhase(event.target.value as EvalPhase)}>
              <option value="pilot">pilot</option>
              <option value="promotion">promotion</option>
            </select>
          </label>
          <button className="primary-button" type="submit" disabled={busy}>Criar e iniciar</button>
        </form>
      )}

      <div className="evals-layout">
        <aside className="experiment-panel" aria-label="Experimentos e runs">
          <div className="panel-title">
            <div>
              <span className="eyebrow">Registry</span>
              <h2>Experimentos</h2>
            </div>
            <FlaskConical size={16} />
          </div>
          <div className="experiment-list">
            {snapshot.experiments.map((experiment) => {
              const runs = snapshot.runs.filter(
                (run) => run.experiment_id === experiment.id,
              );
              return (
                <section className="experiment-item" key={experiment.id}>
                  <div className="experiment-heading">
                    <span className={`status-dot status-dot--${experiment.status}`} />
                    <div>
                      <h3>{experiment.id}</h3>
                      <p>{experiment.runtime_profile} · {experiment.execution_route}</p>
                    </div>
                  </div>
                  <div className="run-list">
                    {runs.map((run) => (
                      <button
                        type="button"
                        className={run.id === selectedRunId ? "active" : ""}
                        key={run.id}
                        onClick={() => {
                          setSelectedRunId(run.id);
                          setExportedReport(null);
                        }}
                      >
                        <span className={`run-state run-state--${runVisualStatus(run)}`}>
                          {run.status === "completed" ? (
                            <Check size={12} />
                          ) : run.status === "running" || run.status === "queued" ? (
                            <Activity size={12} />
                          ) : (
                            <CircleStop size={12} />
                          )}
                        </span>
                        <span>
                          <strong>{run.phase} · {run.tier}</strong>
                          <small>{run.status} · {formatTimestamp(run.updated_at)}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
              );
            })}
          </div>
        </aside>

        {selectedRun ? (
          <EvalRunView
            run={selectedRun}
            report={report}
            busy={busy}
            onStart={() =>
              act(async () => {
                await client.startEvalRun(selectedRun.id);
                await refresh(selectedRun.id);
              })
            }
            onCancel={() =>
              act(async () => {
                await client.cancelEvalRun(selectedRun.id);
                await refresh(selectedRun.id);
              })
            }
            onExport={() =>
              act(async () => {
                const exported = await client.getEvalReport(selectedRun.id);
                setExportedReport(exported);
                downloadJson(`eval-report-${selectedRun.id}.json`, exported);
              })
            }
          />
        ) : (
          <div className="empty-report">Nenhum run disponível.</div>
        )}
      </div>
    </div>
  );
}

function EvalRunView({
  run,
  report,
  busy,
  onStart,
  onCancel,
  onExport,
}: {
  run: EvalRun;
  report?: EvalReport;
  busy: boolean;
  onStart: () => void;
  onCancel: () => void;
  onExport: () => void;
}) {
  const cancelable = ["queued", "running", "canceling"].includes(run.status);
  return (
    <main className="report-panel">
      <header className="report-header">
        <div>
          <span className={`report-status report-status--${runVisualStatus(run)}`}>
            {run.status}
          </span>
          <h2>{run.experiment_id}</h2>
          <p>{run.id}</p>
        </div>
        <div className="report-actions">
          {run.status === "created" && (
            <button className="secondary-button" type="button" onClick={onStart} disabled={busy}>
              Iniciar
            </button>
          )}
          {cancelable && (
            <button className="secondary-button" type="button" onClick={onCancel} disabled={busy}>
              Cancelar
            </button>
          )}
          <button className="secondary-button" type="button" onClick={onExport} disabled={busy}>
            Exportar JSON
          </button>
        </div>
      </header>
      <dl className="run-facts">
        <div><dt>Tier</dt><dd>{run.tier}</dd></div>
        <div><dt>Phase</dt><dd>{run.phase}</dd></div>
        <div><dt>Seeds</dt><dd>{run.seeds.join(", ") || "[]"}</dd></div>
        <div><dt>Reason code</dt><dd>{run.reason_code ?? "-"}</dd></div>
        <div><dt>Created</dt><dd>{run.created_at}</dd></div>
        <div><dt>Updated</dt><dd>{run.updated_at}</dd></div>
      </dl>
      <section className="report-section raw-report">
        <div className="section-heading">
          <div>
            <span className="eyebrow">EvalReport</span>
            <h3>Payload</h3>
          </div>
          {report && <span>{report.created_at}</span>}
        </div>
        {report ? (
          <pre>{JSON.stringify(report.payload, null, 2)}</pre>
        ) : (
          <div className="interval-empty">Nenhum relatório retornado para este run.</div>
        )}
      </section>
    </main>
  );
}

function SettingsArea({
  snapshot,
  client,
}: {
  snapshot: SettingsSnapshot;
  client: HarnessClient;
}) {
  const stored = snapshot.host_config;
  const [form, setForm] = useState({
    allowed_workspace_roots: (stored?.allowed_workspace_roots ?? []).join("\n"),
    tokenizer_path: stored?.tokenizer_path ?? "",
    state_dir: stored?.state_dir ?? "",
    allowed_origins: withLoopbackPair(stored?.allowed_origins ?? []).join("\n"),
    searxng_url: stored?.searxng_url ?? "",
    ollama_url: stored?.ollama_url ?? "http://127.0.0.1:11434",
    browser_executable: stored?.browser_executable ?? "",
  });
  const [yolo, setYolo] = useState(snapshot.yolo_enabled);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const field = (name: keyof typeof form) => (event: {
    target: { value: string };
  }) => setForm((current) => ({ ...current, [name]: event.target.value }));

  const lines = (value: string) =>
    value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

  const save = (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const searxng = form.searxng_url.trim();
    const browser = form.browser_executable.trim();
    void client
      .updateHostConfig({
        allowed_workspace_roots: lines(form.allowed_workspace_roots),
        tokenizer_path: form.tokenizer_path.trim(),
        state_dir: form.state_dir.trim(),
        allowed_origins: lines(form.allowed_origins),
        searxng_url: searxng === "" ? null : searxng,
        ollama_url: form.ollama_url.trim(),
        browser_executable: browser === "" ? null : browser,
      })
      .then(() => setSaved(true))
      .catch((cause: unknown) => setError(errorMessage(cause)))
      .finally(() => setBusy(false));
  };

  const toggleYolo = (next: boolean) => {
    setYolo(next);
    setBusy(true);
    setError(null);
    void client
      .setYoloEnabled(next)
      .catch((cause: unknown) => {
        setYolo(!next);
        setError(errorMessage(cause));
      })
      .finally(() => setBusy(false));
  };

  return (
    <div className="settings-page">
      <header className="area-header">
        <div>
          <span className="eyebrow">Painel de controle</span>
          <h1>Configurações</h1>
          <p>
            {snapshot.mutable
              ? "O que está aqui é o host.json deste servidor. Salvar grava o arquivo; reiniciar aplica."
              : "Snapshot somente leitura: este servidor foi construído sem armazenamento de configuração."}
          </p>
        </div>
      </header>

      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Seções de configuração">
          <a href="#health"><Activity size={15} /> Saúde</a>
          <a href="#host"><HardDrive size={15} /> Host</a>
          <a href="#autonomy"><Zap size={15} /> Autonomia</a>
          <a href="#runtime"><Bot size={15} /> Runtime</a>
          <a href="#loop"><SettingsIcon size={15} /> Loop</a>
        </nav>

        <main className="settings-content">
          <section className="settings-card health-card" id="health">
            <SettingsHeading
              icon={Activity}
              title="Saúde"
              description="Readiness e capabilities observadas em /api/health."
            />
            <div className="health-grid">
              <HealthItem
                label="Harness"
                detail={snapshot.health.reason_code ?? "ready"}
                ready={snapshot.health.ready}
              />
              {Object.entries(snapshot.health.capabilities).map(([capability, enabled]) => (
                <HealthItem
                  key={capability}
                  label={capability}
                  detail={String(enabled)}
                  ready={enabled}
                />
              ))}
            </div>
          </section>

          <section className="settings-card" id="host">
            <SettingsHeading
              icon={HardDrive}
              title="Host"
              description="Raízes, tokenizer, origins e provedores. Fonte única: host.json."
            />
            <form className="settings-form" onSubmit={save}>
              <label className="setup-field">
                <span>Raízes de Workspace autorizadas</span>
                <textarea
                  rows={3}
                  aria-label="Raízes de Workspace autorizadas"
                  value={form.allowed_workspace_roots}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("allowed_workspace_roots")}
                />
                <small>Um caminho absoluto por linha.</small>
              </label>
              <label className="setup-field">
                <span>Origins autorizadas</span>
                <textarea
                  rows={3}
                  aria-label="Origins autorizadas"
                  value={form.allowed_origins}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("allowed_origins")}
                />
                <small>Uma origin por linha, com protocolo, host e porta.</small>
              </label>
              <label className="setup-field">
                <span>Caminho do tokenizer.json</span>
                <input
                  type="text"
                  aria-label="Caminho do tokenizer.json"
                  value={form.tokenizer_path}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("tokenizer_path")}
                />
                <small>O digest é medido do arquivo, não informado.</small>
              </label>
              <label className="setup-field">
                <span>Diretório de estado</span>
                <input
                  type="text"
                  aria-label="Diretório de estado"
                  value={form.state_dir}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("state_dir")}
                />
              </label>
              <label className="setup-field">
                <span>URL do Ollama</span>
                <input
                  type="text"
                  aria-label="URL do Ollama"
                  value={form.ollama_url}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("ollama_url")}
                />
              </label>
              <label className="setup-field">
                <span>Executável do navegador (opcional)</span>
                <input
                  type="text"
                  aria-label="Executável do navegador"
                  value={form.browser_executable}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("browser_executable")}
                />
                <small>
                  Caminho absoluto de um Chromium usado só na escalação do web_fetch. Vazio deixa
                  o harness procurar no PATH, e o comportamento passa a depender da máquina.
                </small>
              </label>
              <label className="setup-field">
                <span>Instância SearXNG (opcional)</span>
                <input
                  type="text"
                  aria-label="Instância SearXNG"
                  value={form.searxng_url}
                  disabled={!snapshot.mutable || busy}
                  onChange={field("searxng_url")}
                />
                <small>
                  Sem ela, web_search cai no DuckDuckGo, que não pede chave. Uma instância em
                  loopback é permitida só para a busca.
                </small>
              </label>
              {error && <p className="setup-error" role="alert">{error}</p>}
              {saved && (
                <p className="settings-notice" role="status">
                  host.json gravado. Reinicie o servidor para aplicar.
                </p>
              )}
              <div className="settings-form-actions">
                <button
                  className="primary-button"
                  type="submit"
                  disabled={!snapshot.mutable || busy}
                >
                  Salvar configuração
                </button>
              </div>
            </form>
          </section>

          <section className="settings-card" id="autonomy">
            <SettingsHeading
              icon={Zap}
              title="Autonomia"
              description="Quanto o modelo pode fazer sem parar para perguntar."
            />
            <label className="settings-switch">
              <input
                type="checkbox"
                checked={yolo}
                disabled={busy}
                onChange={(event) => toggleYolo(event.target.checked)}
              />
              <span>
                <strong>Modo yolo</strong>
                Nenhuma confirmação é pedida: escrita, acesso à web e até escrita derivada de
                conteúdo da web são aprovadas na hora, e os grants necessários são concedidos.
                Cada chamada continua registrada no histórico como <code>waived</code>. Vale para
                conversas novas; cada conversa pode desligar no cabeçalho do chat.
              </span>
            </label>
          </section>

          <section className="settings-card" id="runtime">
            <SettingsHeading
              icon={Bot}
              title="Runtime"
              description="Identidades ativas retornadas pelo snapshot."
            />
            <div className="field-grid">
              <ReadOnlyField label="RuntimeProfile" value={snapshot.runtime_profile} mono />
              <ReadOnlyField
                label="ExecutionRoute"
                value={snapshot.default_execution_route}
                mono
              />
            </div>
          </section>

          <section className="settings-card" id="loop">
            <SettingsHeading
              icon={SettingsIcon}
              title="Loop"
              description="Limites vindos do contrato, não editáveis pelo painel."
            />
            <div className="field-grid">
              {Object.entries(snapshot.loop).map(([key, value]) => (
                <ReadOnlyField key={key} label={key} value={displayValue(value)} mono />
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

function SettingsHeading({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: typeof Activity;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="settings-heading">
      <span className="settings-heading-icon"><Icon size={17} /></span>
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

function HealthItem({ label, detail, ready }: { label: string; detail: string; ready: boolean }) {
  return (
    <div className="health-item">
      <span className={`health-indicator health-indicator--${ready ? "ready" : "warning"}`} />
      <div>
        <strong>{label}</strong>
        <span>{detail}</span>
      </div>
      <span>{ready ? "ready" : "blocked"}</span>
    </div>
  );
}

function ReadOnlyField({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input className={mono ? "mono" : ""} value={value} readOnly disabled />
    </label>
  );
}

function reduceLiveRun(run: LiveRun, event: AgUiEvent): LiveRun {
  switch (event.type) {
    case "RUN_STARTED":
      return { ...run, status: "queued" };
    case "STEP_STARTED":
      return { ...run, status: "running", steps: run.steps + 1 };
    case "REASONING_MESSAGE_CONTENT":
      return { ...run, reasoning: run.reasoning + event.delta };
    case "TOOL_CALL_START":
      return {
        ...run,
        tools: [
          ...run.tools,
          {
            id: event.toolCallId,
            name: event.toolCallName,
            label: event.toolCallName,
            status: "running",
            arguments: {},
            summary: "",
          },
        ],
      };
    case "TOOL_CALL_ARGS":
      return {
        ...run,
        toolArguments: {
          ...run.toolArguments,
          [event.toolCallId]: (run.toolArguments[event.toolCallId] ?? "") + event.delta,
        },
      };
    case "TOOL_CALL_END":
      return {
        ...run,
        tools: run.tools.map((tool) =>
          tool.id === event.toolCallId
            ? { ...tool, arguments: parseArguments(run.toolArguments[event.toolCallId]) }
            : tool,
        ),
      };
    case "TOOL_CALL_RESULT":
      return {
        ...run,
        tools: run.tools.map((tool) =>
          tool.id === event.toolCallId ? applyLiveToolResult(tool, event.content) : tool,
        ),
      };
    case "TEXT_MESSAGE_CONTENT":
      return { ...run, content: run.content + event.delta };
    case "CUSTOM": {
      if (!isRecord(event.value)) return run;
      if (event.name === "harness.context_usage") {
        const usage = readContextUsage(event.value);
        return usage ? { ...run, contextUsage: usage } : run;
      }
      if (event.name !== "harness.turn_outcome") return run;
      const kind = event.value.outcome_kind;
      const reasonCode = event.value.reason_code;
      return typeof kind === "string" && typeof reasonCode === "string"
        ? { ...run, outcome: { kind, reasonCode } }
        : run;
    }
    case "RUN_FINISHED":
      return { ...run, status: "finished" };
    case "RUN_ERROR":
      return { ...run, status: "error", error: event.message };
    default:
      return run;
  }
}

function liveRunMessage(run: LiveRun): ChatMessage {
  return {
    id: `live-${run.runId}`,
    turnId: run.runId,
    role: "assistant",
    content: run.content || run.error || "",
    createdAt: new Date().toISOString(),
    live: true,
    ...(run.reasoning
      ? {
          reasoning: {
            summary: `Raciocínio ao vivo · ${run.steps} passo(s)`,
            content: run.reasoning,
            transient: true as const,
          },
        }
      : {}),
    ...(run.tools.length ? { tools: run.tools } : {}),
    ...(run.contextUsage ? { contextUsage: run.contextUsage } : {}),
    ...(run.outcome ? { liveOutcome: run.outcome } : {}),
  };
}

// A janela é o denominador e vem no próprio evento; sem ela não há barra que
// signifique alguma coisa, então um payload incompleto não vira meia leitura.
function readContextUsage(value: Record<string, unknown>): ContextUsage | undefined {
  const inputTokens = value.estimated_input_tokens;
  const contextWindow = value.context_window;
  const outputBudget = value.output_budget;
  const droppedTurns = value.dropped_turns;
  if (
    typeof inputTokens !== "number" ||
    typeof contextWindow !== "number" ||
    typeof outputBudget !== "number" ||
    typeof droppedTurns !== "number" ||
    contextWindow <= 0
  ) {
    return undefined;
  }
  return { inputTokens, contextWindow, outputBudget, droppedTurns };
}

function applyLiveToolResult(tool: ToolCall, content: string): ToolCall {
  try {
    const payload = JSON.parse(content) as Record<string, unknown>;
    const status = typeof payload.status === "string" ? payload.status : "";
    const error = payload.error;
    const data = payload.data;
    return {
      ...tool,
      status: status === "success" || status === "empty" ? "success" : "error",
      summary: status,
      ...(error !== null && error !== undefined ? { error: displayValue(error) } : {}),
      ...(data !== null && data !== undefined ? { result: displayValue(data) } : {}),
      ...(isRecord(data) && typeof data.diff === "string" ? { diff: data.diff } : {}),
    };
  } catch {
    return { ...tool, status: "error", error: content, summary: "invalid_tool_result" };
  }
}

function parseArguments(value: string | undefined): Record<string, unknown> {
  if (!value) return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : { value: parsed };
  } catch {
    return { raw: value };
  }
}

function runVisualStatus(run: EvalRun) {
  if (run.status === "completed") return "passed";
  if (run.status === "failed" || run.status === "canceled") return "failed";
  if (run.status === "blocked") return "blocked";
  return "running";
}

function formatTimestamp(value: string) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function displayValue(value: unknown) {
  if (typeof value === "string") return value;
  if (value === null) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Falha inesperada no client live.";
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() ?? `run-${Date.now()}-${Math.random()}`;
}

function isRecord(value: unknown): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function downloadJson(fileName: string, value: unknown) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}
