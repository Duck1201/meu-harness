import { describe, expect, it, vi } from "vitest";
import type { AgUiEvent, RunAgentInput } from "../types";
import {
  FetchHarnessClient,
  HarnessApiError,
} from "./FetchHarnessClient";

const json = (value: unknown, init?: ResponseInit) =>
  new Response(JSON.stringify(value), {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

describe("FetchHarnessClient", () => {
  it("preserva status, código e detalhe de erros estruturados", async () => {
    const fetchMock = vi.fn(async () =>
      json(
        {
          error: {
            code: "workspace_root_not_allowed",
            message: "The workspace root is not in the server allowlist.",
          },
        },
        { status: 403 },
      ),
    ) as unknown as typeof fetch;
    const client = new FetchHarnessClient("/api", fetchMock);

    const error = await client.getHealth().catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(HarnessApiError);
    expect(error).toMatchObject({
      status: 403,
      code: "workspace_root_not_allowed",
      message: "The workspace root is not in the server allowlist.",
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/health", undefined);
  });

  it("anexa a imagem por multipart na conversa e devolve o caminho no Workspace", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify({ path: "anexos/20260929-print.png" }), {
          status: 201,
          headers: { "Content-Type": "application/json" },
        }),
    );
    const client = new FetchHarnessClient("/api", fetchMock);
    const file = new File([new Uint8Array([0x89])], "print.png", { type: "image/png" });

    await expect(client.attachImage("chat 1", file)).resolves.toBe("anexos/20260929-print.png");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/conversations/chat%201/attachments");
    expect(init?.method).toBe("POST");
    expect((init?.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("agrupa Conversations e projeta CanonicalHistory sem inventar reasoning ou métricas", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/workspaces") {
        return json({ workspaces: [{ id: "workspace-1", root: "/srv/project" }] });
      }
      if (url === "/api/conversations") {
        return json({
          conversations: [
            {
              id: "conversation-1",
              workspace_id: "workspace-1",
              name: "Contrato real",
              created_at: "2026-08-10T10:00:00Z",
              updated_at: "2026-08-10T10:01:00Z",
              last_active_at: "2026-08-10T10:01:00Z",
              archived_at: null,
            },
          ],
        });
      }
      if (url === "/api/ui/settings") {
        return json({
          mutable: false,
          default_execution_route: "local",
          runtime_profile: "ollama-profile",
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
      if (url === "/api/ui/corpora") {
        return json({
          available: true,
          embedding_model: "bge-m3:latest",
          accepted_extensions: [".txt", ".md", ".html", ".pdf"],
          corpora: [],
          jobs: [],
        });
      }
      if (url === "/api/conversations/conversation-1/grants") {
        return json({
          grants: [
            {
              id: "grant-root",
              conversation_id: "conversation-1",
              permission: "WorkspaceRootGrant",
              scope: "/srv/project",
              granted_at: "2026-08-10T10:00:00Z",
              expires_at: null,
            },
          ],
        });
      }
      if (url === "/api/ui/chat?conversation_id=conversation-1") {
        return json({
          conversation: {
            id: "conversation-1",
            workspace_id: "workspace-1",
            name: "Contrato real",
            created_at: "2026-08-10T10:00:00Z",
            updated_at: "2026-08-10T10:01:00Z",
            last_active_at: "2026-08-10T10:01:00Z",
            archived_at: null,
          },
          pending_requests: [],
          active_turn: null,
          turns: [
            {
              id: "turn-1",
              conversation_id: "conversation-1",
              request_id: "request-1",
              status: "finished",
              started_at: "2026-08-10T10:00:00Z",
              ended_at: "2026-08-10T10:01:00Z",
              terminal_outcome: {
                kind: "completed",
                reason_code: "final_response",
                recorded_at: "2026-08-10T10:01:00Z",
                detail: null,
              },
            },
          ],
          history: [
            {
              id: "history-user",
              sequence: 1,
              conversation_id: "conversation-1",
              turn_id: "turn-1",
              kind: "user_message",
              payload: { content: "Leia o arquivo" },
              created_at: "2026-08-10T10:00:00Z",
            },
            {
              id: "history-attempt",
              sequence: 2,
              conversation_id: "conversation-1",
              turn_id: "turn-1",
              kind: "model_attempt",
              payload: {
                content: null,
                tool_calls: [
                  {
                    id: "call-1",
                    name: "read_file",
                    arguments: { file_path: "README.md" },
                    idempotency_key: null,
                  },
                ],
              },
              created_at: "2026-08-10T10:00:01Z",
            },
            {
              id: "history-result",
              sequence: 3,
              conversation_id: "conversation-1",
              turn_id: "turn-1",
              kind: "tool_result",
              payload: {
                tool_call_id: "call-1",
                tool_name: "read_file",
                status: "success",
                retryable: false,
                data: "conteúdo real",
                error: null,
                meta: {},
              },
              created_at: "2026-08-10T10:00:02Z",
            },
            {
              id: "history-final",
              sequence: 4,
              conversation_id: "conversation-1",
              turn_id: "turn-1",
              kind: "final_response",
              payload: { content: "Resposta persistida" },
              created_at: "2026-08-10T10:01:00Z",
            },
          ],
          feedback: [],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }) as unknown as typeof fetch;
    const client = new FetchHarnessClient("/api", fetchMock);

    const snapshot = await client.getChatSnapshot("conversation-1");

    expect(snapshot.workspaces[0]).toMatchObject({
      name: "project",
      root: "/srv/project",
      conversations: [{ id: "conversation-1", title: "Contrato real" }],
    });
    expect(snapshot.messages).toHaveLength(2);
    expect(snapshot.messages[1]).toMatchObject({
      role: "assistant",
      content: "Resposta persistida",
      terminalOutcome: { kind: "completed", reason_code: "final_response" },
      tools: [
        {
          id: "call-1",
          name: "read_file",
          status: "success",
          arguments: { file_path: "README.md" },
          result: "conteúdo real",
        },
      ],
    });
    expect(snapshot.messages[1]?.reasoning).toBeUndefined();
    expect(snapshot.messages[1]?.metrics).toBeUndefined();
  });

  it("parseia eventos SSE incrementalmente mesmo com JSON e delimitadores fragmentados", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      'data: {"type":"RUN_STAR',
      'TED","threadId":"thread-1","runId":"run-1"}\r',
      '\n\r\ndata: {"type":"TEXT_MESSAGE_CONTENT","messageId":"message-1",',
      '"delta":"olá"}\n\ndata: {"type":"CUSTOM","name":"harness.turn_outcome",',
      '"value":{"outcome_kind":"completed","reason_code":"final_response"}}\n\ndata: {"type":"RUN_FINISHED","threadId":"thread-1",',
      '"runId":"run-1"}\n\n',
    ];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    const fetchMock = vi.fn(async () =>
      new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    ) as unknown as typeof fetch;
    const client = new FetchHarnessClient("/api", fetchMock);
    const events: AgUiEvent[] = [];
    const input: RunAgentInput = {
      threadId: "thread-1",
      runId: "run-1",
      messages: [{ id: "user-1", role: "user", content: "olá" }],
    };

    const terminal = await client.streamAgent(input, (event) => events.push(event));

    expect(events.map((event) => event.type)).toEqual([
      "RUN_STARTED",
      "TEXT_MESSAGE_CONTENT",
      "CUSTOM",
      "RUN_FINISHED",
    ]);
    expect(events[1]).toMatchObject({ delta: "olá" });
    expect(terminal).toEqual({
      type: "RUN_FINISHED",
      threadId: "thread-1",
      runId: "run-1",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/agent",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(input),
        headers: expect.objectContaining({ Accept: "text/event-stream" }),
      }),
    );
  });
});
