import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MockHarnessClient } from "./client/MockHarnessClient";
import { App } from "./App";

describe("App", () => {
  it("botão de estado vazio cria a primeira conversa direto, sem formulário", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    await client.deleteConversation("chat-128");
    await client.deleteConversation("chat-127");
    const create = vi.spyOn(client, "createConversation");
    render(<App client={client} />);

    const heading = await screen.findByRole("heading", { name: "Crie a primeira conversa" });
    const emptyState = heading.closest(".empty-transcript");
    if (!emptyState) throw new Error("empty state container not found");
    await user.click(within(emptyState as HTMLElement).getByRole("button", { name: "Nova conversa" }));

    await waitFor(() =>
      expect(create).toHaveBeenCalledWith("/workspaces/meu-harness", "Nova conversa"),
    );
    expect(screen.queryByRole("combobox", { name: "Raiz autorizada" })).toBeNull();
  });

  it("navega entre as três áreas públicas", async () => {
    const user = userEvent.setup();
    render(<App client={new MockHarnessClient()} />);

    expect(await screen.findByRole("heading", { name: "Refinar retenção por conversa" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Avaliações" }));
    expect(screen.getByRole("heading", { name: "Avaliações" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Configurações" }));
    expect(screen.getByRole("heading", { name: "Configurações" })).toBeInTheDocument();
    expect(screen.getByDisplayValue("gemma4_e4b_qat_ollama")).toBeInTheDocument();
  });

  it("chama o client para seleção, criação, fila, grants e feedback", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    const select = vi.spyOn(client, "selectConversation");
    const create = vi.spyOn(client, "createConversation");
    const edit = vi.spyOn(client, "editPendingRequest");
    const cancel = vi.spyOn(client, "cancelPendingRequest");
    const revoke = vi.spyOn(client, "revokeGrant");
    const addGrant = vi.spyOn(client, "addGrant");
    const feedback = vi.spyOn(client, "addFeedback");
    render(<App client={client} />);

    expect(await screen.findByRole("heading", { name: "Refinar retenção por conversa" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Editar solicitação 1" }));
    const editor = screen.getByRole("textbox", { name: "Editar solicitação 1" });
    await user.clear(editor);
    await user.type(editor, "Fila editada pela API");
    await user.click(screen.getByRole("button", { name: "Salvar solicitação 1" }));
    await waitFor(() => expect(edit).toHaveBeenCalledWith("chat-128", "pending-1", "Fila editada pela API"));

    await user.click(screen.getByRole("button", { name: "Revogar Escrita" }));
    await waitFor(() => expect(revoke).toHaveBeenCalledWith("chat-128", "grant-write"));
    await user.click(await screen.findByRole("button", { name: "Ativar Escrita" }));
    await waitFor(() => expect(addGrant).toHaveBeenCalledWith("chat-128", "WriteGrant"));

    await user.click(screen.getByRole("button", { name: /Funcionou/ }));
    await waitFor(() => expect(feedback).toHaveBeenCalledWith("chat-128", 1, "turn-1"));

    await user.click(screen.getByRole("button", { name: "Cancelar solicitação 1" }));
    await waitFor(() => expect(cancel).toHaveBeenCalledWith("chat-128", "pending-1"));

    await user.click(screen.getByRole("button", { name: /Verificação de página/ }));
    await waitFor(() => expect(select).toHaveBeenCalledWith("chat-127"));

    await user.click(screen.getByRole("button", { name: "Nova conversa" }));
    const name = screen.getByRole("textbox", { name: "Nome" });
    await user.clear(name);
    await user.type(name, "Conversation criada");
    await user.click(screen.getByRole("button", { name: "Criar" }));
    await waitFor(() =>
      expect(create).toHaveBeenCalledWith("/workspaces/meu-harness", "Conversation criada"),
    );
  });

  it("aprova a escrita com dado da web só depois da decisão do Operator", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    client.seedConfirmation({
      id: "turn-1-confirmation-2",
      conversation_id: "chat-128",
      turn_id: "turn-1",
      step_sequence: 2,
      reason_code: "web_taint_confirmation_required",
      tool_calls: [
        {
          id: "write-1",
          name: "write_file",
          arguments: { file_path: "relatorio.md", content: "vindo da web" },
          effects: ["workspace_write"],
        },
      ],
      previews: [
        {
          tool_call_id: "write-1",
          path: "relatorio.md",
          kind: "create",
          diff: "@@ -0,0 +1 @@\n+vindo da web",
          truncated: false,
        },
      ],
    });
    const resolve = vi.spyOn(client, "resolveConfirmation");
    render(<App client={client} />);

    const dialog = await screen.findByRole("dialog");
    expect(
      await screen.findByRole("heading", { name: "Escrita com dado da web" }),
    ).toBeInTheDocument();
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("write_file")).toBeInTheDocument();
    expect(screen.getByText("+vindo da web")).toBeInTheDocument();
    // A write the web asked for cannot be waived away.
    expect(screen.queryByLabelText("Não perguntar mais nesta conversa")).toBeNull();

    // Escape must not answer for the Operator: the Turn is parked on this.
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Aprovar escrita" }));

    await waitFor(() =>
      expect(resolve).toHaveBeenCalledWith("chat-128", "turn-1-confirmation-2", true, false),
    );
    await waitFor(() =>
      expect(screen.queryByRole("heading", { name: "Escrita com dado da web" })).toBeNull(),
    );
  });

  it("uma escrita comum pode ser dispensada, e a dispensa fica revogável", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    client.seedConfirmation({
      id: "turn-2-confirmation-1",
      conversation_id: "chat-128",
      turn_id: "turn-2",
      step_sequence: 1,
      reason_code: "write_confirmation_required",
      tool_calls: [
        {
          id: "write-2",
          name: "write_file",
          arguments: { file_path: "notas.md", content: "sem web" },
        },
      ],
      previews: [
        {
          tool_call_id: "write-2",
          path: "notas.md",
          kind: "create",
          diff: "@@ -0,0 +1 @@\n+sem web",
          truncated: false,
        },
      ],
    });
    const resolve = vi.spyOn(client, "resolveConfirmation");
    const revoke = vi.spyOn(client, "revokeConfirmationWaiver");
    render(<App client={client} />);

    expect(await screen.findByRole("heading", { name: "Confirmar escrita" })).toBeInTheDocument();
    await user.click(screen.getByLabelText("Não perguntar mais nesta conversa"));
    await user.click(screen.getByRole("button", { name: "Aprovar escrita" }));

    await waitFor(() =>
      expect(resolve).toHaveBeenCalledWith("chat-128", "turn-2-confirmation-1", true, true),
    );

    const chip = await screen.findByRole("button", { name: /Escritas sem confirmação/ });
    await user.click(chip);
    await waitFor(() =>
      expect(revoke).toHaveBeenCalledWith("chat-128", "workspace_write"),
    );
  });

  it("liga o yolo em um clique e desliga só na conversa aberta", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    const enable = vi.spyOn(client, "setYoloEnabled");
    const perConversation = vi.spyOn(client, "setConversationYolo");
    render(<App client={client} />);

    await user.click(await screen.findByRole("button", { name: /Yolo desligado/ }));
    await waitFor(() => expect(enable).toHaveBeenCalledWith(true));
    expect(perConversation).toHaveBeenCalledWith("chat-128", false);

    await user.click(await screen.findByRole("button", { name: /Yolo ligado/ }));
    await waitFor(() => expect(perConversation).toHaveBeenCalledWith("chat-128", true));
    expect(enable).toHaveBeenCalledTimes(1);
  });

  it("grava a configuração do host pelo painel e avisa que exige reinício", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    const update = vi.spyOn(client, "updateHostConfig");
    render(<App client={client} />);

    await user.click(await screen.findByRole("button", { name: "Configurações" }));
    const searxng = await screen.findByLabelText("Instância SearXNG");
    await user.type(searxng, "http://127.0.0.1:8080/search");
    await user.click(screen.getByRole("button", { name: "Salvar configuração" }));

    await waitFor(() =>
      expect(update).toHaveBeenCalledWith(
        expect.objectContaining({
          allowed_workspace_roots: ["/workspaces/meu-harness"],
          searxng_url: "http://127.0.0.1:8080/search",
          ollama_url: "http://127.0.0.1:11434",
        }),
      ),
    );
    expect(await screen.findByText(/Reinicie o servidor para aplicar/)).toBeInTheDocument();
  });

  it("exige o setup antes de qualquer outra área e pede reinício ao concluir", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    client.seedSetupRequired();
    const complete = vi.spyOn(client, "completeSetup");
    const chat = vi.spyOn(client, "getChatSnapshot");
    render(<App client={client} />);

    expect(
      await screen.findByRole("heading", { name: "Configurar este host" }),
    ).toBeInTheDocument();
    expect(chat).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText("Token de setup"), "token-do-stderr");
    await user.type(
      screen.getByLabelText("Raízes de Workspace autorizadas"),
      "/workspaces/meu-harness",
    );
    const stateDirField = screen.getByLabelText("Diretório de estado");
    await user.clear(stateDirField);
    await user.type(stateDirField, "/var/lib/meu-harness");
    const tokenizerPathField = screen.getByLabelText("Caminho do tokenizer.json");
    await user.clear(tokenizerPathField);
    await user.type(tokenizerPathField, "/var/lib/meu-harness/tokenizer.json");
    const originsField = screen.getByLabelText("Origins autorizadas");
    // Loopback tem duas grafias e o navegador manda a que foi digitada na barra:
    // o formulário sugere o par para o painel não virar 403 pela outra.
    expect(originsField).toHaveValue(
      `${window.location.origin}\nhttp://127.0.0.1:${window.location.port}`,
    );
    await user.clear(originsField);
    await user.type(originsField, "http://127.0.0.1:8765");
    await user.click(screen.getByRole("button", { name: "Concluir setup" }));

    await waitFor(() =>
      expect(complete).toHaveBeenCalledWith("token-do-stderr", {
        allowed_workspace_roots: ["/workspaces/meu-harness"],
        state_dir: "/var/lib/meu-harness",
        tokenizer_path: "/var/lib/meu-harness/tokenizer.json",
        allowed_origins: ["http://127.0.0.1:8765"],
        searxng_url: null,
      }),
    );
    expect(
      await screen.findByRole("heading", { name: "Configuração gravada" }),
    ).toBeInTheDocument();
  });

  it("pede a senha antes de qualquer área quando o host exige autenticação", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    client.seedAuthenticationRequired("operator-password-1");
    const chat = vi.spyOn(client, "getChatSnapshot");
    const login = vi.spyOn(client, "login");
    render(<App client={client} />);

    expect(await screen.findByRole("heading", { name: "Entrar no Harness" })).toBeInTheDocument();
    expect(chat).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText("Senha de Operator"), "wrong-password");
    await user.click(screen.getByRole("button", { name: "Entrar" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Senha de Operator"), "operator-password-1");
    await user.click(screen.getByRole("button", { name: "Entrar" }));

    await waitFor(() => expect(login).toHaveBeenCalledWith("operator-password-1"));
    expect(
      await screen.findByRole("heading", { name: "Refinar retenção por conversa" }),
    ).toBeInTheDocument();
  });

  it("envia RunAgentInput, chama Stop e aborta o stream no unmount", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    let streamSignal: AbortSignal | undefined;
    const stream = vi.spyOn(client, "streamAgent").mockImplementation(
      async (input, onEvent, signal) => {
        streamSignal = signal;
        onEvent({ type: "RUN_STARTED", threadId: input.threadId, runId: input.runId! });
        await new Promise<void>((_resolve, reject) => {
          signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        });
        return { type: "RUN_FINISHED", threadId: input.threadId, runId: input.runId! };
      },
    );
    const stop = vi.spyOn(client, "stop");
    const view = render(<App client={client} />);

    await screen.findByRole("heading", { name: "Refinar retenção por conversa" });
    await user.type(screen.getByLabelText("Solicitação para o Harness"), "Nova solicitação");
    await user.click(screen.getByRole("button", { name: "Enviar solicitação" }));

    await waitFor(() => expect(stream).toHaveBeenCalled());
    expect(stream.mock.calls[0]?.[0]).toMatchObject({
      threadId: "chat-128",
      messages: [{ role: "user", content: "Nova solicitação" }],
    });
    await user.click(screen.getByRole("button", { name: /Parar/ }));
    await waitFor(() => expect(stop).toHaveBeenCalledWith("chat-128"));

    view.unmount();
    expect(streamSignal?.aborted).toBe(true);
  });

  it("anexa uma imagem, mostra a miniatura e manda o caminho para o modelo", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    const attach = vi.spyOn(client, "attachImage");
    const stream = vi
      .spyOn(client, "streamAgent")
      .mockImplementation(async (input) => ({
        type: "RUN_FINISHED",
        threadId: input.threadId,
        runId: input.runId!,
      }));
    // jsdom não implementa object URLs; a miniatura só precisa de uma string.
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
    render(<App client={client} />);

    await screen.findByRole("heading", { name: "Refinar retenção por conversa" });
    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "print.png", {
      type: "image/png",
    });
    await user.upload(screen.getByLabelText("Escolher imagem"), png);

    expect(await screen.findByRole("img", { name: "print.png" })).toBeInTheDocument();
    expect(attach).toHaveBeenCalledWith("chat-128", png);
    // Sem texto, a imagem sozinha já é uma pergunta.
    await user.click(screen.getByRole("button", { name: "Enviar solicitação" }));

    await waitFor(() => expect(stream).toHaveBeenCalled());
    expect(stream.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        {
          role: "user",
          content: "O que tem nesta imagem?\n\nImagem anexada: anexos/print.png",
        },
      ],
    });
    expect(screen.queryByRole("img", { name: "print.png" })).not.toBeInTheDocument();
  });

  it("mostra o consumo da janela de contexto enquanto o turno roda", async () => {
    const user = userEvent.setup();
    const client = new MockHarnessClient();
    vi.spyOn(client, "streamAgent").mockImplementation(async (input, onEvent, signal) => {
      onEvent({ type: "RUN_STARTED", threadId: input.threadId, runId: input.runId! });
      onEvent({ type: "STEP_STARTED", stepName: "step-1" });
      onEvent({
        type: "CUSTOM",
        name: "harness.context_usage",
        value: {
          estimated_input_tokens: 13107,
          context_window: 65536,
          output_budget: 8192,
          dropped_turns: 2,
        },
      });
      // O stream fica aberto: a barra descreve a execução ao vivo, e um turno já
      // encerrado não tem janela para mostrar.
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
      return { type: "RUN_FINISHED", threadId: input.threadId, runId: input.runId! };
    });
    const view = render(<App client={client} />);

    await screen.findByRole("heading", { name: "Refinar retenção por conversa" });
    await user.type(screen.getByLabelText("Solicitação para o Harness"), "Nova solicitação");
    await user.click(screen.getByRole("button", { name: "Enviar solicitação" }));

    const bar = await screen.findByRole("progressbar", { name: "Uso da janela de contexto" });
    expect(bar.getAttribute("aria-valuenow")).toBe("13107");
    // A barra mede contra o utilizável: janela de 65536 menos os 8192 da resposta.
    expect(bar.getAttribute("aria-valuemax")).toBe("57344");
    expect(screen.getByText("2 turnos descartados")).toBeInTheDocument();

    view.unmount();
  });
});
