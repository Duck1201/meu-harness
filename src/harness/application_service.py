import asyncio
import contextlib
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from .agent_engine import AgentEngine, ResponseMarkup
from .brave_browser import BraveBrowserCapability, BraveEgressGuard
from .composite_tools import CompositeToolExecutor
from .config import CorpusScraperConfig, HarnessConfig, ToolRegistryConfig, VisionConfig
from .context_builder import ContextBuilder
from .conversation_store import ConversationStore, NotFoundError
from .corpus_browser import ScraplingBrowserRenderer
from .corpus_ocr import PageOcr, PdfPageRenderer
from .corpus_scraper import PageRenderer, Scraper
from .corpus_service import (
    CorpusIngestionService,
    CorpusLibrary,
    CorpusLibraryError,
    CorpusRetriever,
    IngestionJob,
)
from .corpus_store import CorpusStoreError
from .corpus_tools import CorpusToolExecutor, granted_corpus_id
from .domain import (
    CORPUS_EFFECT,
    CORPUS_GRANT,
    LOCAL_INFERENCE_EFFECT,
    MUTATION_EFFECT,
    WAIVABLE_CONFIRMATION_REASONS,
    YOLO_CONFIRMATION_REASONS,
    CanonicalHistoryEntry,
    CanonicalHistoryEntryKind,
    Conversation,
    Corpus,
    Document,
    Feedback,
    Grant,
    JsonValue,
    PendingRequest,
    ToolCall,
    ToolResult,
    TurnStatus,
    waived_reason_code,
)
from .evals import (
    BenchmarkLease,
    BrowserBenchCaseRunner,
    CompositeCaseRunner,
    ContractCaseRunner,
    EvalService,
    EvalStore,
    EvalTier,
    ModelCaseRunner,
    PortugueseDetector,
    RegressionDraft,
    load_eval_catalog,
)
from .local_tools import RegistryToolExecutor
from .observability_store import ObservabilityStore
from .ports import (
    AgentEvent,
    AgentEventKind,
    ConfirmationDecision,
    ConfirmationPreview,
    ConfirmationRequest,
    CorpusAnswerJudge,
    EmbeddingRuntime,
    EngineReadiness,
    EventSink,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelRuntime,
    ModelRuntimeError,
    StopSignal,
    TextTokenCounter,
    TokenEstimator,
    ToolBatchPreflight,
    ToolExecutor,
    ToolExecutorFactory,
    ToolSchema,
    VisionRuntime,
)
from .system_prompt import build_system_prompt
from .vision_tools import VisionToolExecutor, image_format
from .web_tools import BrowserCapability, BrowserEgressGuard, WebToolExecutor
from .workspace_coordinator import WorkspaceCoordinator


def _browser_renderer(config: CorpusScraperConfig) -> PageRenderer | None:
    escalation = config.html_crawl.browser_escalation
    if escalation.mode != "symptom":
        return None
    return ScraplingBrowserRenderer(
        stealth=escalation.stealth, timeout_milliseconds=escalation.timeout_milliseconds
    )


# Onde as imagens enviadas pela UI entram no Workspace: uma pasta visível, porque
# o Operator e o modelo precisam achar o arquivo pelo mesmo caminho.
_ATTACHMENTS_DIRECTORY = "anexos"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ApplicationRuntime(ModelRuntime, Protocol):
    async def verify_profile(self) -> "RuntimeVerification": ...

    async def aclose(self) -> None: ...


class RuntimeVerification(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def reason_code(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class WorkspaceInfo:
    id: str
    root: Path


class ApplicationServiceError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ApplicationService:
    def __init__(
        self,
        *,
        store: ConversationStore,
        observability_store: ObservabilityStore,
        config: HarnessConfig,
        runtime: ApplicationRuntime,
        estimator: TokenEstimator,
        allowed_workspace_roots: Sequence[str | Path],
        search_endpoint: str | None = None,
        browser_executable: str | Path | None = None,
        operator_notes: str = "",
        benchmark_lease: BenchmarkLease | None = None,
        eval_service: EvalService | None = None,
        browser_capability: BrowserCapability | None = None,
        browser_egress_guard: BrowserEgressGuard | None = None,
        corpus_directory: str | Path | None = None,
        embedder: EmbeddingRuntime | None = None,
        answer_judge: CorpusAnswerJudge | None = None,
        vision_runtime: VisionRuntime | None = None,
        page_renderer: PdfPageRenderer | None = None,
        page_ocr: PageOcr | None = None,
    ) -> None:
        roots: list[Path] = []
        for candidate in allowed_workspace_roots:
            path = Path(candidate)
            if not path.is_absolute():
                raise ValueError("allowed workspace roots must be absolute")
            resolved = path.resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError("allowed workspace roots must be directories")
            if resolved not in roots:
                roots.append(resolved)
        self.store = store
        self.observability_store = observability_store
        self.config = config
        self.runtime = runtime
        self.estimator = estimator
        # Default aqui e não em build_system_prompt: nesta fronteira "" é ausência
        # declarada, enquanto lá dentro um default deixaria o corpus montar um
        # prompt diferente do de produção sem ninguém notar.
        self._operator_notes = operator_notes
        if search_endpoint is not None and not search_endpoint.strip():
            raise ValueError("search_endpoint must not be blank")
        normalized_search_endpoint = (
            search_endpoint.strip() if search_endpoint is not None else None
        )
        self._allowed_roots = tuple(roots)
        if (
            benchmark_lease is not None
            and eval_service is not None
            and benchmark_lease is not eval_service.lease
        ):
            raise ValueError("eval_service and ApplicationService must share BenchmarkLease")
        self.benchmark_lease = benchmark_lease or (
            eval_service.lease if eval_service is not None else BenchmarkLease()
        )
        self.eval_service = eval_service or _default_eval_service(
            store=store,
            lease=self.benchmark_lease,
            registry=config.tool_registry,
            config=config,
            runtime=runtime,
            estimator=estimator,
            operator_notes=operator_notes,
            browser_executable=browser_executable,
        )
        if Path(self.eval_service.store.database).resolve() == Path(store.database).resolve():
            raise ValueError("EvalStore must be separate from ConversationStore")
        self._workspaces_by_root: dict[Path, WorkspaceInfo] = {}
        self._runtime_readiness = EngineReadiness(ready=False, reason_code="runtime_not_verified")
        self._initialized = False
        self._shutting_down = False
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._stop_signals: dict[str, _CooperativeStopSignal] = {}
        self._worker_lock = asyncio.Lock()
        # Conversas sendo apagadas: o worker cancelado não pode se recriar para a
        # fila de uma conversa que está deixando de existir.
        self._retiring: set[str] = set()
        self._workspace_coordinator = WorkspaceCoordinator(
            store, config.tool_registry.effects_by_tool
        )
        if browser_capability is None or browser_egress_guard is None:
            brave_guard = BraveEgressGuard(executable=browser_executable)
            browser_capability = browser_capability or BraveBrowserCapability(guard=brave_guard)
            browser_egress_guard = browser_egress_guard or brave_guard
        # Sem modelo de embedding declarado no perfil, ou sem diretório, não há
        # Corpus: a aba diz isso e a tool recusa, em vez de o harness inventar um
        # acervo vazio que responderia "não encontrei" para sempre.
        self.embedder = embedder
        counter = estimator if isinstance(estimator, TextTokenCounter) else None
        # Uma condição só para tudo que é Corpus: sem embedder ou sem contador de
        # tokens não há como indexar nem buscar, e uma biblioteca que só lista
        # arquivos seria uma disponibilidade mentirosa na aba.
        self.corpus_library = (
            _corpus_library(config, corpus_directory)
            if embedder is not None and counter is not None
            else None
        )
        self._corpus_retriever = (
            CorpusRetriever(
                library=self.corpus_library,
                embedder=embedder,
                counter=counter,
                config=config.corpus,
                judge=answer_judge if config.corpus.answer_judge.mode == "advisory" else None,
            )
            if self.corpus_library is not None and embedder is not None and counter is not None
            else None
        )
        self.corpus_ingestion = (
            CorpusIngestionService(
                library=self.corpus_library,
                embedder=embedder,
                counter=counter,
                config=config.corpus,
                scraper=Scraper(
                    config=config.corpus.scraper,
                    renderer=_browser_renderer(config.corpus.scraper),
                ),
                recorder=observability_store,
                page_renderer=page_renderer,
                page_ocr=page_ocr,
            )
            if self.corpus_library is not None and embedder is not None and counter is not None
            else None
        )
        self._tool_executor_factory: ToolExecutorFactory = _ConversationToolExecutorFactory(
            store=store,
            registry=config.tool_registry,
            workspace_root=self.workspace_root,
            coordinator=self._workspace_coordinator,
            search_endpoint=normalized_search_endpoint,
            max_read_bytes=config.context.max_tool_read_bytes,
            max_search_bytes=config.context.max_tool_search_bytes,
            browser_capability=browser_capability,
            browser_egress_guard=browser_egress_guard,
            corpus_retriever=self._corpus_retriever,
            vision_runtime=vision_runtime,
            vision_config=config.vision,
        )
        self._vision_offered = vision_runtime is not None
        self._event_bus = _LiveEventBus()
        self._event_sink = _ServiceEventSink(self._event_bus, observability_store)
        self._confirmation_gate = OperatorConfirmationGate(self._event_sink, self.store)

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.store.initialize()
        await self.observability_store.initialize()
        await self.eval_service.initialize()
        await self.store.recover_stale_active_turns()
        for root in self._allowed_roots:
            revision = await self.store.create_workspace(str(root))
            self._workspaces_by_root[root] = WorkspaceInfo(
                id=revision.workspace_id,
                root=root,
            )
        try:
            verification = await self.runtime.verify_profile()
            self._runtime_readiness = EngineReadiness(
                ready=verification.ready,
                reason_code=verification.reason_code,
            )
        except Exception:
            self._runtime_readiness = EngineReadiness(
                ready=False, reason_code="runtime_verification_failed"
            )
        self._initialized = True
        for conversation in await self.store.list_conversations(include_archived=True):
            if await self.store.list_pending_requests(conversation.id):
                await self._ensure_worker(conversation.id)

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = tuple(self._workers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.eval_service.shutdown()
        if self.corpus_ingestion is not None:
            await self.corpus_ingestion.shutdown()
        await self.runtime.aclose()

    @property
    def readiness(self) -> EngineReadiness:
        if not self._runtime_readiness.ready:
            return self._runtime_readiness
        if not self.estimator.validated:
            readiness = getattr(self.estimator, "readiness", None)
            if isinstance(readiness, EngineReadiness):
                return readiness
            return EngineReadiness(ready=False, reason_code="token_estimator_not_validated")
        return EngineReadiness(ready=True)

    def list_workspaces(self) -> tuple[WorkspaceInfo, ...]:
        return tuple(self._workspaces_by_root.values())

    async def create_conversation(
        self, workspace_root: str, *, name: str = "New conversation"
    ) -> Conversation:
        candidate = Path(workspace_root)
        if not candidate.is_absolute():
            raise ApplicationServiceError(
                "workspace_root_must_be_absolute",
                "Workspace roots supplied to the API must be absolute.",
                status_code=422,
            )
        try:
            root = candidate.resolve(strict=True)
        except OSError as error:
            raise ApplicationServiceError(
                "workspace_root_not_allowed",
                "The workspace root is not in the server allowlist.",
                status_code=403,
            ) from error
        workspace = self._workspaces_by_root.get(root)
        if workspace is None:
            raise ApplicationServiceError(
                "workspace_root_not_allowed",
                "The workspace root is not in the server allowlist.",
                status_code=403,
            )
        conversation = await self.store.create_conversation(workspace.id, name=name)
        await self.store.grant(
            conversation.id,
            permission="WorkspaceRootGrant",
            scope=str(root),
        )
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation:
        return await self.store.get_conversation(conversation_id)

    async def attach_image(self, conversation_id: str, filename: str, data: bytes) -> str:
        """Grava uma imagem enviada pelo Operator no Workspace e devolve o caminho.

        O modelo de chat não recebe bytes: a imagem vira um arquivo do Workspace,
        e o `describe_image` a lê pelo caminho, com o mesmo gate de sempre. O
        formato é o do magic byte, não o da extensão, e só entra o que a tool de
        visão aceita — um anexo que ela recusaria é erro agora, não no meio do Turn.
        """
        conversation = await self.get_conversation(conversation_id)
        root = self.workspace_root(conversation.workspace_id)
        vision = self.config.vision
        if len(data) > vision.max_image_bytes:
            raise ApplicationServiceError(
                "image_too_large",
                f"A imagem tem {len(data)} bytes e o limite é {vision.max_image_bytes}.",
                status_code=413,
            )
        detected = image_format(data[:16])
        if detected is None or detected not in vision.accepted_formats:
            raise ApplicationServiceError(
                "image_format_not_accepted",
                "Só PNG, JPEG e WebP são aceitos: são os formatos que a tool de visão lê.",
                status_code=415,
            )
        folder = root / _ATTACHMENTS_DIRECTORY
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ApplicationServiceError(
                "attachments_folder_unavailable",
                f"`{_ATTACHMENTS_DIRECTORY}` no Workspace não é uma pasta comum.",
                status_code=409,
            )
        folder.mkdir(exist_ok=True)
        stem = _SAFE_NAME.sub("-", Path(filename).stem).strip("-.")[:60] or "imagem"
        extension = "jpg" if detected == "jpeg" else detected
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{stem}.{extension}"
        descriptor, temp_name = tempfile.mkstemp(dir=folder, prefix=".upload-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temp_name, folder / name)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return f"{_ATTACHMENTS_DIRECTORY}/{name}"

    async def list_conversations(self, *, include_archived: bool = False) -> list[Conversation]:
        return await self.store.list_conversations(include_archived=include_archived)

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        name: str | None = None,
        archived: bool | None = None,
        yolo_disabled: bool | None = None,
    ) -> Conversation:
        conversation = await self.store.get_conversation(conversation_id)
        if name is not None:
            conversation = await self.store.rename_conversation(conversation_id, name)
        if archived is not None:
            conversation = await self.store.set_conversation_archived(
                conversation_id, archived=archived
            )
        if yolo_disabled is not None:
            await self.store.set_conversation_yolo_disabled(conversation_id, yolo_disabled)
            conversation = await self.store.get_conversation(conversation_id)
        return conversation

    async def set_yolo_enabled(self, enabled: bool) -> None:
        await self.store.set_yolo_enabled(enabled)

    async def yolo_enabled(self) -> bool:
        return await self.store.yolo_enabled()

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.store.get_conversation(conversation_id)
        await self._retire((conversation_id,))

    async def delete_all_conversations(self) -> int:
        """Apaga todas as Conversations, arquivadas inclusive, e diz quantas eram.

        Os acervos de Corpus e os arquivos do Workspace ficam: são do Operator, não
        da conversa. Turn em andamento é cancelado antes, como no shutdown.
        """
        conversations = await self.store.list_conversations(include_archived=True)
        await self._retire(tuple(conversation.id for conversation in conversations))
        return len(conversations)

    async def _retire(self, conversation_ids: tuple[str, ...]) -> None:
        """Cancela o worker de cada conversa e só então a apaga do store.

        Apagar com o worker vivo deixava o Turn escrevendo numa conversa que não
        existe mais, e o worker se recriava para a fila que a cascata já tinha
        levado junto.
        """
        self._retiring.update(conversation_ids)
        try:
            async with self._worker_lock:
                tasks = [
                    task
                    for conversation_id in conversation_ids
                    if (task := self._workers.get(conversation_id)) is not None
                ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # Um worker parado numa confirmação sai pelo finally do gate, que
            # descarta a pendência: não sobra pergunta para uma conversa apagada.
            for conversation_id in conversation_ids:
                with contextlib.suppress(NotFoundError):
                    await self.store.delete_conversation(conversation_id)
        finally:
            self._retiring.difference_update(conversation_ids)

    async def enqueue_request(self, conversation_id: str, content: str) -> PendingRequest:
        conversation = await self.store.get_conversation(conversation_id)
        if conversation.archived_at is not None:
            raise ApplicationServiceError(
                "conversation_archived",
                "Archived conversations cannot accept requests.",
                status_code=409,
            )
        pending = await self.store.enqueue_request(conversation_id, content)
        await self._ensure_worker(conversation_id)
        return pending

    async def edit_pending_request(
        self, conversation_id: str, request_id: str, content: str
    ) -> PendingRequest:
        request = await self.store.get_request(request_id)
        if request.conversation_id != conversation_id:
            raise ApplicationServiceError(
                "request_not_in_conversation",
                "The request does not belong to this conversation.",
                status_code=404,
            )
        return await self.store.edit_pending_request(request_id, content)

    async def cancel_pending_request(self, conversation_id: str, request_id: str) -> PendingRequest:
        request = await self.store.get_request(request_id)
        if request.conversation_id != conversation_id:
            raise ApplicationServiceError(
                "request_not_in_conversation",
                "The request does not belong to this conversation.",
                status_code=404,
            )
        return await self.store.cancel_pending_request(request_id)

    async def grant(self, conversation_id: str, permission: str) -> Grant:
        await self.store.get_conversation(conversation_id)
        scopes = {
            "WriteGrant": "workspace",
            "WebAccessGrant": "public-network",
        }
        scope = scopes.get(permission)
        if scope is None:
            raise ApplicationServiceError(
                "grant_type_not_public",
                "Only WriteGrant and WebAccessGrant can be changed through this API.",
                status_code=422,
            )
        return await self.store.grant(conversation_id, permission, scope)

    async def select_corpus(self, conversation_id: str, corpus_id: str | None) -> Grant | None:
        """Selecting a Corpus *is* granting access to it; "Desligado" revokes.

        One act, not two. A separate "which corpus is on" column would be a second
        place to disagree with the grant the policy actually enforces.
        """
        await self.store.get_conversation(conversation_id)
        library = self.corpus_library
        if library is None:
            raise ApplicationServiceError(
                "corpus_unavailable",
                "Este perfil não declara modelo de embedding, então não há Corpus.",
                status_code=409,
            )
        for grant in (await self.store.get_session_policy(conversation_id)).grants:
            if grant.permission == CORPUS_GRANT:
                await self.store.revoke_grant(conversation_id, grant.id)
        if corpus_id is None:
            return None
        try:
            await library.read(corpus_id)
        except CorpusLibraryError as error:
            raise ApplicationServiceError(error.code, str(error), status_code=404) from error
        return await self.store.grant(conversation_id, CORPUS_GRANT, corpus_id)

    async def selected_corpus(self, conversation_id: str) -> str | None:
        return granted_corpus_id(await self.store.get_session_policy(conversation_id))

    async def list_corpora(self) -> tuple[Corpus, ...]:
        return await self._library().list()

    async def create_corpus(self, name: str, description: str = "") -> Corpus:
        return await self._corpus_call(self._library().create(name=name, description=description))

    async def rename_corpus(
        self,
        corpus_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Corpus:
        return await self._corpus_call(
            self._library().rename(corpus_id, name=name, description=description)
        )

    async def delete_corpus(self, corpus_id: str) -> None:
        await self._corpus_call(self._library().delete(corpus_id))

    async def list_corpus_documents(self, corpus_id: str) -> tuple[Document, ...]:
        return await self._corpus_call(self._library().store(corpus_id).list_documents())

    async def delete_corpus_document(self, corpus_id: str, document_id: str) -> None:
        await self._corpus_call(self._library().store(corpus_id).delete_document(document_id))

    async def upload_to_corpus(
        self,
        corpus_id: str,
        filename: str,
        data: bytes,
    ) -> "IngestionJob":
        return await self._corpus_call(self._ingestion().ingest_upload(corpus_id, filename, data))

    async def start_corpus_scrape(self, corpus_id: str, seed: str) -> "IngestionJob":
        return await self._corpus_call(self._ingestion().start_scrape(corpus_id, seed))

    def list_corpus_jobs(self, corpus_id: str | None = None) -> tuple["IngestionJob", ...]:
        return self._ingestion().list_jobs(corpus_id)

    def cancel_corpus_job(self, job_id: str) -> "IngestionJob":
        try:
            return self._ingestion().cancel(job_id)
        except CorpusLibraryError as error:
            raise ApplicationServiceError(error.code, str(error), status_code=404) from error

    def _library(self) -> CorpusLibrary:
        if self.corpus_library is None:
            raise ApplicationServiceError(
                "corpus_unavailable",
                "Este perfil não declara modelo de embedding, então não há Corpus.",
                status_code=409,
            )
        return self.corpus_library

    def _ingestion(self) -> "CorpusIngestionService":
        if self.corpus_ingestion is None:
            raise ApplicationServiceError(
                "corpus_unavailable",
                "Este perfil não declara modelo de embedding, então não há Corpus.",
                status_code=409,
            )
        return self.corpus_ingestion

    async def _corpus_call[T](self, awaitable: Awaitable[T]) -> T:
        try:
            return await awaitable
        except CorpusLibraryError as error:
            status = 404 if error.code.endswith("not_found") else 422
            raise ApplicationServiceError(error.code, str(error), status_code=status) from error
        except CorpusStoreError as error:
            raise ApplicationServiceError("corpus_store_error", str(error), status_code=422) from (
                error
            )

    async def list_grants(self, conversation_id: str) -> tuple[Grant, ...]:
        return (await self.store.get_session_policy(conversation_id)).grants

    async def revoke_grant(self, conversation_id: str, grant_id: str) -> Grant:
        grant = next(
            (
                item
                for item in (await self.store.get_session_policy(conversation_id)).grants
                if item.id == grant_id
            ),
            None,
        )
        if grant is None:
            raise ApplicationServiceError(
                "grant_not_found",
                "The active grant was not found.",
                status_code=404,
            )
        if grant.permission == "WorkspaceRootGrant":
            raise ApplicationServiceError(
                "workspace_root_grant_not_public",
                "WorkspaceRootGrant is managed by the server allowlist.",
                status_code=422,
            )
        return await self.store.revoke_grant(conversation_id, grant_id)

    async def add_feedback(
        self,
        conversation_id: str,
        *,
        rating: int,
        comment: str | None,
        turn_id: str | None,
    ) -> Feedback:
        return await self.store.add_feedback(
            conversation_id,
            rating=rating,
            comment=comment,
            turn_id=turn_id,
        )

    async def create_regression_draft(
        self, conversation_id: str, feedback_id: str
    ) -> RegressionDraft:
        feedback = next(
            (
                item
                for item in await self.store.list_feedback(conversation_id)
                if item.id == feedback_id
            ),
            None,
        )
        if feedback is None:
            raise ApplicationServiceError(
                "feedback_not_found",
                "The feedback was not found in this Conversation.",
                status_code=404,
            )
        if feedback.turn_id is None:
            raise ApplicationServiceError(
                "feedback_not_linked_to_turn",
                "Regression drafts require feedback linked to a Turn.",
                status_code=422,
            )
        turn = next(
            (
                item
                for item in await self.store.list_turns(conversation_id)
                if item.id == feedback.turn_id
            ),
            None,
        )
        if turn is None:
            raise ApplicationServiceError(
                "turn_not_found",
                "The feedback Turn was not found in this Conversation.",
                status_code=404,
            )
        return await self.eval_service.create_regression_draft(feedback, turn)

    async def chat_snapshot(self, conversation_id: str) -> dict[str, object]:
        conversation = await self.store.get_conversation(conversation_id)
        pending = await self.store.list_pending_requests(conversation_id)
        history = await self.store.list_canonical_history(conversation_id)
        turns = await self.store.list_turns(conversation_id)
        feedback = await self.store.list_feedback(conversation_id)
        active = next((turn for turn in turns if turn.status is TurnStatus.ACTIVE), None)
        return {
            "conversation": conversation,
            "pending_requests": pending,
            "history": history,
            "active_turn": active,
            "turns": turns,
            "feedback": feedback,
            "pending_confirmation": self._confirmation_gate.pending(conversation_id),
            "confirmation_waivers": sorted(await self.store.waived_confirmations(conversation_id)),
            "yolo": await self.store.yolo_active(conversation_id),
            "corpus_id": granted_corpus_id(await self.store.get_session_policy(conversation_id)),
        }

    async def stop(self, conversation_id: str) -> None:
        await self.store.get_conversation(conversation_id)
        signal = self._stop_signals.get(conversation_id)
        if signal is None:
            raise ApplicationServiceError(
                "no_active_turn",
                "The conversation has no active Turn to stop.",
                status_code=409,
            )
        signal.request_stop()
        # A Turn parked on a confirmation would otherwise ignore the stop until the
        # turn deadline expires.
        self._confirmation_gate.abandon(conversation_id, "operator_stop")

    async def pending_confirmation(self, conversation_id: str) -> ConfirmationRequest | None:
        await self.store.get_conversation(conversation_id)
        return self._confirmation_gate.pending(conversation_id)

    async def resolve_confirmation(
        self,
        conversation_id: str,
        confirmation_id: str,
        *,
        approved: bool,
        waive: bool = False,
    ) -> None:
        await self.store.get_conversation(conversation_id)
        await self._confirmation_gate.resolve(
            conversation_id, confirmation_id, approved=approved, waive=waive
        )

    async def confirmation_waivers(self, conversation_id: str) -> frozenset[str]:
        await self.store.get_conversation(conversation_id)
        return await self.store.waived_confirmations(conversation_id)

    async def revoke_confirmation_waiver(self, conversation_id: str, effect: str) -> None:
        await self.store.get_conversation(conversation_id)
        await self.store.revoke_confirmation_waiver(conversation_id, effect)

    def subscribe(self, conversation_id: str) -> asyncio.Queue[AgentEvent]:
        return self._event_bus.subscribe(conversation_id)

    def unsubscribe(self, conversation_id: str, queue: asyncio.Queue[AgentEvent]) -> None:
        self._event_bus.unsubscribe(conversation_id, queue)

    async def _ensure_worker(self, conversation_id: str) -> None:
        async with self._worker_lock:
            if self._shutting_down:
                raise RuntimeError("application service is shutting down")
            worker = self._workers.get(conversation_id)
            if worker is None or worker.done():
                self._workers[conversation_id] = asyncio.create_task(
                    self._run_conversation(conversation_id),
                    name=f"harness-conversation-{conversation_id}",
                )

    async def _run_conversation(self, conversation_id: str) -> None:
        try:
            while True:
                await self.benchmark_lease.enter_chat(conversation_id)
                try:
                    stop_signal = _CooperativeStopSignal()
                    self._stop_signals[conversation_id] = stop_signal
                    engine = self._build_engine(stop_signal)
                    turn = await engine.start_next_turn(conversation_id)
                    if self._stop_signals.get(conversation_id) is stop_signal:
                        self._stop_signals.pop(conversation_id, None)
                finally:
                    await self.benchmark_lease.leave_chat(conversation_id)
                if turn is None:
                    break
        finally:
            self._stop_signals.pop(conversation_id, None)
            async with self._worker_lock:
                current = asyncio.current_task()
                if self._workers.get(conversation_id) is current:
                    self._workers.pop(conversation_id, None)
                    if (
                        not self._shutting_down
                        and conversation_id not in self._retiring
                        and await self.store.list_pending_requests(conversation_id)
                    ):
                        self._workers[conversation_id] = asyncio.create_task(
                            self._run_conversation(conversation_id),
                            name=f"harness-conversation-{conversation_id}",
                        )

    def _build_engine(self, stop_signal: StopSignal) -> AgentEngine:
        return AgentEngine(
            store=self.store,
            runtime=self.runtime,
            tool_executor_factory=self._tool_executor_factory,
            context_builder=ContextBuilder(
                self.estimator,
                context_window=self.config.context.initial_budget_tokens,
                output_budget=self.config.loop.max_output_tokens,
                automation_role=self.config.runtime_profile.internal_automation_role,
            ),
            event_sink=self._event_sink,
            # An engine is built per Turn, so a conversation that crosses midnight
            # gets the new date on its next Turn without anything having to refresh.
            system_prompt=build_system_prompt(
                self.config,
                today=datetime.now(UTC).date(),
                operator_notes=self._operator_notes,
                vision_tool_offered=self._vision_offered,
            ),
            tool_schemas=(),
            model_options={
                "temperature": self.config.execution_route.sampling.temperature,
                "presence_penalty": self.config.execution_route.sampling.presence_penalty,
            },
            seed=None,
            max_model_invocations=self.config.loop.max_steps,
            max_tool_calls_per_step=self.config.loop.max_tool_calls_per_step,
            max_tool_calls_per_turn=self.config.loop.max_tool_calls_per_turn,
            max_read_calls_per_turn=self.config.loop.max_read_calls_per_turn,
            max_malformed_model_attempts=self.config.loop.max_malformed_model_attempts,
            tool_effects=self.config.tool_registry.effects_by_tool,
            max_turn_duration_seconds=self.config.loop.max_turn_duration_seconds,
            runtime_readiness=self._runtime_readiness,
            stop_signal=stop_signal,
            think=self.config.execution_route.sampling.thinking,
            response_markup=ResponseMarkup.of(self.config.runtime_profile),
            turn_retrieval=(
                CorpusTurnRetrieval(
                    store=self.store,
                    retriever=self._corpus_retriever,
                    runtime=self.runtime,
                    observability_store=self.observability_store,
                )
                if self._corpus_retriever is not None
                else None
            ),
            confirmation_gate=self._confirmation_gate,
        )

    def workspace_root(self, workspace_id: str) -> Path:
        for workspace in self._workspaces_by_root.values():
            if workspace.id == workspace_id:
                return workspace.root
        raise ApplicationServiceError(
            "workspace_not_registered",
            "The workspace is not registered by this server.",
            status_code=404,
        )


def _default_eval_service(
    *,
    store: ConversationStore,
    lease: BenchmarkLease,
    registry: ToolRegistryConfig,
    config: HarnessConfig,
    runtime: ModelRuntime,
    estimator: TokenEstimator,
    operator_notes: str,
    browser_executable: str | Path | None,
) -> EvalService:
    project_root = Path(__file__).resolve().parents[2]
    catalog = load_eval_catalog(
        project_root / "evals/fixtures/regressions.json",
        project_root / "evals/experiments.json",
        contract_root=project_root,
    )
    conversation_database = Path(store.database)
    if conversation_database.name == ":memory:":
        raise ValueError("ConversationStore must use a file when evals are enabled")
    eval_database = conversation_database.with_name("evals.sqlite3")
    contract = ContractCaseRunner(config=config)
    # O mesmo executável declarado em host.json: a bancada tem de medir o
    # navegador que a produção usa, não o primeiro que aparecer no PATH.
    browser_guard = BraveEgressGuard(executable=browser_executable)
    model = ModelCaseRunner(
        config=config,
        runtime=runtime,
        estimator=estimator,
        browser_guard=browser_guard,
        operator_notes=operator_notes,
    )
    bench = BrowserBenchCaseRunner(registry=registry, browser_guard=browser_guard)
    # An experiment picks its fixtures by tag, so one tier has to cover several
    # fixture types: the composite routes each one to the runner that can execute it.
    live = CompositeCaseRunner((contract, bench, model))
    return EvalService(
        store=EvalStore(eval_database),
        catalog=catalog,
        lease=lease,
        runners={
            EvalTier.CONTRACT: contract,
            EvalTier.MODEL_SMOKE: live,
            EvalTier.EXPERIMENT: live,
        },
    )


def _corpus_library(config: HarnessConfig, directory: str | Path | None) -> CorpusLibrary | None:
    embedding = config.runtime_profile.embedding
    if directory is None or embedding is None:
        return None
    return CorpusLibrary(
        directory,
        embedding_model=embedding.id,
        embedding_dimensions=embedding.dimensions,
    )


class _ConversationToolExecutorFactory:
    def __init__(
        self,
        *,
        store: ConversationStore,
        registry: ToolRegistryConfig,
        workspace_root: Callable[[str], Path],
        coordinator: WorkspaceCoordinator,
        search_endpoint: str | None,
        max_read_bytes: int,
        max_search_bytes: int,
        browser_capability: BrowserCapability | None = None,
        browser_egress_guard: BrowserEgressGuard | None = None,
        corpus_retriever: CorpusRetriever | None = None,
        vision_runtime: VisionRuntime | None = None,
        vision_config: VisionConfig | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._vision_runtime = vision_runtime
        self._vision_config = vision_config or VisionConfig()
        self._workspace_root = workspace_root
        self._coordinator = coordinator
        self._search_endpoint = search_endpoint
        self._max_read_bytes = max_read_bytes
        self._max_search_bytes = max_search_bytes
        self._browser_capability = browser_capability
        self._browser_egress_guard = browser_egress_guard
        self._corpus_retriever = corpus_retriever

    async def effective_tool_schemas(self, conversation_id: str) -> tuple[ToolSchema, ...]:
        """The enabled catalogue, whatever the conversation has been granted.

        Hiding write_file until the Operator grants WriteGrant does not stop the
        model from calling it: it calls the name it remembers, without the schema,
        and the call arrives missing required arguments. What the Operator then
        sees is invalid_tool_arguments, which names neither the grant nor the fix.
        Offering the whole catalogue costs nothing — authorization is by effect in
        preflight, and a schema grants no access — and the refusal becomes the one
        the Operator can act on: write_grant_required.
        """
        del conversation_id
        # web_search is always offered: its fallback provider needs no credential,
        # so there is no configuration under which the tool cannot answer at all.
        # corpus_search is the exception that proves the rule: without an embedding
        # model there is no executor behind it, and offering a name that can only
        # answer unknown_tool teaches the model to spend steps on it.
        return tuple(
            definition.tool_schema()
            for definition in self._registry.model_tools
            if definition.status == "enabled"
            and (self._corpus_retriever is not None or CORPUS_EFFECT not in definition.effects)
            # O mesmo vale para a visão: sem o modelo configurado, describe_image
            # só saberia responder unknown_tool.
            and (
                self._vision_runtime is not None or LOCAL_INFERENCE_EFFECT not in definition.effects
            )
        )

    async def create(self, conversation_id: str) -> ToolExecutor:
        conversation = await self._store.get_conversation(conversation_id)
        policy = await self._store.get_session_policy(conversation_id)
        root = self._workspace_root(conversation.workspace_id)
        registry = self._registry
        local_executor = RegistryToolExecutor(
            registry=registry,
            workspace_root=root,
            session_policy=policy,
            max_read_bytes=self._max_read_bytes,
            max_search_bytes=self._max_search_bytes,
        )
        local: ToolExecutor = _CoordinatedToolExecutor(
            executor=local_executor,
            coordinator=self._coordinator,
            workspace_id=conversation.workspace_id,
        )
        web = WebToolExecutor(
            registry=registry,
            session_policy=policy,
            search_endpoint=self._search_endpoint,
            browser_capability=self._browser_capability,
            browser_egress_guard=self._browser_egress_guard,
        )
        corpus = (
            CorpusToolExecutor(
                registry=registry,
                session_policy=policy,
                retriever=self._corpus_retriever,
            )
            if self._corpus_retriever is not None
            else None
        )
        vision = (
            VisionToolExecutor(
                registry=registry,
                workspace_root=root,
                session_policy=policy,
                runtime=self._vision_runtime,
                config=self._vision_config,
            )
            if self._vision_runtime is not None
            else None
        )
        routes: dict[str, ToolExecutor] = {}
        for definition in registry.model_tools:
            # Pelo efeito declarado, nunca pelo nome: uma tool nova é roteada por
            # ter dito o que faz, e uma sem executor não é oferecida ao modelo.
            if CORPUS_EFFECT in definition.effects:
                if corpus is not None:
                    routes[definition.name] = corpus
                continue
            if LOCAL_INFERENCE_EFFECT in definition.effects:
                if vision is not None:
                    routes[definition.name] = vision
                continue
            routes[definition.name] = web if "data_egress" in definition.effects else local
        return CompositeToolExecutor(routes=routes)


class CorpusTurnRetrieval:
    """The automation side of the same retrieval the tool performs.

    The rewrite is one short generation and it buys two things at once: a query
    that stands on its own — "and the second one?" retrieves nothing — and an
    English one for the lexical leg, which is blind to language. The dense leg
    keeps the Operator's own words, so a bad rewrite degrades the search instead
    of replacing it. When the rewrite fails there is simply no lexical leg:
    feeding it Portuguese against an English Corpus measurably ranks worse than
    not searching lexically at all.
    """

    def __init__(
        self,
        *,
        store: ConversationStore,
        retriever: CorpusRetriever,
        runtime: ModelRuntime,
        observability_store: ObservabilityStore,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._runtime = runtime
        self._observability_store = observability_store
        self._portuguese = PortugueseDetector()

    async def for_turn(
        self,
        conversation_id: str,
        question: str,
    ) -> Mapping[str, JsonValue] | None:
        policy = await self._store.get_session_policy(conversation_id)
        corpus_id = granted_corpus_id(policy)
        if corpus_id is None:
            return None
        history = await self._store.list_canonical_history(conversation_id)
        lexical = await self._standalone_english_query(question, history)
        try:
            retrieval = await self._retriever.retrieve(
                corpus_id,
                question,
                lexical_query=lexical,
            )
        except CorpusLibraryError as error:
            return {"status": "blocked", "reason_code": error.code, "detail": str(error)}
        # Contagens e classes, nunca a pergunta nem a passagem: o store de
        # telemetria recusa conteúdo por construção, e esta é a informação que
        # diz se a recuperação está entregando alguma coisa.
        await self._observability_store.record(
            event_type="corpus.retrieval",
            payload={
                "corpus_id": retrieval.corpus_id,
                "passages": len(retrieval.chunks),
                "rewritten": lexical is not None,
                "taints": list(retrieval.taints),
            },
        )
        payload = retrieval.payload()
        return {
            "corpus_id": retrieval.corpus_id,
            "search_query": lexical,
            **cast(Mapping[str, JsonValue], payload),
        }

    async def _standalone_english_query(
        self,
        question: str,
        history: Sequence[CanonicalHistoryEntry],
    ) -> str | None:
        previous = [
            content
            for entry in history[-6:]
            if entry.kind is CanonicalHistoryEntryKind.USER_MESSAGE
            and isinstance(content := entry.payload.get("content"), str)
        ][-2:]
        context = "\n".join(f"- {item}" for item in previous)
        instruction = (
            "Translate the request below into an English search query that stands "
            "on its own. Keep names, identifiers, error codes and numbers exactly "
            "as written. Answer with the English query alone: no quotes, no "
            "explanation, and never repeat the original wording.\n\n"
            "Example\n"
            "Request: Em que porta o proxy escuta?\n"
            "Query: which port does the proxy listen on\n\n"
            f"{'Earlier requests:\n' + context + '\n\n' if context else ''}"
            f"Request: {question}\nQuery:"
        )
        try:
            response = await self._runtime.generate(
                ModelRequest(
                    messages=(ModelMessage(role=ModelRole.USER, content=instruction),),
                    tools=(),
                    options={"temperature": 0},
                    seed=0,
                    max_output_tokens=64,
                    think=False,
                )
            )
        except ModelRuntimeError:
            return None
        content = (response.content or "").strip().splitlines()
        query = content[0].strip().strip('"').removeprefix("Query:").strip() if content else ""
        if not query:
            return None
        # Um 4B pedido para traduzir às vezes devolve a pergunta como veio. Entregar
        # isso à perna lexical é pior do que não ter perna lexical: contra um acervo
        # em inglês, uma palavra em comum carrega o casamento inteiro e derruba a
        # perna densa, que já tinha acertado. Sem reescrita, sem BM25.
        return None if self._portuguese.is_portuguese(query) else query


class _CoordinatedToolExecutor:
    def __init__(
        self,
        *,
        executor: ToolExecutor,
        coordinator: WorkspaceCoordinator,
        workspace_id: str,
    ) -> None:
        self._executor = executor
        self._coordinator = coordinator
        self._workspace_id = workspace_id

    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight:
        return await self._executor.preflight(calls)

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self._coordinator.execute(self._workspace_id, call, self._executor)

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None:
        # A preview reads; the coordinator's lock exists to serialise writes.
        return await self._executor.preview(call)


class _CooperativeStopSignal:
    def __init__(self) -> None:
        self._requested = False

    @property
    def stop_requested(self) -> bool:
        return self._requested

    def request_stop(self) -> None:
        self._requested = True


class OperatorConfirmationGate:
    """Holds one pending confirmation per Conversation until the Operator decides.

    Registration and announcement happen in the same call so a decision cannot
    arrive before the request is visible. Nothing is persisted: a confirmation only
    matters while its Turn is alive, and a Turn does not survive a restart.
    """

    def __init__(self, event_sink: EventSink, store: ConversationStore) -> None:
        self._event_sink = event_sink
        self._store = store
        self._pending: dict[str, tuple[ConfirmationRequest, asyncio.Future[ConfirmationDecision]]]
        self._pending = {}

    def pending(self, conversation_id: str) -> ConfirmationRequest | None:
        entry = self._pending.get(conversation_id)
        return None if entry is None else entry[0]

    async def will_announce(self, request: ConfirmationRequest) -> bool:
        return not await self._waived(request)

    async def _waived(self, request: ConfirmationRequest) -> bool:
        if request.reason_code in YOLO_CONFIRMATION_REASONS and await self._store.yolo_active(
            request.conversation_id
        ):
            return True
        return request.reason_code in WAIVABLE_CONFIRMATION_REASONS and MUTATION_EFFECT in (
            await self._store.waived_confirmations(request.conversation_id)
        )

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationDecision:
        if request.conversation_id in self._pending:
            raise RuntimeError("a Conversation runs one Turn at a time")
        if await self._waived(request):
            # The Operator said to stop asking for this Conversation. The waiver
            # covers the plain write, never the tainted one: the web asking for a
            # write is a different question, and it was never answered.
            await self._record_grant(request)
            return ConfirmationDecision(
                approved=True,
                reason_code=waived_reason_code(request.reason_code),
            )
        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        self._pending[request.conversation_id] = (request, future)
        try:
            await self._event_sink.emit(
                AgentEvent(
                    kind=AgentEventKind.CONFIRMATION_REQUIRED,
                    turn_id=request.turn_id,
                    step_sequence=request.step_sequence,
                    payload={
                        "confirmation_id": request.id,
                        "reason_code": request.reason_code,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments": dict(call.arguments),
                            }
                            for call in request.tool_calls
                        ],
                        "previews": [
                            {
                                "tool_call_id": preview.tool_call_id,
                                "path": preview.path,
                                "kind": preview.kind,
                                "diff": preview.diff,
                                "truncated": preview.truncated,
                            }
                            for preview in request.previews
                        ],
                    },
                    conversation_id=request.conversation_id,
                    request_id=request.request_id,
                )
            )
            return await future
        finally:
            self._pending.pop(request.conversation_id, None)

    async def resolve(
        self,
        conversation_id: str,
        confirmation_id: str,
        *,
        approved: bool,
        waive: bool = False,
    ) -> None:
        entry = self._pending.get(conversation_id)
        if entry is None or entry[0].id != confirmation_id:
            raise ApplicationServiceError(
                "confirmation_not_pending",
                "There is no pending confirmation with this identifier.",
                status_code=409,
            )
        request, future = entry
        if future.done():
            return
        if waive and approved and request.reason_code in WAIVABLE_CONFIRMATION_REASONS:
            # Waiving is its own act, not a side effect of approving: the Operator
            # ticked a box that says so, and it is recorded where it can be revoked.
            await self._store.waive_confirmation(conversation_id, MUTATION_EFFECT)
        if approved:
            await self._record_grant(request)
        future.set_result(
            ConfirmationDecision(
                approved=approved,
                reason_code=_decision_reason_code(request.reason_code, approved=approved),
            )
        )

    def abandon(self, conversation_id: str, reason_code: str) -> None:
        entry = self._pending.get(conversation_id)
        if entry is None or entry[1].done():
            return
        entry[1].set_result(ConfirmationDecision(approved=False, reason_code=reason_code))

    async def _record_grant(self, request: ConfirmationRequest) -> None:
        """Gives the grant the approved batch is missing.

        Policy still requires the grant for every workspace_write and every
        data_egress; what changed is where the Operator gives it. Being asked to
        find a chip before anything has happened, and then to approve the same call
        in a dialog — or worse, to re-send the prompt — is one decision charged
        twice, so the dialog is where it is taken, with the call in view.
        """
        granted = _DIALOG_GRANTS.get(request.reason_code)
        if granted is None:
            return
        permission, scope = granted
        await self._store.grant(request.conversation_id, permission, scope)


# The grant each dialog reason gives when approved, with the scope the public
# grant API uses for the same permission.
_DIALOG_GRANTS = {
    "write_grant_required": ("WriteGrant", "workspace"),
    "web_access_grant_required": ("WebAccessGrant", "public-network"),
}


def _decision_reason_code(requested: str, *, approved: bool) -> str:
    """Answers in the words of the question: …_required becomes …_approved/_denied."""
    stem = requested.removesuffix("_required")
    return f"{stem}_approved" if approved else f"{stem}_denied"


class _LiveEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[AgentEvent]]] = {}

    def subscribe(self, conversation_id: str) -> asyncio.Queue[AgentEvent]:
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._subscribers.setdefault(conversation_id, set()).add(queue)
        return queue

    def unsubscribe(self, conversation_id: str, queue: asyncio.Queue[AgentEvent]) -> None:
        subscribers = self._subscribers.get(conversation_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(conversation_id, None)

    async def emit(self, event: AgentEvent) -> None:
        for queue in tuple(self._subscribers.get(event.conversation_id, ())):
            queue.put_nowait(event)


class _ServiceEventSink:
    def __init__(self, bus: _LiveEventBus, observability_store: ObservabilityStore) -> None:
        self._bus = bus
        self._observability_store = observability_store

    async def emit(self, event: AgentEvent) -> None:
        await self._bus.emit(event)
        payload: dict[str, JsonValue] = {
            "kind": event.kind.value,
            "conversation_id": event.conversation_id,
            "request_id": event.request_id,
        }
        for key in (
            "base_seed",
            "id",
            "name",
            "seed",
            "tool_call_id",
            "status",
            "retryable",
            "outcome_kind",
            "reason_code",
            "estimated_input_tokens",
            "context_window",
            "output_budget",
            "dropped_turn_ids",
            "output_tokens",
            "eval_ms",
            "prompt_tokens",
            "prompt_eval_ms",
        ):
            value = event.payload.get(key)
            if value is not None:
                payload[key] = value
        # `meta` inteiro não pode ser copiado: ele carrega `final_url`, que é
        # conteúdo. O corte de um resultado é a única coisa dali que responde se
        # os tetos das tools estão apertando, então só ele sobe.
        meta = event.payload.get("meta")
        if isinstance(meta, Mapping):
            truncated = meta.get("truncated")
            if isinstance(truncated, bool):
                payload["truncated"] = truncated
        await self._observability_store.record(
            event_type=f"agent.{event.kind.value}",
            payload=payload,
            turn_id=event.turn_id,
            step_sequence=event.step_sequence,
        )
