from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

from .domain import JsonValue
from .ports import ToolSchema


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class SamplingConfig(ConfigModel):
    temperature: float
    presence_penalty: float
    thinking: bool


class ExecutionRouteConfig(ConfigModel):
    id: str
    status: str
    runtime_profile: str
    sampling: SamplingConfig
    streaming: bool


class LoopConfig(ConfigModel):
    max_steps: int
    max_tool_calls_per_step: int
    max_tool_calls_per_turn: int
    max_read_calls_per_turn: int = 40
    max_turn_duration_seconds: float
    # Teto de uma geração, não do Turn: numa bateria de 828 casos de bancada, 14
    # morreram entre 120 e 154 s contra uma mediana de 6 a 17 s, sempre nas
    # fixtures de texto longo — o timeout de 120 s do httpx cortava e o harness
    # relatava provedor indisponível. Fica acima da pior geração observada e
    # abaixo de `max_turn_duration_seconds`, que continua sendo quem encerra.
    model_generation_timeout_seconds: float
    max_malformed_model_attempts: int
    max_output_tokens: int
    offer_tools_on_final_step: bool


class ContextConfig(ConfigModel):
    initial_budget_tokens: int
    # Teto de uma página de resultado, não de um arquivo: `read_file` e
    # `grep_search` paginam por `next_offset`. O valor vive aqui porque quem o
    # limita é o orçamento — `max_tool_calls_per_step` resultados no teto têm de
    # caber na janela, e `tests/test_config.py` falha se pararem de caber.
    max_tool_read_bytes: int
    max_tool_search_bytes: int


class CorpusIngestionConfig(ConfigModel):
    accepted_extensions: tuple[str, ...]
    # Quem lê HTML na ingestão. `scrapling` limpa conteúdo oculto antes de extrair;
    # o padrão só muda com o experimento de ingestão medido.
    html_extractor: Literal["builtin", "scrapling"] = "builtin"
    max_upload_bytes: int
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_minimum_tokens: int


class CorpusRetrievalConfig(ConfigModel):
    dense_candidates: int
    lexical_candidates: int
    reciprocal_rank_constant: int
    dense_similarity_floor: float
    injected_passages: int
    max_injected_tokens: int


class CorpusBrowserEscalationConfig(ConfigModel):
    # `symptom` renderiza no browser a página que o HTTP trouxe sem texto; o
    # padrão fica `disabled` até a rota ser medida (ADR 0014).
    mode: Literal["disabled", "symptom"] = "disabled"
    thin_page_chars: int = 200
    stealth: bool = False
    timeout_milliseconds: int = 30_000


class CorpusCrawlConfig(ConfigModel):
    respect_robots_txt: bool
    prefer_sitemap: bool
    max_depth: int
    max_pages: int
    max_total_bytes: int
    delay_milliseconds: int
    browser_escalation: CorpusBrowserEscalationConfig = CorpusBrowserEscalationConfig()


class CorpusMediaWikiConfig(ConfigModel):
    page_batch: int
    thin_extract_chars: int
    backoff_seconds: tuple[float, ...]


class CorpusScraperConfig(ConfigModel):
    mediawiki: CorpusMediaWikiConfig
    html_crawl: CorpusCrawlConfig


class CorpusEmbeddingConfig(ConfigModel):
    batch_size: int


class CorpusAnswerJudgeConfig(ConfigModel):
    # `advisory` anota cada passagem com a probabilidade de trazer o fato pedido;
    # nasce `disabled` até o experimento medir que ajuda (ADR 0015).
    mode: Literal["disabled", "advisory"] = "disabled"
    threshold: float = 0.5
    model: str = ""
    ollama_tag: str = ""
    ollama_digest: str = ""


class VisionConfig(ConfigModel):
    # `enabled` oferece `describe_image`. O padrão do modelo é `disabled`; o
    # contrato liga desde a promoção de `vision_tool_enablement` (ADR 0016).
    mode: Literal["disabled", "enabled"] = "disabled"
    ollama_tag: str = ""
    ollama_digest: str = ""
    max_image_bytes: int = 10 * 1024 * 1024
    accepted_formats: tuple[str, ...] = ("png", "jpeg", "webp")
    max_output_tokens: int = 700
    context_tokens: int = 8192


class CorpusConfig(ConfigModel):
    embedding: CorpusEmbeddingConfig
    ingestion: CorpusIngestionConfig
    retrieval: CorpusRetrievalConfig
    scraper: CorpusScraperConfig
    answer_judge: CorpusAnswerJudgeConfig = CorpusAnswerJudgeConfig()


class ModelIdentityConfig(ConfigModel):
    id: str
    base_model: str


class CapabilityConfig(ConfigModel):
    support: str
    evidence: Mapping[str, JsonValue]
    gate_status: str


class EmbeddingIdentityConfig(ConfigModel):
    id: str
    digest_sha256: str
    dimensions: int
    max_input_tokens: int


class RuntimeBackend(StrEnum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


class RuntimeIdentityConfig(ConfigModel):
    backend: RuntimeBackend


# Markup the Qwen template uses for a tool call written as text. Kept as the
# default because the functional profile was measured with it; a profile from
# another family declares its own in `tool_markup_leak_markers`.
QWEN_TOOL_MARKUP = (
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
QWEN_REASONING_MARKUP = ("<think>", "</think>")


class RuntimeProfileConfig(ConfigModel):
    id: str
    status: str
    release_eligible: bool
    profile_digest_sha256: str
    model: ModelIdentityConfig
    runtime: RuntimeIdentityConfig
    installation: Mapping[str, JsonValue]
    # O que o motor procura numa resposta final para saber que uma tool call ou o
    # raciocínio vazou para o texto. É do template, não do harness: cada família
    # escreve a sua marcação, e um detector com as tags de outra deixa o vazamento
    # chegar ao Operator como resposta.
    tool_markup_leak_markers: tuple[str, ...] = QWEN_TOOL_MARKUP
    reasoning_leak_markers: tuple[str, ...] = QWEN_REASONING_MARKUP
    # Papel das automações internas no ModelView; `user` para o template que
    # descarta mensagem de ferramenta sem chamada correspondente.
    internal_automation_role: Literal["tool", "user"] = "tool"

    @property
    def context_window(self) -> int:
        """A janela com que o runtime foi instalado: `num_ctx`, texto no Modelfile."""
        parameters = self.installation.get("parameters")
        raw = parameters.get("num_ctx") if isinstance(parameters, Mapping) else None
        if isinstance(raw, bool) or not isinstance(raw, str | int):
            raise ValueError(f"{self.id}: installation.parameters.num_ctx is missing")
        return int(raw)

    capabilities: Mapping[str, CapabilityConfig] = {}
    # Ausente é uma resposta: um perfil sem modelo de embedding não tem Corpus,
    # e o harness prefere dizer isso a inventar um padrão.
    embedding: EmbeddingIdentityConfig | None = None


class ModelProfilesConfig(ConfigModel):
    schema_version: int
    active_runtime_profile: str
    runtime_profiles: tuple[RuntimeProfileConfig, ...]


class ReplayPolicy(StrEnum):
    NEVER_CACHE_WORKSPACE_READS = "never_cache_workspace_reads"
    RECONCILE_POSTCONDITION_BEFORE_REPLAY = "reconcile_postcondition_before_replay"
    CACHE_NORMALIZED_ARGUMENTS_PER_TURN = "cache_normalized_arguments_per_turn"
    CACHE_EXACT_URL_PER_TURN = "cache_exact_url_per_turn"


class ToolDefinitionConfig(ConfigModel):
    name: str
    status: str
    description: str
    parameters: Mapping[str, JsonValue]
    effects: tuple[str, ...] = ()
    required_grants: tuple[str, ...] = ()
    replay_policy: ReplayPolicy

    def tool_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def normalized_arguments(self, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Drops optional arguments the model filled with an empty string.

        Offered a schema, the runtime fills every property in it, optional ones
        included: asked to create a file it sent ``expected_current_sha256: ""``
        alongside the content, and the whole batch died on the digest pattern.
        An empty optional is the model saying nothing, so it is read as nothing —
        the argument goes back to being absent, and absent is what the rest of
        preflight already decides on. Required properties are left alone: an empty
        string there is a value, and writing an empty file is a real request.
        """
        required = self.parameters.get("required")
        names = (
            frozenset(item for item in required if isinstance(item, str))
            if isinstance(required, Sequence) and not isinstance(required, str)
            else frozenset[str]()
        )
        return {key: value for key, value in arguments.items() if value != "" or key in names}


class ToolRegistryConfig(ConfigModel):
    schema_version: int
    registry_version: str
    model_tools: tuple[ToolDefinitionConfig, ...]

    @property
    def effects_by_tool(self) -> Mapping[str, tuple[str, ...]]:
        """Effect classes per tool name, for whoever budgets or judges by class."""
        return {definition.name: definition.effects for definition in self.model_tools}


class _HarnessFile(ConfigModel):
    schema_version: int
    contract_version: str
    runtime_profiles: str
    tool_registry: str
    default_runtime_profile: str
    default_execution_route: str
    execution_routes: tuple[ExecutionRouteConfig, ...]
    loop: LoopConfig
    context: ContextConfig
    corpus: CorpusConfig
    vision: VisionConfig = VisionConfig()


class HarnessConfig(ConfigModel):
    schema_version: int
    contract_version: str
    default_runtime_profile: str
    default_execution_route: str
    execution_routes: tuple[ExecutionRouteConfig, ...]
    loop: LoopConfig
    context: ContextConfig
    corpus: CorpusConfig
    vision: VisionConfig = VisionConfig()
    model_profiles: ModelProfilesConfig
    tool_registry: ToolRegistryConfig

    @property
    def execution_route(self) -> ExecutionRouteConfig:
        for route in self.execution_routes:
            if route.id == self.default_execution_route:
                return route
        raise ValueError(f"execution route not found: {self.default_execution_route}")

    @property
    def runtime_profile(self) -> RuntimeProfileConfig:
        return self.runtime_profile_by_id(self.execution_route.runtime_profile)

    def runtime_profile_by_id(self, profile_id: str) -> RuntimeProfileConfig:
        for profile in self.model_profiles.runtime_profiles:
            if profile.id == profile_id:
                return profile
        raise ValueError(f"runtime profile not found: {profile_id}")

    def for_runtime_profile(self, profile_id: str) -> Self:
        """The same contract with the default route pointed at another profile.

        A bake-off arm changes the model and nothing else: route, loop, context and
        registry stay the ones the control arm ran with. Deriving the whole config,
        instead of handing the runner a profile on the side, keeps every reader —
        the system prompt's capability lines included — looking at one profile.
        """
        self.runtime_profile_by_id(profile_id)
        routes = tuple(
            route.model_copy(update={"runtime_profile": profile_id})
            if route.id == self.default_execution_route
            else route
            for route in self.execution_routes
        )
        return self.model_copy(update={"execution_routes": routes})

    @property
    def tool_schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(
            tool.tool_schema()
            for tool in self.tool_registry.model_tools
            if tool.status == "enabled"
        )


def load_config(path: str | Path = Path("config/harness.json")) -> HarnessConfig:
    harness_path = Path(path)
    harness = _HarnessFile.model_validate_json(harness_path.read_text(encoding="utf-8"))
    project_root = harness_path.resolve().parent.parent

    profiles_path = _referenced_path(project_root, harness.runtime_profiles)
    profiles = ModelProfilesConfig.model_validate_json(profiles_path.read_text(encoding="utf-8"))

    registry_path = _referenced_path(project_root, harness.tool_registry)
    registry = ToolRegistryConfig.model_validate_json(registry_path.read_text(encoding="utf-8"))

    loaded = HarnessConfig(
        schema_version=harness.schema_version,
        contract_version=harness.contract_version,
        default_runtime_profile=harness.default_runtime_profile,
        default_execution_route=harness.default_execution_route,
        execution_routes=harness.execution_routes,
        loop=harness.loop,
        context=harness.context,
        corpus=harness.corpus,
        vision=harness.vision,
        model_profiles=profiles,
        tool_registry=registry,
    )
    _ = loaded.execution_route
    _ = loaded.runtime_profile
    return loaded


def _referenced_path(project_root: Path, reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else project_root / path
