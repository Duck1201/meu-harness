from pathlib import Path

from harness import HarnessConfig, ReplayPolicy, load_config


def test_config_loader_reads_harness_profiles_and_tool_registry() -> None:
    config = load_config(Path("config/harness.json"))

    assert isinstance(config, HarnessConfig)
    assert config.loop.max_steps == 15
    assert config.loop.max_output_tokens == 8192
    assert config.default_execution_route == "local_web_tools"
    assert config.execution_route.id == "local_web_tools"
    assert config.runtime_profile.model.id == "harness-gemma4-e4b-qat:latest"
    assert (
        config.runtime_profile.profile_digest_sha256
        == "bd0b01eca795c325997c317115539f6465193c1daece8579bc80a038d21a9a92"
    )
    assert config.context.initial_budget_tokens == 65536
    assert config.context.max_tool_read_bytes == 32768
    assert config.context.max_tool_search_bytes == 32768
    assert [tool.name for tool in config.tool_registry.model_tools] == [
        "read_file",
        "write_file",
        "edit",
        "list_directory",
        "glob",
        "grep_search",
        "web_search",
        "web_fetch",
        "corpus_search",
        "calculate",
        "get_weather",
    ]
    assert config.tool_schemas[0].parameters["type"] == "object"
    assert (
        config.tool_registry.model_tools[0].replay_policy
        is ReplayPolicy.NEVER_CACHE_WORKSPACE_READS
    )


# Bytes por token varia muito com o conteúdo, e o pior caso não é código: 64 KiB
# de Python deram 4,36 bytes/token, mas um `read_file` real de markdown em pt-BR
# com números deu 2,83 (32.768 bytes viraram 11.580 tokens, turno medido em
# 2026-08-15). O piso é 2,5 porque a conta só protege se errar para o lado caro.
_BYTES_PER_TOKEN_FLOOR = 2.5

# Medido no mesmo lugar: system prompt (239) mais os schemas das 11 tools (1.663).
_FIXED_FLOOR_TOKENS = 1902


def test_a_full_batch_of_reads_at_the_cap_still_fits_the_context_budget() -> None:
    """O pior passo tem de caber na janela, senão o Turn morre sem recurso.

    `ContextBuilder` descarta turnos antigos para caber no orçamento, mas quando o
    turno corrente sozinho estoura não sobra o que descartar: ele levanta
    `ContextBudgetExceeded` e o Turn termina em `context_budget_exceeded`. Subir o
    teto de leitura, subir o fan-out por passo ou baixar o orçamento sem refazer
    esta conta reabre exatamente esse caminho.
    """
    config = load_config(Path("config/harness.json"))

    worst_case_bytes = config.loop.max_tool_calls_per_step * config.context.max_tool_read_bytes
    worst_case_tokens = worst_case_bytes / _BYTES_PER_TOKEN_FLOOR
    available = config.context.initial_budget_tokens - config.loop.max_output_tokens

    assert worst_case_tokens + _FIXED_FLOOR_TOKENS <= available
