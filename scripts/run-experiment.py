#!/usr/bin/env python
"""Runs one experiment from the manifest and prints its report.

The campaign has to be reproducible from the repository, not from a REPL: this
drives the same EvalService the web surface drives, with the same protocol —
15 cases per arm for a pilot, 50 for a promotion, three recorded seeds.

    uv run python scripts/run-experiment.py guarded_web_brave_escalation --phase pilot
    uv run python scripts/run-experiment.py --tier model_smoke
    uv run python scripts/run-experiment.py --tier model_smoke --profile <runtime_profile_id>
    uv run python scripts/run-experiment.py runtime_profile_bakeoff --phase pilot

A braço que declara `runtime_profile` roda naquele perfil: o switch descarrega o
modelo anterior, sobe o do braço e mede o orçamento com o tokenizer dele, lido de
`--tokenizer-dir/<profile_id>.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import (  # noqa: E402
    BenchmarkLease,
    EngineReadiness,
    EvalPhase,
    EvalRunStatus,
    EvalStore,
    EvalTier,
    OllamaEmbeddingRuntime,
    load_config,
)
from harness.brave_browser import BraveEgressGuard  # noqa: E402
from harness.config import RuntimeBackend  # noqa: E402
from harness.evals import (  # noqa: E402
    EvalService,
    ModelCaseRunner,
    build_live_runner,
    load_eval_catalog,
)
from harness.evals.runtime_switch import (  # noqa: E402
    OllamaLauncher,
    ProfileRuntimeSwitch,
    RuntimeSwitchError,
)
from harness.system_prompt import load_operator_notes  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434"


async def run(
    experiment_id: str,
    phase: EvalPhase,
    tier: EvalTier,
    tokenizer: Path,
    tokenizer_dir: Path,
    profile: str | None,
    database: Path,
) -> int:
    route_config = load_config()
    config = route_config.for_runtime_profile(profile) if profile is not None else route_config
    catalog = load_eval_catalog(
        ROOT / "evals/fixtures/regressions.json",
        ROOT / "evals/experiments.json",
        contract_root=ROOT,
    )
    # O embedding indexa o acervo das fixtures e é o mesmo em todo braço: um
    # bake-off compara o modelo de chat diante das mesmas passagens.
    embedding = route_config.runtime_profile.embedding
    switch = ProfileRuntimeSwitch(
        launchers={
            RuntimeBackend.OLLAMA: OllamaLauncher(
                base_url=OLLAMA_URL,
                timeout=config.loop.model_generation_timeout_seconds,
                keep_loaded=frozenset({embedding.id} if embedding is not None else ()),
            ),
        },
        tokenizer_dir=tokenizer_dir,
        tokenizer_overrides={route_config.runtime_profile.id: tokenizer},
    )
    try:
        runtime, estimator = await switch.activate(config.runtime_profile)
    except RuntimeSwitchError as error:
        print(f"RuntimeProfile is not ready: {error}", file=sys.stderr)
        return 2
    browser = await BraveEgressGuard().readiness()
    if not browser.ready:
        print(f"Browser is not ready: {browser.reason_code}", file=sys.stderr)
        await switch.release()
        return 2

    guard = BraveEgressGuard()
    # O acervo das fixtures é indexado pelo mesmo embedding da produção, então o
    # que chega ao modelo é a passagem que o piso do contrato deixaria passar —
    # inclusive nenhuma. Com o embedder determinístico, uma pergunta fora do
    # assunto ainda traz passagem, e o experimento mediria o modelo diante de
    # material que a produção nunca entregaria.
    embedder = (
        OllamaEmbeddingRuntime(
            base_url=OLLAMA_URL,
            model=embedding.id,
            expected_digest=embedding.digest_sha256,
            dimensions=embedding.dimensions,
        )
        if embedding is not None
        else None
    )
    model_runner = ModelCaseRunner(
        config=config,
        runtime=runtime,
        estimator=estimator,
        operator_notes=load_operator_notes(ROOT / "SYSTEM-PROMPT.md"),
        runtime_readiness=EngineReadiness(ready=True),
        browser_guard=guard,
        embedder=embedder,
        runtime_switch=switch,
    )
    live = build_live_runner(config, model_runner, browser_guard=guard)
    service = EvalService(
        store=EvalStore(database),
        catalog=catalog,
        lease=BenchmarkLease(),
        runners={EvalTier.EXPERIMENT: live, EvalTier.MODEL_SMOKE: live},
    )
    await service.initialize()
    try:
        run = await service.create_run(
            experiment_id=experiment_id,
            tier=tier,
            phase=phase,
        )
        print(
            f"run {run.id} seeds={list(run.seeds)} tier={tier.value} phase={phase.value}",
            file=sys.stderr,
        )
        # Congelado junto dos digests de contrato quando o resultado é transcrito
        # para evals/experiments.json#results[].frozen.
        print(
            f"operator_prompt_digest={model_runner.operator_prompt_digest}",
            file=sys.stderr,
        )
        print(f"runtime_profile_id={config.runtime_profile.id}", file=sys.stderr)
        await service.start(run.id)
        while True:
            current = await service.status(run.id)
            if current.status in {
                EvalRunStatus.COMPLETED,
                EvalRunStatus.BLOCKED,
                EvalRunStatus.FAILED,
                EvalRunStatus.CANCELED,
            }:
                break
            await asyncio.sleep(1)
        report = await service.report(run.id)
        print(json.dumps(report.payload, indent=2, ensure_ascii=False, sort_keys=True))
        # O tokenizer de cada perfil é arquivo do host, não do repositório; o
        # digest dele entra no congelamento do resultado ao lado do perfil.
        for profile_id, digest in sorted(switch.tokenizer_digests.items()):
            print(f"tokenizer_digest[{profile_id}]={digest}", file=sys.stderr)
        return 0 if current.status is EvalRunStatus.COMPLETED else 1
    finally:
        await service.shutdown()
        await switch.release()
        if embedder is not None:
            await embedder.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_id",
        nargs="?",
        default=EvalTier.MODEL_SMOKE.value,
        help='experiment from the manifest, or "model_smoke" for the whole corpus',
    )
    parser.add_argument("--phase", choices=[phase.value for phase in EvalPhase], default="pilot")
    parser.add_argument(
        "--tier",
        choices=[EvalTier.MODEL_SMOKE.value, EvalTier.EXPERIMENT.value],
        default=EvalTier.EXPERIMENT.value,
        help="model_smoke runs every fixture that has a runner, in one arm",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".harness/tokenizer.json",
        help="tokenizer.json of the route's RuntimeProfile, used for the context budget",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=ROOT / ".harness/tokenizers",
        help="one <runtime_profile_id>.json per bake-off profile",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="run every arm on this RuntimeProfile instead of the route's",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="EvalStore database; a temporary file by default",
    )
    arguments = parser.parse_args()
    if not arguments.tokenizer.is_file():
        print(f"tokenizer not found: {arguments.tokenizer}", file=sys.stderr)
        return 2
    if arguments.database is not None:
        return asyncio.run(
            run(
                arguments.experiment_id,
                EvalPhase(arguments.phase),
                EvalTier(arguments.tier),
                arguments.tokenizer,
                arguments.tokenizer_dir,
                arguments.profile,
                arguments.database,
            )
        )
    with tempfile.TemporaryDirectory(prefix="harness-eval-") as temporary:
        return asyncio.run(
            run(
                arguments.experiment_id,
                EvalPhase(arguments.phase),
                EvalTier(arguments.tier),
                arguments.tokenizer,
                arguments.tokenizer_dir,
                arguments.profile,
                Path(temporary) / "evals.sqlite3",
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
