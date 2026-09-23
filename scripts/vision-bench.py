#!/usr/bin/env python
"""Mede o modelo de visão do contrato nas imagens de evals/vision (ADR 0016).

Usa o mesmo `OllamaVisionRuntime` que `describe_image` usa em produção — mesmo
prompt, temperatura e seed —, então o número é a leitura que o Operator vai
receber, no digest que `config/harness.json#vision` fixa.

    uv run python scripts/vision-bench.py
    uv run python scripts/vision-bench.py --model minicpm-v4.6:1b --digest <sha256>
    uv run python scripts/vision-bench.py --cases ~/harness-prints

`--cases` aponta para outra pasta com o mesmo formato (`cases.json` e as
imagens). É por onde entram os prints reais do Operator, que ficam fora do
repositório quando mostram gente ou dado pessoal.

Cada caso tem gabarito exato com formas equivalentes (`24,7` para `24.7`). Os
casos de honestidade passam quando o modelo diz que não consegue ler, e falham se
ele escreve o que não está legível.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import load_config  # noqa: E402
from harness.vision_runtime import OllamaVisionRuntime  # noqa: E402

CASES = ROOT / "evals/vision"
_IGNORANCE = (
    "não consigo",
    "não é possível",
    "não foi possível",
    "ilegível",
    "não há",
    "não aparece",
    "não está",
    "não indica",
    "não mostra",
    "não contém",
    "não existe",
    "impossível",
    "não dá para",
    "não posso",
    "não é legível",
    "não identifico",
)
_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")


def _as_read(text: str) -> str:
    return _ESCAPE.sub(r"\1", text).replace("**", "").replace("`", "").lower()


def judge(case: dict[str, object], answer: str) -> bool:
    text = _as_read(answer)
    forbid = [str(item).lower() for item in case["forbid"]]  # type: ignore[union-attr]
    if any(item in text for item in forbid):
        return False
    if case["honesty"]:
        return any(phrase in text for phrase in _IGNORANCE)
    expect = case["expect"]
    return all(
        any(str(form).lower() in text for form in forms)  # type: ignore[union-attr]
        for forms in expect  # type: ignore[union-attr]
    )


async def run(model: str, digest: str, directory: Path) -> int:
    settings = load_config().vision
    runtime = OllamaVisionRuntime(
        base_url="http://127.0.0.1:11434",
        model=model,
        expected_digest=digest,
        max_output_tokens=settings.max_output_tokens,
        context_tokens=settings.context_tokens,
    )
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    passed = 0
    latencies: list[float] = []
    try:
        for case in cases:
            image = (directory / case["image"]).read_bytes()
            answer = await runtime.describe(image, case["question"])
            ok = judge(case, answer.text)
            passed += ok
            latencies.append(answer.latency_ms)
            mark = "ok " if ok else "ERR"
            print(f"{mark} {case['id']:9} {answer.latency_ms:7.0f} ms  {answer.text[:120]!r}")
    finally:
        await runtime.aclose()
    latencies.sort()
    print(
        json.dumps(
            {
                "model": model,
                "digest": digest,
                "passed": f"{passed}/{len(cases)}",
                "median_latency_ms": latencies[len(latencies) // 2],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed == len(cases) else 1


def main() -> int:
    settings = load_config().vision
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=settings.ollama_tag)
    parser.add_argument("--digest", default=settings.ollama_digest)
    parser.add_argument("--cases", type=Path, default=CASES)
    arguments = parser.parse_args()
    return asyncio.run(run(arguments.model, arguments.digest, arguments.cases.expanduser()))


if __name__ == "__main__":
    raise SystemExit(main())
