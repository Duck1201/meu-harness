#!/usr/bin/env python
"""Imprime o system prompt exato que o modelo recebe hoje.

O prompt é derivado dos contratos e de fatos do host (ADR 0007): data em UTC,
visão oferecida ou não e o texto do Operator em SYSTEM-PROMPT.md. Este script
monta o prompt pelo mesmo caminho da produção.

    uv run python scripts/show-system-prompt.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import load_config  # noqa: E402
from harness.system_prompt import build_system_prompt, load_operator_notes  # noqa: E402


def main() -> None:
    config = load_config()
    print(
        build_system_prompt(
            config,
            today=datetime.now(UTC).date(),
            operator_notes=load_operator_notes(ROOT / "SYSTEM-PROMPT.md"),
            vision_tool_offered=config.vision.mode == "enabled",
        )
    )


if __name__ == "__main__":
    main()
