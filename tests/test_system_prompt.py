from datetime import date
from pathlib import Path

import pytest

from harness import load_config
from harness.config import CapabilityConfig
from harness.system_prompt import build_system_prompt, load_operator_notes

_TODAY = date(2026, 8, 11)


def test_prompt_states_the_capability_the_profile_does_not_declare() -> None:
    config = load_config()

    prompt = build_system_prompt(config, today=_TODAY, operator_notes="", vision_tool_offered=False)

    assert "workspace root" in prompt
    assert "A tool result is authoritative" in prompt
    assert "Creating a file is a single write_file call" in prompt
    # The active profile declares vision as unknown, so the model is told it cannot see.
    assert config.runtime_profile.capabilities["vision"].support == "unknown"
    assert "You cannot see images" in prompt


def test_a_declared_capability_is_not_announced_as_missing() -> None:
    config = load_config()
    profile = config.runtime_profile
    seeing = profile.model_copy(
        update={
            "capabilities": {
                **profile.capabilities,
                "vision": CapabilityConfig(
                    support="supported",
                    evidence={"kind": "local_measurement", "source": "test"},
                    gate_status="passed",
                ),
            }
        }
    )
    profiles = config.model_profiles.model_copy(update={"runtime_profiles": (seeing,)})
    sighted = config.model_copy(update={"model_profiles": profiles})

    prompt = build_system_prompt(
        sighted, today=_TODAY, operator_notes="", vision_tool_offered=False
    )

    assert "You cannot see images" not in prompt
    assert "A tool result is authoritative" in prompt


def test_prompt_carries_the_date_it_was_given_and_forbids_looking_it_up() -> None:
    config = load_config()

    prompt = build_system_prompt(config, today=_TODAY, operator_notes="", vision_tool_offered=False)

    assert "Today's date is 2026-08-11 (UTC)." in prompt
    assert "do not call a tool to look it up" in prompt
    # Nothing in the prompt reads the clock: another date in, another date out.
    assert "2026-08-11" not in build_system_prompt(
        config, today=date(2027, 1, 2), operator_notes="", vision_tool_offered=False
    )


def test_an_empty_operator_block_leaves_the_prompt_untouched() -> None:
    config = load_config()

    prompt = build_system_prompt(config, today=_TODAY, operator_notes="", vision_tool_offered=False)

    # The exact string the harness produced before SYSTEM-PROMPT.md existed: an
    # absent Operator block must not cost even a trailing space.
    assert prompt == (
        "Use the available tools when needed. All workspace paths supplied to "
        "tools must be relative to the workspace root. A tool result is "
        "authoritative. Do not call another tool to confirm what a result in this "
        "turn already reported: if a search listed the files, that is the list; "
        "if an edit reported success, the file changed. In particular, after glob "
        "or grep_search, do not read the files they named unless the request is "
        "about their contents. Creating a file is a single write_file call: a "
        "path that does not exist yet takes no expected_current_sha256 and "
        "nothing has to be read or located first. Once a tool reports a path is "
        "absent, treat it as absent — do not call the same tool again with "
        "different arguments to look for it. Today's date is 2026-08-11 (UTC). It "
        "comes from the host and is authoritative: do not derive the date from "
        "your training data and do not call a tool to look it up. You cannot see "
        "images. If a request depends on looking at one, say so plainly instead "
        "of searching the workspace or the web for it."
    )


def test_the_operator_block_is_appended_at_the_end() -> None:
    config = load_config()

    prompt = build_system_prompt(
        config, today=_TODAY, operator_notes="Prefira respostas curtas.", vision_tool_offered=False
    )

    assert prompt.endswith(" Prefira respostas curtas.")
    assert "  " not in prompt


def test_a_missing_file_yields_no_operator_notes(tmp_path: Path) -> None:
    assert load_operator_notes(tmp_path / "nao-existe.md") == ""


def test_html_comments_are_notes_for_the_reader_not_for_the_model(tmp_path: Path) -> None:
    path = tmp_path / "SYSTEM-PROMPT.md"
    path.write_text("<!--\nexplicação do arquivo\n-->\n\n<!-- outra -->\n", encoding="utf-8")

    assert load_operator_notes(path) == ""


def test_the_whole_file_is_the_operator_text(tmp_path: Path) -> None:
    path = tmp_path / "SYSTEM-PROMPT.md"
    path.write_text("<!-- nota -->\nResponda em pt-BR.\n\nSeja breve.\n", encoding="utf-8")

    assert load_operator_notes(path) == "Responda em pt-BR.\n\nSeja breve."


def test_an_oversized_operator_block_is_refused_before_the_turn(tmp_path: Path) -> None:
    path = tmp_path / "SYSTEM-PROMPT.md"
    path.write_text(f"{'a' * 5000}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="limite é 4000"):
        load_operator_notes(path)


def test_the_file_in_the_repo_carries_no_operator_text() -> None:
    assert load_operator_notes(Path("SYSTEM-PROMPT.md")) == ""
