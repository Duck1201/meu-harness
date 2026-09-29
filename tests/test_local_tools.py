import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from harness import (
    Grant,
    RegistryToolExecutor,
    SessionPolicy,
    ToolCall,
    ToolResult,
    load_config,
)
from harness.domain import JsonValue

# Os tetos de bytes vêm do contrato, não de um número solto aqui: um teste que
# lê um teto diferente do de produção mede outro executor.
_CONTEXT = load_config().context


def _matched_paths(result: ToolResult) -> list[str]:
    assert isinstance(result.data, Mapping)
    matches = cast(list[Mapping[str, JsonValue]], result.data["matches"])
    return [str(match["path"]) for match in matches]


def executor_for(
    workspace_root: Path,
    *permissions: str,
    host_denied_paths: tuple[str, ...] = (),
    max_read_bytes: int = _CONTEXT.max_tool_read_bytes,
    max_search_bytes: int = _CONTEXT.max_tool_search_bytes,
) -> RegistryToolExecutor:
    now = datetime.now(UTC)
    policy = SessionPolicy(
        conversation_id="conversation-1",
        grants=tuple(
            Grant(
                id=f"grant-{permission}",
                conversation_id="conversation-1",
                permission=permission,
                scope="workspace",
                granted_at=now,
            )
            for permission in permissions
        ),
    )
    return RegistryToolExecutor(
        registry=load_config().tool_registry,
        workspace_root=workspace_root,
        session_policy=policy,
        host_denied_paths=host_denied_paths,
        max_read_bytes=max_read_bytes,
        max_search_bytes=max_search_bytes,
    )


def test_preflight_rejects_absolute_and_parent_traversal_paths(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = executor_for(tmp_path, "WorkspaceRootGrant")

        absolute = await executor.preflight(
            (ToolCall(id="absolute", name="read_file", arguments={"file_path": "/etc/passwd"}),)
        )
        traversal = await executor.preflight(
            (ToolCall(id="traversal", name="read_file", arguments={"file_path": "../secret"}),)
        )
        nul = await executor.preflight(
            (ToolCall(id="nul", name="read_file", arguments={"file_path": "bad\x00path"}),)
        )

        assert absolute.allowed is False
        assert absolute.reason_code == "absolute_path_not_allowed"
        assert traversal.allowed is False
        assert traversal.reason_code == "path_traversal_not_allowed"
        assert nul.allowed is False
        assert nul.reason_code == "nul_in_path"

    asyncio.run(scenario())


def test_read_file_returns_exact_line_page_and_total_lines(tmp_path: Path) -> None:
    async def scenario() -> None:
        content = "zero\num\ndois\n"
        (tmp_path / "notes.txt").write_text(content, encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")
        call = ToolCall(
            id="read",
            name="read_file",
            arguments={"file_path": "notes.txt", "offset": 1, "max_lines": 1},
        )

        assert (await executor.preflight((call,))).allowed is True
        result = await executor.execute(call)

        assert result.status.value == "success"
        assert result.error is None
        assert result.data == {
            "file_path": "notes.txt",
            "content": "um\n",
            "total_lines": 3,
            "offset": 1,
            "line_count": 1,
            "next_offset": 2,
        }
        assert result.meta == {
            "producer": "local_filesystem",
            "truncated": True,
            "taints": [],
        }

    asyncio.run(scenario())


def test_read_file_byte_limit_keeps_next_line_addressable(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "limited.txt").write_bytes(b"aa\nbbbb\ncc\n")
        executor = executor_for(
            tmp_path,
            "WorkspaceRootGrant",
            max_read_bytes=5,
        )

        result = await executor.execute(
            ToolCall(
                id="limited-read",
                name="read_file",
                arguments={"file_path": "limited.txt", "max_lines": 3},
            )
        )

        assert result.status.value == "success"
        assert isinstance(result.data, Mapping)
        assert result.data["content"] == "aa\n"
        assert result.data["line_count"] == 1
        assert result.data["next_offset"] == 1
        assert result.meta["truncated"] is True

    asyncio.run(scenario())


def test_preview_shows_what_the_mutation_would_change(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        (tmp_path / "notes.md").write_bytes(b"first\nsecond\n")

        created = await executor.preview(
            ToolCall(
                id="create",
                name="write_file",
                arguments={"file_path": "new.md", "content": "hello\n"},
            )
        )
        assert created is not None
        assert (created.path, created.kind, created.truncated) == ("new.md", "create", False)
        assert "+hello" in created.diff

        replaced = await executor.preview(
            ToolCall(
                id="replace",
                name="write_file",
                arguments={"file_path": "notes.md", "content": "first\nthird\n"},
            )
        )
        assert replaced is not None
        assert replaced.kind == "replace"
        assert "-second" in replaced.diff
        assert "+third" in replaced.diff

        edited = await executor.preview(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "notes.md",
                    "old_string": "second",
                    "new_string": "changed",
                },
            )
        )
        assert edited is not None
        assert edited.kind == "edit"
        assert "+changed" in edited.diff

        # An edit that would fail has nothing honest to preview.
        assert (
            await executor.preview(
                ToolCall(
                    id="absent",
                    name="edit",
                    arguments={"file_path": "notes.md", "old_string": "zzz", "new_string": "y"},
                )
            )
            is None
        )

        # Previewing is not writing.
        assert (tmp_path / "notes.md").read_bytes() == b"first\nsecond\n"
        assert not (tmp_path / "new.md").exists()

        # Nothing to show for a read, and nothing at all for a denied path.
        assert (
            await executor.preview(
                ToolCall(id="read", name="read_file", arguments={"file_path": "notes.md"})
            )
            is None
        )
        assert (
            await executor.preview(
                ToolCall(
                    id="denied",
                    name="write_file",
                    arguments={"file_path": ".ssh/id_rsa", "content": "no"},
                )
            )
            is None
        )

    asyncio.run(scenario())


def test_empty_optional_argument_is_read_as_absent(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "a.py").write_text("TODO one\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("TODO two\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        # The runtime fills every property in the schema, this one included: an
        # empty include is no filter, not a filter that matches nothing.
        search = ToolCall(
            id="grep",
            name="grep_search",
            arguments={"pattern": "TODO", "include": ""},
        )

        assert (await executor.preflight((search,))).allowed is True
        assert _matched_paths(await executor.execute(search)) == ["a.py", "b.md"]

        # An empty required argument is a value: an empty file is a real request.
        empty_file = ToolCall(
            id="empty",
            name="write_file",
            arguments={"file_path": "blank.md", "content": ""},
        )
        assert (await executor.preflight((empty_file,))).allowed is True
        assert (await executor.execute(empty_file)).status.value == "success"
        assert (tmp_path / "blank.md").read_bytes() == b""

    asyncio.run(scenario())


def test_write_file_atomically_creates_internal_parents(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(
            id="write",
            name="write_file",
            arguments={"file_path": "nested/deep/file.txt", "content": "new content\n"},
        )

        assert (await executor.preflight((call,))).allowed is True
        result = await executor.execute(call)

        assert result.status.value == "success"
        assert result.data == {
            "file_path": "nested/deep/file.txt",
            "created": True,
            "changed": True,
            "created_directories": ["nested", "nested/deep"],
        }
        assert result.meta == {
            "producer": "local_filesystem",
            "truncated": False,
            "taints": [],
            "limitations": ["external_filesystem_toctou"],
        }
        assert (tmp_path / "nested/deep/file.txt").read_bytes() == b"new content\n"
        assert list((tmp_path / "nested/deep").glob(".*.tmp")) == []

    asyncio.run(scenario())


def _edit(tmp_path: Path, before: bytes, **arguments: JsonValue) -> tuple[ToolResult, bytes]:
    async def scenario() -> ToolResult:
        (tmp_path / "target.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(id="edit", name="edit", arguments={"file_path": "target.txt", **arguments})
        assert (await executor.preflight((call,))).allowed is True
        return await executor.execute(call)

    result = asyncio.run(scenario())
    return result, (tmp_path / "target.txt").read_bytes()


def test_edit_replaces_exact_text_and_preserves_every_other_byte(tmp_path: Path) -> None:
    result, after = _edit(
        tmp_path,
        b"alpha\r\n  beta  \r\ngamma\nomega",
        old_string="  beta  ",
        new_string=" replacement ",
    )

    assert result.status.value == "success"
    assert after == b"alpha\r\n replacement \r\ngamma\nomega"
    assert isinstance(result.data, Mapping)
    assert result.data["changed"] is True
    assert result.data["replacements"] == 1


def test_edit_shows_the_model_the_region_it_wrote(tmp_path: Path) -> None:
    # Without it the model had no way to see a broken edit and reported success.
    before = "".join(f"line {number}\n" for number in range(1, 11)).encode()
    result, _ = _edit(tmp_path, before, old_string="line 5\n", new_string="LINE FIVE\n")

    assert isinstance(result.data, Mapping)
    assert result.data["excerpt"] == {
        "start_line": 3,
        "text": "line 3\nline 4\nLINE FIVE\nline 6\nline 7",
    }


def test_edit_that_misses_says_where_the_text_parted_from_the_file(tmp_path: Path) -> None:
    result, after = _edit(
        tmp_path,
        b"function run() {\n  return 1;\n}\n",
        old_string="function run() {\n    return 1;",
        new_string="function run() {\n  return 2;",
    )

    assert result.status.value == "failed"
    assert result.retryable is True
    assert result.error is not None
    assert result.error["code"] == "old_string_not_found"
    assert "line 1" in str(result.error["message"])
    assert after == b"function run() {\n  return 1;\n}\n"


def test_edit_refuses_an_ambiguous_match_unless_asked_to_replace_all(tmp_path: Path) -> None:
    before = b"name = 'Seu Nome'\ntitle = 'Seu Nome'\n"
    ambiguous, unchanged = _edit(tmp_path, before, old_string="Seu Nome", new_string="Miguel")
    everywhere, replaced = _edit(
        tmp_path, before, old_string="Seu Nome", new_string="Miguel", replace_all=True
    )

    assert ambiguous.status.value == "failed"
    assert ambiguous.error is not None
    assert ambiguous.error["code"] == "old_string_not_unique"
    assert "lines 1, 2" in str(ambiguous.error["message"])
    assert unchanged == before
    assert everywhere.status.value == "success"
    assert isinstance(everywhere.data, Mapping)
    assert everywhere.data["replacements"] == 2
    assert replaced == b"name = 'Miguel'\ntitle = 'Miguel'\n"


def test_edit_with_identical_strings_is_refused_before_it_runs(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        batch = await executor.preflight(
            (
                ToolCall(
                    id="same",
                    name="edit",
                    arguments={"file_path": "a.txt", "old_string": "x", "new_string": "x"},
                ),
            )
        )
        assert batch.allowed is False
        assert batch.reason_code == "invalid_tool_arguments"

    asyncio.run(scenario())


_APP_JS = b"function run(ok, value) {\n  if (ok) {\n    return value;\n  }\n}\n"
_TABBED_APP_JS = b"function run(ok, value) {\n\tif (ok) {\n\t  return value;\n\t}\n}\n"
_APP_BODY = "  if (ok) {\n    return value;\n  }\n"


def test_edit_keeps_the_line_break_the_replacement_left_out(tmp_path: Path) -> None:
    """Observed in 5 of 6 Gemma 4 seeds: the block arrives without its final break.

    Refusing cost the Turn its second call, and the model then told the Operator
    the edit had gone through. The break is only restored when a line follows —
    at the end of the file dropping it is a real edit — so there is no line the
    model could have meant to weld on.
    """
    result, after = _edit(
        tmp_path, _APP_JS, old_string=_APP_BODY, new_string="\tif (ok) {\n\t  return value;\n\t}"
    )

    assert result.status.value == "success"
    assert after == _TABBED_APP_JS


def test_edit_keeps_a_crlf_break_the_replacement_left_out(tmp_path: Path) -> None:
    result, after = _edit(
        tmp_path, b"alpha\r\nbeta\r\ngamma\r\n", old_string="beta\r\n", new_string="BETA"
    )

    assert result.status.value == "success"
    assert after == b"alpha\r\nBETA\r\ngamma\r\n"


def test_edit_deleting_lines_does_not_leave_a_blank_one(tmp_path: Path) -> None:
    result, after = _edit(tmp_path, b"keep\ndrop\nkeep\n", old_string="drop\n", new_string="")

    assert result.status.value == "success"
    assert after == b"keep\nkeep\n"


def test_edit_decodes_a_block_the_model_escaped_twice(tmp_path: Path) -> None:
    """Observed in 3 of 6 Gemma 4 seeds: backslash-t and backslash-n as text.

    The model reads the file as escaped JSON and writes the escapes back. A
    multi-line old_string replaced by one line whose only breaks are escapes is
    that mistake, not a line that holds a string literal.
    """
    result, after = _edit(
        tmp_path,
        _APP_JS,
        old_string=_APP_BODY,
        new_string="\\tif (ok) {\\n\\t  return value;\\n\\t}",
    )

    assert result.status.value == "success"
    assert after == _TABBED_APP_JS


def test_edit_finds_an_old_string_the_model_escaped_twice(tmp_path: Path) -> None:
    # Both halves escaped: old_string only matches the file once decoded.
    result, after = _edit(
        tmp_path,
        _APP_JS,
        old_string="  if (ok) {\\n    return value;\\n  }\\n",
        new_string="\\tif (ok) {\\n\\t  return value;\\n\\t}\\n",
    )

    assert result.status.value == "success"
    assert after == _TABBED_APP_JS


def test_the_preview_shows_the_repaired_block_the_edit_will_write(tmp_path: Path) -> None:
    # The Operator approves the preview: it cannot show backslash-t while the
    # executor writes a tab.
    async def scenario() -> str:
        (tmp_path / "app.js").write_bytes(_APP_JS)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        preview = await executor.preview(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "app.js",
                    "old_string": _APP_BODY,
                    "new_string": "\\tif (ok) {\\n\\t  return value;\\n\\t}",
                },
            )
        )
        assert preview is not None
        return preview.diff

    diff = asyncio.run(scenario())

    assert "+\tif (ok) {" in diff
    assert "\\t" not in diff
    assert "+}" not in diff


def test_edit_keeps_escapes_that_belong_to_the_code(tmp_path: Path) -> None:
    # One line replacing one line: an escape there is the code's own string literal.
    result, after = _edit(
        tmp_path,
        b"a = 1\nprint(a)\nz = 2\n",
        old_string="print(a)",
        new_string='print("a\\tb\\n")',
    )

    assert result.status.value == "success"
    assert after == b'a = 1\nprint("a\\tb\\n")\nz = 2\n'


def test_edit_of_the_last_line_may_drop_the_final_newline(tmp_path: Path) -> None:
    # Nothing follows the last line, so there is no line to weld and the edit stands.
    result, after = _edit(tmp_path, b"alpha\nomega\n", old_string="omega\n", new_string="omega")

    assert result.status.value == "success"
    assert after == b"alpha\nomega"


def test_duplicate_edit_reconciles_postcondition_without_reapplying(tmp_path: Path) -> None:
    async def scenario() -> None:
        before = b"one\ntwo\nthree\n"
        after = b"one\nchanged\nthree\n"
        (tmp_path / "lines.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(
            id="edit-once",
            name="edit",
            arguments={"file_path": "lines.txt", "old_string": "two", "new_string": "changed"},
            idempotency_key="stable-edit",
        )

        first = await executor.execute(call)
        replay = await executor.execute(call)

        assert first.status.value == "success"
        assert replay.status.value == "success"
        assert replay.data == first.data
        assert replay.meta == {
            **first.meta,
            "replayed": True,
        }
        assert (tmp_path / "lines.txt").read_bytes() == after

    asyncio.run(scenario())


def test_write_replay_requires_recorded_postcondition(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(
            id="write-once",
            name="write_file",
            arguments={"file_path": "replay.txt", "content": "committed\n"},
            idempotency_key="stable-write",
        )

        first = await executor.execute(call)
        replay = await executor.execute(call)
        (tmp_path / "replay.txt").write_text("external change\n", encoding="utf-8")
        conflict = await executor.execute(call)

        assert first.status.value == "success"
        assert replay.status.value == "success"
        assert replay.meta["replayed"] is True
        assert conflict.status.value == "failed"
        assert conflict.error is not None
        assert conflict.error["code"] == "replay_postcondition_conflict"
        assert (tmp_path / "replay.txt").read_text(encoding="utf-8") == "external change\n"

    asyncio.run(scenario())


def test_list_directory_filters_sensitive_paths_and_pages_sorted_entries(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / "c.txt").write_text("c", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b-dir").mkdir()
        (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")
        call = ToolCall(
            id="list",
            name="list_directory",
            arguments={"directory_path": ".", "limit": 2},
        )

        result = await executor.execute(call)
        second_page = await executor.execute(
            ToolCall(
                id="list-page-2",
                name="list_directory",
                arguments={"directory_path": ".", "offset": 2, "limit": 2},
            )
        )

        assert result.status.value == "success"
        assert result.data == {
            "directory_path": ".",
            "entries": [
                {"name": "a.txt", "path": "a.txt", "type": "file"},
                {"name": "b-dir", "path": "b-dir", "type": "directory"},
            ],
            "offset": 0,
            "next_offset": 2,
        }
        assert result.meta["truncated"] is True
        assert isinstance(second_page.data, Mapping)
        assert second_page.data["entries"] == [{"name": "c.txt", "path": "c.txt", "type": "file"}]
        assert second_page.data["next_offset"] is None

    asyncio.run(scenario())


def test_glob_is_sorted_limited_and_does_not_recurse_through_symlinks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / "src/nested").mkdir(parents=True)
        (tmp_path / "src/z.py").write_text("z", encoding="utf-8")
        (tmp_path / "src/a.py").write_text("a", encoding="utf-8")
        (tmp_path / "src/nested/b.py").write_text("b", encoding="utf-8")
        (tmp_path / "target").mkdir()
        (tmp_path / "target/through-link.py").write_text("hidden", encoding="utf-8")
        (tmp_path / "src/link").symlink_to(tmp_path / "target", target_is_directory=True)
        executor = executor_for(tmp_path, "WorkspaceRootGrant")
        call = ToolCall(
            id="glob",
            name="glob",
            arguments={"directory_path": "src", "pattern": "**/*.py", "limit": 2},
        )

        result = await executor.execute(call)

        assert result.status.value == "success"
        assert result.data == {
            "directory_path": "src",
            "pattern": "**/*.py",
            "paths": ["src/a.py", "src/nested/b.py"],
            "offset": 0,
            "next_offset": 2,
        }
        assert result.meta["truncated"] is True

    asyncio.run(scenario())


def test_grep_search_is_sorted_limited_and_does_not_recurse_through_symlinks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs/z.txt").write_text("needle z\n", encoding="utf-8")
        (tmp_path / "docs/a.txt").write_text("needle a\nneedle b\n", encoding="utf-8")
        (tmp_path / "linked").mkdir()
        (tmp_path / "linked/hidden.txt").write_text("needle hidden\n", encoding="utf-8")
        (tmp_path / "docs/link").symlink_to(tmp_path / "linked", target_is_directory=True)
        executor = executor_for(tmp_path, "WorkspaceRootGrant")
        call = ToolCall(
            id="grep",
            name="grep_search",
            arguments={"directory_path": "docs", "pattern": "needle", "limit": 2},
        )

        result = await executor.execute(call)

        assert result.status.value == "success"
        assert result.data == {
            "directory_path": "docs",
            "pattern": "needle",
            "is_regex": False,
            "matches": [
                {"path": "docs/a.txt", "line_number": 1, "line": "needle a"},
                {"path": "docs/a.txt", "line_number": 2, "line": "needle b"},
            ],
            "offset": 0,
            "next_offset": 2,
        }
        assert result.meta["truncated"] is True

    asyncio.run(scenario())


def test_grep_search_byte_limit_reports_remaining_matches(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "matches.txt").write_text("abcde\nlater\n", encoding="utf-8")
        executor = executor_for(
            tmp_path,
            "WorkspaceRootGrant",
            max_search_bytes=5,
        )

        result = await executor.execute(
            ToolCall(
                id="byte-limited-grep",
                name="grep_search",
                arguments={"pattern": "a", "limit": 2},
            )
        )

        assert isinstance(result.data, Mapping)
        assert result.data["matches"] == [
            {"path": "matches.txt", "line_number": 1, "line": "abcde"}
        ]
        assert result.data["next_offset"] == 1
        assert result.meta["truncated"] is True

    asyncio.run(scenario())


def test_read_symlink_must_resolve_inside_workspace_and_mutations_reject_symlinks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        inside = workspace / "inside.txt"
        inside.write_text("inside", encoding="utf-8")
        (workspace / "inside-dir").mkdir()
        (workspace / "outside-link").symlink_to(outside)
        (workspace / "inside-link").symlink_to(inside)
        (workspace / "inside-dir-link").symlink_to(
            workspace / "inside-dir", target_is_directory=True
        )
        executor = executor_for(workspace, "WorkspaceRootGrant", "WriteGrant")

        outside_read = await executor.preflight(
            (
                ToolCall(
                    id="outside-read",
                    name="read_file",
                    arguments={"file_path": "outside-link"},
                ),
            )
        )
        inside_read_call = ToolCall(
            id="inside-read",
            name="read_file",
            arguments={"file_path": "inside-link"},
        )
        inside_read = await executor.execute(inside_read_call)
        symlink_write = await executor.preflight(
            (
                ToolCall(
                    id="symlink-write",
                    name="write_file",
                    arguments={"file_path": "inside-link", "content": "changed"},
                ),
            )
        )
        symlink_parent_write = await executor.preflight(
            (
                ToolCall(
                    id="symlink-parent-write",
                    name="write_file",
                    arguments={"file_path": "inside-dir-link/new.txt", "content": "new"},
                ),
            )
        )

        assert outside_read.allowed is False
        assert outside_read.reason_code == "path_outside_workspace"
        assert inside_read.status.value == "success"
        assert symlink_write.allowed is False
        assert symlink_write.reason_code == "symlink_mutation_denied"
        assert symlink_parent_write.reason_code == "symlink_mutation_denied"
        assert inside.read_text(encoding="utf-8") == "inside"

    asyncio.run(scenario())


def test_default_sensitive_paths_are_denied_but_env_examples_are_readable(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / ".env.example").write_text("NAME=value\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")

        for index, path in enumerate(
            (
                ".env",
                ".env.local",
                ".git/config",
                ".aws/credentials",
                "id_ed25519",
                "server.pem",
            )
        ):
            result = await executor.preflight(
                (
                    ToolCall(
                        id=f"sensitive-{index}",
                        name="read_file",
                        arguments={"file_path": path},
                    ),
                )
            )
            assert result.allowed is False
            assert result.reason_code == "sensitive_path_denied"

        example = await executor.execute(
            ToolCall(
                id="env-example",
                name="read_file",
                arguments={"file_path": ".env.example"},
            )
        )
        assert example.status.value == "success"

    asyncio.run(scenario())


def test_host_path_denies_are_additive_to_default_sensitive_policy(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "generated").mkdir()
        executor = executor_for(
            tmp_path,
            "WorkspaceRootGrant",
            host_denied_paths=("generated",),
        )

        host_denied = await executor.preflight(
            (
                ToolCall(
                    id="host-denied",
                    name="read_file",
                    arguments={"file_path": "generated/file.txt"},
                ),
            )
        )
        default_denied = await executor.preflight(
            (
                ToolCall(
                    id="default-denied",
                    name="read_file",
                    arguments={"file_path": ".env"},
                ),
            )
        )

        assert host_denied.reason_code == "host_path_denied"
        assert default_denied.reason_code == "sensitive_path_denied"

    asyncio.run(scenario())


def test_preflight_validates_full_batch_schema_catalog_and_effect_grants(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        read_only = executor_for(tmp_path, "WorkspaceRootGrant")
        missing_write_grant = await read_only.preflight(
            (
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"file_path": "new.txt", "content": "new"},
                ),
            )
        )
        unknown = await read_only.preflight(
            (ToolCall(id="unknown", name="remove_file", arguments={}),)
        )

        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        valid_write = ToolCall(
            id="valid-write",
            name="write_file",
            arguments={"file_path": "must-not-exist.txt", "content": "content"},
        )
        invalid_later_call = ToolCall(
            id="invalid-read",
            name="read_file",
            arguments={"file_path": "anything.txt", "max_lines": 0, "extra": True},
        )
        batch = await executor.preflight((valid_write, invalid_later_call))

        assert missing_write_grant.allowed is False
        assert missing_write_grant.reason_code == "write_grant_required"
        assert unknown.allowed is False
        assert unknown.reason_code == "unknown_tool"
        assert batch.allowed is False
        assert batch.reason_code == "invalid_tool_arguments"
        assert not (tmp_path / "must-not-exist.txt").exists()

    asyncio.run(scenario())


def test_expired_session_grants_are_not_effective(tmp_path: Path) -> None:
    async def scenario() -> None:
        now = datetime.now(UTC)
        policy = SessionPolicy(
            conversation_id="conversation-1",
            grants=(
                Grant(
                    id="workspace",
                    conversation_id="conversation-1",
                    permission="WorkspaceRootGrant",
                    scope="workspace",
                    granted_at=now,
                ),
                Grant(
                    id="expired-write",
                    conversation_id="conversation-1",
                    permission="WriteGrant",
                    scope="workspace",
                    granted_at=now - timedelta(hours=2),
                    expires_at=now - timedelta(hours=1),
                ),
            ),
        )
        executor = RegistryToolExecutor(
            registry=load_config().tool_registry,
            workspace_root=tmp_path,
            session_policy=policy,
            max_read_bytes=_CONTEXT.max_tool_read_bytes,
            max_search_bytes=_CONTEXT.max_tool_search_bytes,
        )

        result = await executor.preflight(
            (
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"file_path": "new.txt", "content": "new"},
                ),
            )
        )

        assert result.allowed is False
        assert result.reason_code == "write_grant_required"

    asyncio.run(scenario())


def test_write_file_replaces_without_a_digest_and_detects_byte_noop(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "existing.txt"
        path.write_bytes(b"same\n")
        path.chmod(0o640)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")

        noop_result = await executor.execute(
            ToolCall(
                id="noop",
                name="write_file",
                arguments={"file_path": "existing.txt", "content": "same\n"},
            )
        )
        replace_call = ToolCall(
            id="replace",
            name="write_file",
            arguments={"file_path": "existing.txt", "content": "replaced\n"},
        )
        assert (await executor.preflight((replace_call,))).allowed is True
        replace_result = await executor.execute(replace_call)

        assert noop_result.status.value == "success"
        assert noop_result.data == {
            "file_path": "existing.txt",
            "created": False,
            "changed": False,
            "created_directories": [],
        }
        assert replace_result.status.value == "success"
        assert path.read_bytes() == b"replaced\n"
        assert path.stat().st_mode & 0o777 == 0o640

    asyncio.run(scenario())


def test_workspace_reads_are_never_cached_after_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "fresh.txt"
        path.write_text("before\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        read_call = ToolCall(
            id="same-read",
            name="read_file",
            arguments={"file_path": "fresh.txt"},
        )

        before = await executor.execute(read_call)
        await executor.execute(
            ToolCall(
                id="write-fresh",
                name="write_file",
                arguments={"file_path": "fresh.txt", "content": "after\n"},
            )
        )
        after = await executor.execute(read_call)

        assert before.data != after.data
        assert isinstance(after.data, Mapping)
        assert after.data["content"] == "after\n"

    asyncio.run(scenario())


def test_not_found_and_invalid_utf8_are_failed_without_host_paths(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")

        missing = await executor.execute(
            ToolCall(
                id="missing",
                name="read_file",
                arguments={"file_path": "missing.txt"},
            )
        )
        invalid = await executor.execute(
            ToolCall(
                id="invalid",
                name="read_file",
                arguments={"file_path": "binary.dat"},
            )
        )

        assert missing.status.value == "failed"
        assert missing.error is not None and missing.error["code"] == "path_not_found"
        assert invalid.status.value == "failed"
        assert invalid.error is not None and invalid.error["code"] == "invalid_utf8"
        assert str(tmp_path) not in str(missing.error)
        assert str(tmp_path) not in str(invalid.error)

    asyncio.run(scenario())


def test_not_found_names_the_real_paths_the_model_almost_typed(tmp_path: Path) -> None:
    # A real Turn: "vibe-coder/nota-fiscal" for "Vibe coder/nota.avif", three
    # guesses, and the Turn ended without the file.
    async def scenario() -> None:
        (tmp_path / "Vibe coder").mkdir()
        (tmp_path / "Vibe coder" / "nota.avif").write_bytes(b"\x00")
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")

        file_guess = await executor.execute(
            ToolCall(id="f", name="read_file", arguments={"file_path": "vibe-coder/nota-fiscal"})
        )
        directory_guess = await executor.execute(
            ToolCall(id="d", name="list_directory", arguments={"directory_path": "vibe coder"})
        )
        secret_guess = await executor.execute(
            ToolCall(id="s", name="read_file", arguments={"file_path": "env"})
        )

        assert file_guess.error is not None
        assert "Vibe coder/nota.avif" in str(file_guess.error["message"])
        assert directory_guess.error is not None
        assert "Vibe coder" in str(directory_guess.error["message"])
        # A denied path is never offered as a suggestion.
        assert secret_guess.error is not None
        assert ".env" not in str(secret_guess.error["message"])

    asyncio.run(scenario())


def test_reading_an_image_points_to_describe_image(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "print.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")
        result = await executor.execute(
            ToolCall(id="img", name="read_file", arguments={"file_path": "print.png"})
        )

        assert result.error is not None
        assert result.error["code"] == "invalid_utf8"
        assert "describe_image" in str(result.error["message"])

    asyncio.run(scenario())


def test_grep_search_filters_by_file_pattern_and_can_ignore_case(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("Token = 1\n", encoding="utf-8")
        (tmp_path / "notes.md").write_text("token here\n", encoding="utf-8")
        executor = executor_for(tmp_path, "WorkspaceRootGrant")

        async def paths(**arguments: JsonValue) -> list[str]:
            return _matched_paths(
                await executor.execute(
                    ToolCall(
                        id="g", name="grep_search", arguments={"pattern": "token", **arguments}
                    )
                )
            )

        assert await paths() == ["notes.md"]
        assert await paths(ignore_case=True) == ["notes.md", "src/app.py"]
        assert await paths(ignore_case=True, include="*.py") == ["src/app.py"]
        assert await paths(ignore_case=True, include="src/**/*.py") == ["src/app.py"]

    asyncio.run(scenario())


def test_calculate_needs_no_grant_and_evaluates_arithmetic_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        # No grant at all: pure_compute demands none, and the gate reads the effect.
        executor = executor_for(tmp_path)

        answer = await executor.execute(
            ToolCall(id="c1", name="calculate", arguments={"expression": "2 + 3 * (10 - 4) / 2"})
        )
        divide = await executor.execute(
            ToolCall(id="c2", name="calculate", arguments={"expression": "1/0"})
        )
        smuggled = await executor.execute(
            ToolCall(id="c3", name="calculate", arguments={"expression": "__import__('os').sep"})
        )
        conditional = await executor.execute(
            ToolCall(id="c4", name="calculate", arguments={"expression": "1 if 2 else 3"})
        )

        assert answer.status.value == "success"
        assert isinstance(answer.data, Mapping)
        assert answer.data["result"] == 11
        assert answer.meta["producer"] == "local_compute"
        assert divide.status.value == "failed"
        assert divide.error is not None and divide.error["code"] == "division_by_zero"
        assert smuggled.status.value == "blocked"
        assert smuggled.error is not None
        assert smuggled.error["code"] == "invalid_tool_arguments"
        assert conditional.status.value == "blocked"

    asyncio.run(scenario())


def test_calculate_refuses_an_exponent_that_would_hang_the_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = executor_for(tmp_path)

        huge = await executor.execute(
            ToolCall(id="c1", name="calculate", arguments={"expression": "9 ** 9 ** 9"})
        )

        assert huge.status.value == "blocked"
        assert huge.error is not None and huge.error["code"] == "expression_too_large"

    asyncio.run(scenario())
