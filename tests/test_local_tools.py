import asyncio
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness import (
    Grant,
    RegistryToolExecutor,
    SessionPolicy,
    ToolCall,
    ToolResult,
    load_config,
)

# Os tetos de bytes vêm do contrato, não de um número solto aqui: um teste que
# lê um teto diferente do de produção mede outro executor.
_CONTEXT = load_config().context


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


def test_read_file_returns_exact_line_page_and_content_digest(tmp_path: Path) -> None:
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
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
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
        digest = hashlib.sha256(b"first\nsecond\n").hexdigest()

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
                arguments={
                    "file_path": "notes.md",
                    "content": "first\nthird\n",
                    "expected_current_sha256": digest,
                },
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
                    "start_line": 2,
                    "end_line": 2,
                    "replacement": "changed\n",
                    "expected_current_sha256": digest,
                },
            )
        )
        assert edited is not None
        assert edited.kind == "edit"
        assert "+changed" in edited.diff

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
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        # The runtime fills every property in the schema, this one included.
        create = ToolCall(
            id="write",
            name="write_file",
            arguments={
                "file_path": "notes.md",
                "content": "first\n",
                "expected_current_sha256": "",
            },
        )

        assert (await executor.preflight((create,))).allowed is True
        assert (await executor.execute(create)).status.value == "success"
        assert (tmp_path / "notes.md").read_bytes() == b"first\n"

        # Absent is absent: replacing the file it just created still needs the digest.
        replace_call = ToolCall(
            id="replace",
            name="write_file",
            arguments={
                "file_path": "notes.md",
                "content": "second\n",
                "expected_current_sha256": "",
            },
        )
        batch = await executor.preflight((replace_call,))
        assert batch.allowed is False
        assert batch.reason_code == "expected_sha256_required"
        assert (tmp_path / "notes.md").read_bytes() == b"first\n"

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

        digest = hashlib.sha256(b"new content\n").hexdigest()
        assert result.status.value == "success"
        assert result.data == {
            "file_path": "nested/deep/file.txt",
            "before_sha256": None,
            "after_sha256": digest,
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


def test_edit_preserves_bytes_outside_inclusive_line_range(tmp_path: Path) -> None:
    async def scenario() -> None:
        before = b"alpha\r\n  beta  \r\ngamma\nomega"
        expected_after = b"alpha\r\n replacement \ngamma\nomega"
        (tmp_path / "mixed.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(
            id="edit",
            name="edit",
            arguments={
                "file_path": "mixed.txt",
                "start_line": 2,
                "end_line": 2,
                "replacement": " replacement \n",
                "expected_current_sha256": hashlib.sha256(before).hexdigest(),
            },
        )

        assert (await executor.preflight((call,))).allowed is True
        result = await executor.execute(call)

        assert result.status.value == "success"
        assert result.data == {
            "file_path": "mixed.txt",
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(expected_after).hexdigest(),
            "changed": True,
            "created_directories": [],
        }
        assert (tmp_path / "mixed.txt").read_bytes() == expected_after

    asyncio.run(scenario())


_APP_JS = b"function run(ok, value) {\n  if (ok) {\n    return value;\n  }\n}\n"
_TABBED_APP_JS = b"function run(ok, value) {\n\tif (ok) {\n\t  return value;\n\t}\n}\n"


def _edit_app_js(tmp_path: Path, replacement: str) -> tuple[ToolResult, bytes]:
    async def scenario() -> ToolResult:
        (tmp_path / "app.js").write_bytes(_APP_JS)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        return await executor.execute(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "app.js",
                    "start_line": 2,
                    "end_line": 4,
                    "replacement": replacement,
                    "expected_current_sha256": hashlib.sha256(_APP_JS).hexdigest(),
                },
            )
        )

    result = asyncio.run(scenario())
    return result, (tmp_path / "app.js").read_bytes()


def test_edit_keeps_the_line_break_the_replacement_left_out(tmp_path: Path) -> None:
    """Observed in 5 of 6 Gemma 4 seeds: the block arrives without its final break.

    Refusing cost the Turn its second call, and the model then told the Operator
    the edit had gone through. The break is only restored when a line follows —
    at the end of the file dropping it is a real edit — so there is no line the
    model could have meant to weld on.
    """
    result, after = _edit_app_js(tmp_path, "\tif (ok) {\n\t  return value;\n\t}")

    assert result.status.value == "success"
    assert after == _TABBED_APP_JS


def test_edit_keeps_a_crlf_break_the_replacement_left_out(tmp_path: Path) -> None:
    async def scenario() -> ToolResult:
        before = b"alpha\r\nbeta\r\ngamma\r\n"
        (tmp_path / "crlf.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        return await executor.execute(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "crlf.txt",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement": "BETA",
                    "expected_current_sha256": hashlib.sha256(before).hexdigest(),
                },
            )
        )

    assert asyncio.run(scenario()).status.value == "success"
    assert (tmp_path / "crlf.txt").read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


def test_edit_decodes_a_block_the_model_escaped_twice(tmp_path: Path) -> None:
    """Observed in 3 of 6 Gemma 4 seeds: backslash-t and backslash-n as text.

    The model reads the file as escaped JSON and writes the escapes back. A
    multi-line range replaced by one line whose only breaks are escapes is that
    mistake, not a line that holds a string literal.
    """
    result, after = _edit_app_js(tmp_path, "\\tif (ok) {\\n\\t  return value;\\n\\t}")

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
                    "start_line": 2,
                    "end_line": 4,
                    "replacement": "\\tif (ok) {\\n\\t  return value;\\n\\t}",
                    "expected_current_sha256": hashlib.sha256(_APP_JS).hexdigest(),
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
    async def scenario() -> ToolResult:
        before = b"a = 1\nprint(a)\nz = 2\n"
        (tmp_path / "s.py").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        return await executor.execute(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "s.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement": 'print("a\\tb\\n")\n',
                    "expected_current_sha256": hashlib.sha256(before).hexdigest(),
                },
            )
        )

    assert asyncio.run(scenario()).status.value == "success"
    assert (tmp_path / "s.py").read_bytes() == b'a = 1\nprint("a\\tb\\n")\nz = 2\n'


def test_edit_of_the_last_line_may_drop_the_final_newline(tmp_path: Path) -> None:
    async def scenario() -> None:
        before = b"alpha\nomega\n"
        (tmp_path / "tail.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")

        result = await executor.execute(
            ToolCall(
                id="edit",
                name="edit",
                arguments={
                    "file_path": "tail.txt",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement": "omega",
                    "expected_current_sha256": hashlib.sha256(before).hexdigest(),
                },
            )
        )

        # Nothing follows the last line, so there is no line to weld and the edit stands.
        assert result.status.value == "success"
        assert (tmp_path / "tail.txt").read_bytes() == b"alpha\nomega"

    asyncio.run(scenario())


def test_duplicate_edit_reconciles_postcondition_without_reapplying(tmp_path: Path) -> None:
    async def scenario() -> None:
        before = b"one\ntwo\nthree\n"
        after = b"one\nchanged\nthree\n"
        (tmp_path / "lines.txt").write_bytes(before)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        call = ToolCall(
            id="edit-once",
            name="edit",
            arguments={
                "file_path": "lines.txt",
                "start_line": 2,
                "end_line": 2,
                "replacement": "changed\n",
                "expected_current_sha256": hashlib.sha256(before).hexdigest(),
            },
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
                    arguments={
                        "file_path": "inside-link",
                        "content": "changed",
                        "expected_current_sha256": hashlib.sha256(b"inside").hexdigest(),
                    },
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


def test_write_file_rejects_stale_sha_and_detects_byte_noop(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "existing.txt"
        path.write_bytes(b"same\n")
        path.chmod(0o640)
        executor = executor_for(tmp_path, "WorkspaceRootGrant", "WriteGrant")
        stale = ToolCall(
            id="stale",
            name="write_file",
            arguments={
                "file_path": "existing.txt",
                "content": "different\n",
                "expected_current_sha256": "0" * 64,
            },
        )

        stale_result = await executor.execute(stale)
        digest = hashlib.sha256(b"same\n").hexdigest()
        noop_result = await executor.execute(
            ToolCall(
                id="noop",
                name="write_file",
                arguments={
                    "file_path": "existing.txt",
                    "content": "same\n",
                    "expected_current_sha256": digest,
                },
            )
        )
        replace_result = await executor.execute(
            ToolCall(
                id="replace",
                name="write_file",
                arguments={
                    "file_path": "existing.txt",
                    "content": "replaced\n",
                    "expected_current_sha256": digest,
                },
            )
        )

        assert stale_result.status.value == "failed"
        assert stale_result.retryable is True
        assert stale_result.error == {
            "code": "content_conflict",
            "message": "The current file digest does not match expected_current_sha256.",
        }
        assert noop_result.status.value == "success"
        assert noop_result.data == {
            "file_path": "existing.txt",
            "before_sha256": digest,
            "after_sha256": digest,
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
                arguments={
                    "file_path": "fresh.txt",
                    "content": "after\n",
                    "expected_current_sha256": hashlib.sha256(b"before\n").hexdigest(),
                },
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
