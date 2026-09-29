from __future__ import annotations

import ast
import difflib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

from jsonschema import Draft202012Validator

from .config import ReplayPolicy, ToolDefinitionConfig, ToolRegistryConfig
from .domain import SessionPolicy, ToolCall, ToolResult, ToolResultStatus, grant_reason_code
from .ports import ConfirmationPreview, ToolBatchPreflight
from .workspace_paths import PathPolicyError as _PreflightIssue
from .workspace_paths import WorkspacePathPolicy
from .workspace_paths import relative_parts as _relative_parts

_WORKSPACE_EFFECTS = frozenset({"workspace_read", "workspace_write"})
# A confirmation dialog is read, not scrolled forever. Past this the diff is cut
# and says so, so the Operator knows they are deciding on a summary.
_PREVIEW_DIFF_MAX_LINES = 400


@dataclass(frozen=True, slots=True)
class _MutationLedgerEntry:
    arguments_fingerprint: str
    after_sha256: str
    result: ToolResult


class RegistryToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistryConfig,
        workspace_root: str | Path,
        session_policy: SessionPolicy,
        host_denied_paths: Sequence[str] = (),
        max_read_bytes: int,
        max_search_bytes: int,
    ) -> None:
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace_root must be a directory")
        self._registry = {tool.name: tool for tool in registry.model_tools}
        # Which executor owns a tool follows from its declared effect, never from its
        # name: anything that leaves the host is the web executor's, the rest is ours.
        self._local_tool_names = frozenset(
            tool.name for tool in registry.model_tools if "data_egress" not in tool.effects
        )
        self._workspace_root = root
        self._effective_grants = session_policy.effective_grants
        if max_read_bytes < 1 or max_search_bytes < 1:
            raise ValueError("byte limits must be positive")
        self._max_read_bytes = max_read_bytes
        self._max_search_bytes = max_search_bytes
        self._paths = WorkspacePathPolicy(root, host_denied_paths)
        self._mutation_ledger: dict[tuple[str, str], _MutationLedgerEntry] = {}

    async def preflight(self, calls: Sequence[ToolCall]) -> ToolBatchPreflight:
        issues: list[_PreflightIssue] = []
        seen_ids: set[str] = set()
        mutation_paths: set[tuple[str, ...]] = set()
        for raw_call in calls:
            call = self._normalized(raw_call)
            try:
                if not call.id or call.id in seen_ids:
                    raise _PreflightIssue(
                        "duplicate_tool_call_id", "Tool call IDs must be non-empty and unique."
                    )
                seen_ids.add(call.id)
                definition = self._validate_call(
                    call, allow_known_replay=self._ledger_entry(call) is not None
                )
                if call.name in {"write_file", "edit"}:
                    path = _relative_parts(_string_argument(call, "file_path"), allow_dot=False)
                    if path in mutation_paths:
                        raise _PreflightIssue(
                            "conflicting_batch_mutations",
                            "A batch cannot mutate the same path more than once.",
                        )
                    mutation_paths.add(path)
                del definition
            except _PreflightIssue as issue:
                issues.append(issue)
        if issues:
            first = issues[0]
            return ToolBatchPreflight(
                allowed=False,
                reason_code=first.code,
                detail=first.detail,
            )
        return ToolBatchPreflight(allowed=True)

    async def execute(self, call: ToolCall) -> ToolResult:
        call = self._normalized(call)
        try:
            ledger_entry = self._ledger_entry(call)
            self._validate_call(call, allow_known_replay=ledger_entry is not None)
        except _PreflightIssue as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        if ledger_entry is not None:
            return self._reconcile_replay(call, ledger_entry)
        if call.name == "read_file":
            return self._read_file(call)
        if call.name == "write_file":
            result = self._write_file(call)
            self._record_mutation(call, result)
            return result
        if call.name == "edit":
            result = self._edit(call)
            self._record_mutation(call, result)
            return result
        if call.name == "list_directory":
            return self._list_directory(call)
        if call.name == "glob":
            return self._glob(call)
        if call.name == "grep_search":
            return self._grep_search(call)
        if call.name == "calculate":
            return self._calculate(call)
        return _error_result(
            call,
            ToolResultStatus.FAILED,
            "tool_not_implemented",
            "Local tool is not implemented.",
            retryable=False,
        )

    def _ledger_entry(self, call: ToolCall) -> _MutationLedgerEntry | None:
        definition = self._registry.get(call.name)
        if (
            definition is None
            or definition.replay_policy is not ReplayPolicy.RECONCILE_POSTCONDITION_BEFORE_REPLAY
        ):
            return None
        return self._mutation_ledger.get(_ledger_key(call))

    def _record_mutation(self, call: ToolCall, result: ToolResult) -> None:
        if result.status is not ToolResultStatus.SUCCESS:
            return
        path = self._workspace_root.joinpath(
            *_relative_parts(_string_argument(call, "file_path"), allow_dot=False)
        )
        try:
            after_sha = sha256(path.read_bytes()).hexdigest()
        except OSError:
            return
        self._mutation_ledger[_ledger_key(call)] = _MutationLedgerEntry(
            arguments_fingerprint=_arguments_fingerprint(call),
            after_sha256=after_sha,
            result=result,
        )

    def _reconcile_replay(self, call: ToolCall, entry: _MutationLedgerEntry) -> ToolResult:
        if entry.arguments_fingerprint != _arguments_fingerprint(call):
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "idempotency_conflict",
                "The replay key was already used with different arguments.",
                retryable=False,
                mutation=True,
            )
        path = self._workspace_root.joinpath(
            *_relative_parts(_string_argument(call, "file_path"), allow_dot=False)
        )
        try:
            current_sha = sha256(path.read_bytes()).hexdigest()
        except OSError:
            current_sha = None
        if current_sha != entry.after_sha256:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "replay_postcondition_conflict",
                "The mutation postcondition no longer holds; replay was not applied.",
                retryable=True,
                mutation=True,
            )
        meta = dict(entry.result.meta)
        meta["replayed"] = True
        return replace(entry.result, tool_call_id=call.id, meta=meta)

    def _not_found_message(self, relative_path: str, kind: str) -> str:
        """Not found, plus the nearest real paths: the model's guess is usually close.

        Observed in real Turns: "vibe-coder/nota-fiscal" for "Vibe coder/nota.avif".
        A bare "not found" left the model guessing a third name, and the guard
        against repeated lookups then ended the Turn with nothing found.
        """
        candidates = self._workspace_paths(directories=kind == "directory")
        wanted = relative_path.strip("/").lower()
        by_lower = {candidate.lower(): candidate for candidate in candidates}
        close = difflib.get_close_matches(wanted, list(by_lower), n=3, cutoff=0.5)
        name = wanted.rsplit("/", 1)[-1]
        stem = name.split(".", 1)[0]
        named = [
            candidate
            for lowered, candidate in by_lower.items()
            if stem and stem in lowered.rsplit("/", 1)[-1]
        ]
        suggestions = list(dict.fromkeys([*(by_lower[item] for item in close), *named]))[:5]
        message = f"The requested {kind} was not found: {relative_path}."
        if len(suggestions) == 1:
            return f"{message} It exists as {suggestions[0]}: use exactly that path."
        if suggestions:
            return f"{message} Similar paths that do exist: {', '.join(suggestions)}."
        return f"{message} Use glob with a pattern such as **/*{stem or name}* to search by name."

    def _workspace_paths(self, *, directories: bool) -> list[str]:
        """Readable paths under the root, bounded, for not-found suggestions only."""
        found: list[str] = []
        for current, dirnames, filenames in os.walk(self._workspace_root, followlinks=False):
            base = Path(current).relative_to(self._workspace_root).parts
            dirnames.sort()
            for name in dirnames if directories else filenames:
                logical = (*base, name)
                try:
                    self._validate_policy_path(logical)
                    self._validate_read_path(logical)
                except _PreflightIssue:
                    continue
                found.append("/".join(logical))
                if len(found) >= _SUGGESTION_SCAN_LIMIT:
                    return found
            kept: list[str] = []
            for name in dirnames:
                try:
                    self._validate_policy_path((*base, name))
                except _PreflightIssue:
                    continue
                kept.append(name)
            dirnames[:] = kept
        return found

    def _read_file(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "file_path")
        parts = _relative_parts(relative_path, allow_dot=False)
        try:
            path = self._resolve_read_path(parts, strict=True)
            if not path.is_file():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_file",
                    "The requested path is not a regular file.",
                    retryable=False,
                )
            content_bytes = path.read_bytes()
        except FileNotFoundError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "path_not_found",
                self._not_found_message(relative_path, "file"),
                retryable=False,
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied access to the requested file.",
                retryable=False,
            )
        except _PreflightIssue as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        except OSError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The file could not be read.",
                retryable=True,
            )
        try:
            content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "invalid_utf8",
                _binary_file_message(relative_path),
                retryable=False,
            )

        offset = cast(int, call.arguments.get("offset", 0))
        limit = cast(int, call.arguments.get("max_lines", 500))
        lines = content_bytes.splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        page_parts: list[bytes] = []
        page_size = 0
        consumed_lines = 0
        partial_line = False
        for line in selected:
            if page_size + len(line) <= self._max_read_bytes:
                page_parts.append(line)
                page_size += len(line)
                consumed_lines += 1
                continue
            if not page_parts:
                rendered, partial_line = _truncate_utf8(line.decode("utf-8"), self._max_read_bytes)
                page_parts.append(rendered.encode("utf-8"))
                consumed_lines = 1
            break
        content = b"".join(page_parts).decode("utf-8")
        truncated = offset + consumed_lines < len(lines) or partial_line
        next_offset = offset + consumed_lines if truncated else None
        data = {
            "file_path": relative_path,
            "content": content,
            "total_lines": len(lines),
            "offset": offset,
            "line_count": consumed_lines,
            "next_offset": next_offset,
        }
        status = ToolResultStatus.EMPTY if not content else ToolResultStatus.SUCCESS
        return ToolResult(
            tool_call_id=call.id,
            status=status,
            retryable=False,
            data=data,
            error=None,
            meta=_meta(truncated=truncated),
        )

    def _list_directory(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "directory_path", default=".")
        parts = _relative_parts(relative_path, allow_dot=True)
        try:
            path = self._resolve_read_path(parts, strict=True)
            if not path.is_dir():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_directory",
                    "The requested path is not a directory.",
                    retryable=False,
                )
            entries: list[dict[str, str]] = []
            for entry in path.iterdir():
                entry_parts = (*parts, entry.name)
                try:
                    self._validate_policy_path(entry_parts)
                    self._validate_read_path(entry_parts)
                except _PreflightIssue:
                    continue
                if entry.is_symlink():
                    entry_type = "symlink"
                elif entry.is_dir():
                    entry_type = "directory"
                elif entry.is_file():
                    entry_type = "file"
                else:
                    entry_type = "other"
                entries.append(
                    {
                        "name": entry.name,
                        "path": "/".join(entry_parts),
                        "type": entry_type,
                    }
                )
            entries.sort(key=lambda item: item["path"])
        except FileNotFoundError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "path_not_found",
                self._not_found_message(relative_path, "directory"),
                retryable=False,
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied access to the requested directory.",
                retryable=False,
            )
        except _PreflightIssue as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        except OSError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The directory could not be listed.",
                retryable=True,
            )
        offset = cast(int, call.arguments.get("offset", 0))
        limit = cast(int, call.arguments.get("limit", 200))
        page = entries[offset : offset + limit]
        truncated = offset + len(page) < len(entries)
        data = {
            "directory_path": relative_path,
            "entries": page,
            "offset": offset,
            "next_offset": offset + len(page) if truncated else None,
        }
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS if page else ToolResultStatus.EMPTY,
            retryable=False,
            data=data,
            error=None,
            meta=_meta(truncated=truncated),
        )

    def _glob(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "directory_path", default=".")
        pattern = _string_argument(call, "pattern")
        parts = _relative_parts(relative_path, allow_dot=True)
        try:
            base = self._resolve_read_path(parts, strict=True)
            if not base.is_dir():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_directory",
                    "The requested path is not a directory.",
                    retryable=False,
                )
            matches: list[str] = []
            for match in base.glob(pattern, recurse_symlinks=False):
                child_parts = match.relative_to(base).parts
                logical_parts = (*parts, *child_parts)
                if _has_symlink_ancestor(base, child_parts):
                    continue
                try:
                    self._validate_policy_path(logical_parts)
                    self._validate_read_path(logical_parts)
                except _PreflightIssue:
                    continue
                matches.append("/".join(logical_parts))
            matches.sort()
        except FileNotFoundError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "path_not_found",
                "The requested directory was not found.",
                retryable=False,
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied the glob operation.",
                retryable=False,
            )
        except _PreflightIssue as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        except (OSError, ValueError):
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The glob operation could not be completed.",
                retryable=True,
            )
        offset = cast(int, call.arguments.get("offset", 0))
        limit = cast(int, call.arguments.get("limit", 200))
        page = matches[offset : offset + limit]
        truncated = offset + len(page) < len(matches)
        data = {
            "directory_path": relative_path,
            "pattern": pattern,
            "paths": page,
            "offset": offset,
            "next_offset": offset + len(page) if truncated else None,
        }
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS if page else ToolResultStatus.EMPTY,
            retryable=False,
            data=data,
            error=None,
            meta=_meta(truncated=truncated),
        )

    def _calculate(self, call: ToolCall) -> ToolResult:
        expression = _string_argument(call, "expression")
        try:
            value = _evaluate_expression(expression)
        except _ExpressionError as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        except ZeroDivisionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "division_by_zero",
                "The expression divides by zero.",
                retryable=False,
            )
        except (OverflowError, ValueError):
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "expression_not_computable",
                "The expression could not be computed.",
                retryable=False,
            )
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={"expression": expression, "result": value},
            error=None,
            meta=_meta(truncated=False, producer="local_compute"),
        )

    def _grep_search(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "directory_path", default=".")
        pattern = _string_argument(call, "pattern")
        is_regex = cast(bool, call.arguments.get("is_regex", False))
        ignore_case = cast(bool, call.arguments.get("ignore_case", False))
        include = cast(str | None, call.arguments.get("include"))
        expression = (
            re.compile(
                pattern if is_regex else re.escape(pattern), re.IGNORECASE if ignore_case else 0
            )
            if is_regex or ignore_case
            else None
        )
        parts = _relative_parts(relative_path, allow_dot=True)
        offset = cast(int, call.arguments.get("offset", 0))
        limit = cast(int, call.arguments.get("limit", 200))
        try:
            base = self._resolve_read_path(parts, strict=True)
            if not base.is_dir():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_directory",
                    "The requested path is not a directory.",
                    retryable=False,
                )
            matches: list[dict[str, Any]] = []
            matched_count = 0
            has_more = False
            byte_truncated = False
            result_bytes = 0
            stop = False
            for current_root, directory_names, file_names in os.walk(base, followlinks=False):
                current = Path(current_root)
                current_parts = current.relative_to(base).parts
                allowed_directories: list[str] = []
                for name in sorted(directory_names):
                    candidate = current / name
                    logical_parts = (*parts, *current_parts, name)
                    if candidate.is_symlink():
                        continue
                    try:
                        self._validate_policy_path(logical_parts)
                        self._validate_read_path(logical_parts)
                    except _PreflightIssue:
                        continue
                    allowed_directories.append(name)
                directory_names[:] = allowed_directories

                for name in sorted(file_names):
                    logical_parts = (*parts, *current_parts, name)
                    if include is not None and not _included(include, (*current_parts, name)):
                        continue
                    try:
                        self._validate_policy_path(logical_parts)
                        self._validate_read_path(logical_parts)
                    except _PreflightIssue:
                        continue
                    try:
                        file_path = self._resolve_read_path(logical_parts, strict=True)
                        text = file_path.read_text(encoding="utf-8")
                    except (_PreflightIssue, OSError, UnicodeDecodeError):
                        continue
                    logical_path = "/".join(logical_parts)
                    for line_number, line in enumerate(text.splitlines(), start=1):
                        found = (
                            expression.search(line) is not None if expression else pattern in line
                        )
                        if not found:
                            continue
                        if matched_count < offset:
                            matched_count += 1
                            continue
                        if len(matches) >= limit:
                            has_more = True
                            stop = True
                            break
                        remaining = self._max_search_bytes - result_bytes
                        if remaining <= 0:
                            has_more = True
                            byte_truncated = True
                            stop = True
                            break
                        rendered_line, line_truncated = _truncate_utf8(line, remaining)
                        match: dict[str, Any] = {
                            "path": logical_path,
                            "line_number": line_number,
                            "line": rendered_line,
                        }
                        if line_truncated:
                            match["line_truncated"] = True
                            byte_truncated = True
                        matches.append(match)
                        matched_count += 1
                        result_bytes += len(rendered_line.encode("utf-8"))
                        if line_truncated:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
        except FileNotFoundError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "path_not_found",
                "The requested directory was not found.",
                retryable=False,
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied the search operation.",
                retryable=False,
            )
        except _PreflightIssue as issue:
            return _error_result(
                call,
                ToolResultStatus.BLOCKED,
                issue.code,
                issue.detail,
                retryable=False,
            )
        except OSError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The search operation could not be completed.",
                retryable=True,
            )
        truncated = has_more or byte_truncated
        data = {
            "directory_path": relative_path,
            "pattern": pattern,
            "is_regex": is_regex,
            "matches": matches,
            "offset": offset,
            "next_offset": offset + len(matches) if truncated else None,
        }
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS if matches else ToolResultStatus.EMPTY,
            retryable=False,
            data=data,
            error=None,
            meta=_meta(truncated=truncated),
        )

    def _write_file(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "file_path")
        content = _string_argument(call, "content").encode("utf-8")
        parts = _relative_parts(relative_path, allow_dot=False)
        path = self._workspace_root.joinpath(*parts)
        try:
            exists = path.exists()
            if exists and not path.is_file():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_file",
                    "The mutation target is not a regular file.",
                    retryable=False,
                    mutation=True,
                )
            before = path.read_bytes() if exists else None
            before_sha = sha256(before).hexdigest() if before is not None else None
            after_sha = sha256(content).hexdigest()
            if before == content:
                return self._mutation_result(
                    call,
                    relative_path=relative_path,
                    before_sha=before_sha,
                    after_sha=after_sha,
                    changed=False,
                    created_directories=[],
                )
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) if exists else None
            created_directories = self._create_parent_directories(parts[:-1])
            _atomic_replace(path, content, mode=mode)
            return self._mutation_result(
                call,
                relative_path=relative_path,
                before_sha=before_sha,
                after_sha=after_sha,
                changed=True,
                created_directories=created_directories,
            )
        except FileNotFoundError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "path_not_found",
                "A mutation path component was not found.",
                retryable=True,
                mutation=True,
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied the mutation.",
                retryable=False,
                mutation=True,
            )
        except OSError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The atomic mutation could not be committed.",
                retryable=True,
                mutation=True,
            )

    def _edit(self, call: ToolCall) -> ToolResult:
        relative_path = _string_argument(call, "file_path")
        parts = _relative_parts(relative_path, allow_dot=False)
        path = self._workspace_root.joinpath(*parts)
        try:
            if not path.exists():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "path_not_found",
                    self._not_found_message(relative_path, "file"),
                    retryable=False,
                    mutation=True,
                )
            if not path.is_file():
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "not_a_file",
                    "The mutation target is not a regular file.",
                    retryable=False,
                    mutation=True,
                )
            before = path.read_bytes()
            try:
                edited = _replaced_text(
                    before.decode("utf-8"),
                    _string_argument(call, "old_string"),
                    _string_argument(call, "new_string"),
                    replace_all=bool(call.arguments.get("replace_all", False)),
                )
            except UnicodeDecodeError:
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    "invalid_utf8",
                    "The file is not UTF-8 text, so it cannot be edited as text.",
                    retryable=False,
                    mutation=True,
                )
            except _EditIssue as issue:
                return _error_result(
                    call,
                    ToolResultStatus.FAILED,
                    issue.code,
                    issue.detail,
                    retryable=True,
                    mutation=True,
                )
            after = edited.text.encode("utf-8")
            if after != before:
                mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
                _atomic_replace(path, after, mode=mode)
            return ToolResult(
                tool_call_id=call.id,
                status=ToolResultStatus.SUCCESS,
                retryable=False,
                data={
                    "file_path": relative_path,
                    "changed": after != before,
                    "replacements": edited.count,
                    "excerpt": _excerpt_around(edited.text, edited.first_offset, edited.length),
                },
                error=None,
                meta=_meta(truncated=False, mutation=True),
            )
        except PermissionError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_permission_denied",
                "The filesystem denied the mutation.",
                retryable=False,
                mutation=True,
            )
        except OSError:
            return _error_result(
                call,
                ToolResultStatus.FAILED,
                "filesystem_error",
                "The atomic mutation could not be committed.",
                retryable=True,
                mutation=True,
            )

    def _create_parent_directories(self, parts: tuple[str, ...]) -> list[str]:
        current = self._workspace_root
        created: list[str] = []
        for index, part in enumerate(parts):
            current = current / part
            if current.exists():
                if current.is_symlink():
                    raise OSError("symlink appeared in mutation path")
                if not current.is_dir():
                    raise NotADirectoryError
                continue
            current.mkdir()
            created.append("/".join(parts[: index + 1]))
        return created

    def _mutation_result(
        self,
        call: ToolCall,
        *,
        relative_path: str,
        before_sha: str | None,
        after_sha: str,
        changed: bool,
        created_directories: list[str],
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            status=ToolResultStatus.SUCCESS,
            retryable=False,
            data={
                "file_path": relative_path,
                "created": before_sha is None,
                "changed": changed,
                "created_directories": created_directories,
            },
            error=None,
            meta=_meta(truncated=False, mutation=True),
        )

    async def preview(self, call: ToolCall) -> ConfirmationPreview | None:
        """The diff an Operator is being asked to approve, or None if there is none.

        Runs the same path guards as a mutation, so a preview cannot read what a
        write could not touch. Anything that stops it from producing a readable
        diff — a binary file, a path the policy denies, a tool that changes
        nothing — returns None rather than a half-answer: the dialog then says
        it could not preview, instead of showing an empty diff as if nothing
        would change.
        """
        call = self._normalized(call)
        if call.name not in {"write_file", "edit"}:
            return None
        try:
            parts = _relative_parts(_string_argument(call, "file_path"), allow_dot=False)
            self._validate_policy_path(parts)
            self._validate_mutation_path(parts)
        except _PreflightIssue:
            return None
        path = self._workspace_root.joinpath(*parts)
        relative_path = "/".join(parts)
        try:
            before = path.read_bytes() if path.is_file() else b""
            exists = path.is_file()
            after = self._previewed_bytes(call, before)
        except (OSError, ValueError):
            return None
        if after is None:
            return None
        try:
            before_text = before.decode("utf-8")
            after_text = after.decode("utf-8")
        except UnicodeDecodeError:
            return None
        diff, truncated = _unified_diff(before_text, after_text, relative_path)
        return ConfirmationPreview(
            tool_call_id=call.id,
            path=relative_path,
            kind="edit" if call.name == "edit" else ("replace" if exists else "create"),
            diff=diff,
            truncated=truncated,
        )

    def _previewed_bytes(self, call: ToolCall, before: bytes) -> bytes | None:
        if call.name == "write_file":
            return _string_argument(call, "content").encode("utf-8")
        try:
            edited = _replaced_text(
                before.decode("utf-8"),
                _string_argument(call, "old_string"),
                _string_argument(call, "new_string"),
                replace_all=bool(call.arguments.get("replace_all", False)),
            )
        except _EditIssue:
            return None
        return edited.text.encode("utf-8")

    def _normalized(self, call: ToolCall) -> ToolCall:
        definition = self._registry.get(call.name)
        if definition is None:
            return call
        arguments = definition.normalized_arguments(call.arguments)
        return call if arguments == call.arguments else replace(call, arguments=arguments)

    def _validate_call(
        self, call: ToolCall, *, allow_known_replay: bool = False
    ) -> ToolDefinitionConfig:
        definition = self._registry.get(call.name)
        if definition is None:
            available = ", ".join(sorted(self._effective_catalog()))
            raise _PreflightIssue(
                "unknown_tool", f"Unknown tool. Available local tools: {available}."
            )
        if definition.status != "enabled":
            raise _PreflightIssue("tool_not_enabled", "The requested tool is not enabled.")
        if call.name not in self._local_tool_names:
            raise _PreflightIssue(
                "tool_not_available", "The requested tool is not available in this executor."
            )
        validator = Draft202012Validator(cast(Mapping[str, Any], definition.parameters))
        errors = sorted(
            validator.iter_errors(call.arguments),  # pyright: ignore[reportUnknownMemberType]
            key=lambda error: list(error.path),
        )
        if errors:
            raise _PreflightIssue("invalid_tool_arguments", errors[0].message)
        missing_grants = [
            grant for grant in definition.required_grants if grant not in self._effective_grants
        ]
        if missing_grants:
            raise _PreflightIssue(
                grant_reason_code(missing_grants[0]),
                _grant_detail(missing_grants[0]),
            )
        self._validate_call_paths(call)
        if call.name == "edit" and call.arguments.get("old_string") == call.arguments.get(
            "new_string"
        ):
            raise _PreflightIssue(
                "invalid_tool_arguments",
                "new_string is identical to old_string: nothing to change.",
            )
        if call.name == "grep_search" and call.arguments.get("is_regex", False):
            try:
                re.compile(_string_argument(call, "pattern"))
            except re.error as error:
                raise _PreflightIssue(
                    "invalid_tool_arguments", f"Invalid regular expression: {error.msg}."
                ) from None
        return definition

    def _effective_catalog(self) -> tuple[str, ...]:
        # The same catalogue the model was offered, grants aside: this list answers
        # "which names exist", and a name it cannot spend yet is still a real name.
        # Whether the effect is authorized is preflight's answer, with its own code.
        return tuple(
            name
            for name, definition in self._registry.items()
            if name in self._local_tool_names and definition.status == "enabled"
        )

    def _validate_call_paths(self, call: ToolCall) -> None:
        definition = self._registry.get(call.name)
        if definition is not None and not _WORKSPACE_EFFECTS.intersection(definition.effects):
            # A tool that touches no Workspace has no path to validate.
            return
        if call.name in {"read_file", "write_file", "edit"}:
            parts = _relative_parts(_string_argument(call, "file_path"), allow_dot=False)
            self._validate_policy_path(parts)
            if call.name in {"write_file", "edit"}:
                self._validate_mutation_path(parts)
            else:
                self._validate_read_path(parts)
            return

        directory = _string_argument(call, "directory_path", default=".")
        parts = _relative_parts(directory, allow_dot=True)
        self._validate_policy_path(parts)
        self._validate_read_path(parts)
        if call.name == "glob":
            pattern_parts = _relative_parts(
                _string_argument(call, "pattern"), allow_dot=False, allow_glob=True
            )
            self._validate_policy_path(pattern_parts, literal_only=True)

    def _validate_policy_path(self, parts: tuple[str, ...], *, literal_only: bool = False) -> None:
        self._paths.check(parts, literal_only=literal_only)

    def _validate_read_path(self, parts: tuple[str, ...]) -> None:
        self._resolve_read_path(parts, strict=False)

    def _resolve_read_path(self, parts: tuple[str, ...], *, strict: bool) -> Path:
        return self._paths.resolve_read(parts, strict=strict)

    def _validate_mutation_path(self, parts: tuple[str, ...]) -> None:
        current = self._workspace_root
        for part in parts:
            current = current / part
            try:
                if current.is_symlink():
                    raise _PreflightIssue(
                        "symlink_mutation_denied",
                        "Mutations cannot traverse or replace symbolic links.",
                    )
            except OSError:
                raise _PreflightIssue(
                    "path_validation_failed", "The mutation path could not be validated."
                ) from None


def _string_argument(call: ToolCall, key: str, *, default: str | None = None) -> str:
    value = call.arguments.get(key, default)
    if not isinstance(value, str):
        raise _PreflightIssue("invalid_tool_arguments", f"{key} must be a string.")
    return value


def _has_symlink_ancestor(base: Path, child_parts: tuple[str, ...]) -> bool:
    current = base
    for part in child_parts[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _truncate_utf8(value: str, byte_limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value, False
    candidate = encoded[: max(byte_limit, 0)]
    while candidate:
        try:
            return candidate.decode("utf-8"), True
        except UnicodeDecodeError as error:
            candidate = candidate[: error.start]
    return "", True


def _grant_detail(grant: str) -> str:
    if grant == "WorkspaceRootGrant":
        # The only missing grant no dialog can give: it is derived from the
        # server's root allowlist, so the fix lives in the Settings tab.
        return (
            "The WorkspaceRootGrant is required for this effect. "
            "Add the root in the Settings tab and open a new Conversation."
        )
    return f"The {grant} is required for this effect."


def _ledger_key(call: ToolCall) -> tuple[str, str]:
    return call.name, call.idempotency_key or call.id


def _arguments_fingerprint(call: ToolCall) -> str:
    serialized = json.dumps(
        call.arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(call.name.encode("utf-8") + b"\x00" + serialized).hexdigest()


def _unified_diff(before: str, after: str, path: str) -> tuple[str, bool]:
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    truncated = len(lines) > _PREVIEW_DIFF_MAX_LINES
    if truncated:
        lines = lines[:_PREVIEW_DIFF_MAX_LINES]
    return "".join(lines).rstrip("\n"), truncated


_ESCAPED = re.compile(r'\\([ntr"\\])')
_UNESCAPED = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


class _EditIssue(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class _EditedText:
    text: str
    count: int
    first_offset: int
    length: int


def _unescaped(value: str) -> str:
    return _ESCAPED.sub(lambda match: _UNESCAPED[match.group(1)], value)


def _escaped_block(value: str) -> bool:
    """A block that arrived as escaped JSON: backslash-n text and no real break."""
    return "\\n" in value and "\n" not in value and "\r" not in value


def _replaced_text(
    text: str, old_string: str, new_string: str, *, replace_all: bool
) -> _EditedText:
    """The file after an edit, computed once for the preview and for the write.

    The Operator approves the preview, so it has to show the repaired block the
    executor will write, not the raw one the model sent. Two repairs, both seen
    in Gemma 4 seeds, and both unable to undo a real edit:

    - The file reaches the model as escaped JSON and it writes the escapes back,
      so a multi-line block lands as one line of backslash-n text. When old_string
      only matches after decoding, both strings are decoded; new_string alone is
      decoded when it replaces several lines with none. A single line holding a
      string literal such as "a\\n" never matches either shape.
    - A block that replaced whole lines arrives without its final break, and the
      next line would be welded onto it. Not at the end of the file, where
      dropping the break is a real edit, and not when new_string is empty,
      which deletes the lines.
    """
    old, new = old_string, new_string
    count = text.count(old)
    if count == 0 and _escaped_block(old) and text.count(_unescaped(old)):
        old = _unescaped(old)
        count = text.count(old)
        if _escaped_block(new):
            new = _unescaped(new)
    elif old.count("\n") >= 2 and _escaped_block(new):
        new = _unescaped(new)
    if count == 0:
        raise _EditIssue("old_string_not_found", _not_found_detail(text, old))
    if count > 1 and not replace_all:
        lines = [
            text.count("\n", 0, match.start()) + 1 for match in re.finditer(re.escape(old), text)
        ]
        listed = ", ".join(str(line) for line in lines[:10])
        raise _EditIssue(
            "old_string_not_unique",
            f"old_string appears {count} times (lines {listed}). Include more of the "
            "surrounding lines so it matches exactly once, or set replace_all to true.",
        )
    first = text.index(old)
    if (
        new
        and old.endswith(("\n", "\r"))
        and not new.endswith(("\n", "\r"))
        and first + len(old) < len(text)
    ):
        new += "\r\n" if old.endswith("\r\n") else old[-1]
    after = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    return _EditedText(
        text=after, count=count if replace_all else 1, first_offset=first, length=len(new)
    )


def _not_found_detail(text: str, old: str) -> str:
    """Where the model's memory of the file parted from the file, when it can tell."""
    first_line = next((line.strip() for line in old.splitlines() if line.strip()), "")
    if first_line:
        for number, line in enumerate(text.splitlines(), start=1):
            if first_line in line:
                return (
                    f"old_string does not appear in the file as written. Its first line matches "
                    f"line {number}, so the lines after it, the indentation or the line breaks "
                    "differ. Copy the text exactly as read_file returned it."
                )
    return (
        "old_string does not appear in the file. It must match the current text exactly, "
        "including indentation and line breaks: read_file the file and copy the text from it."
    )


_EXCERPT_CONTEXT_LINES = 2
_SUGGESTION_SCAN_LIMIT = 5000
_EXCERPT_MAX_LINES = 20


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".bmp", ".heic"})


def _included(include: str, parts: tuple[str, ...]) -> bool:
    """A pattern without a slash names files at any depth, as *.py does in grep tools."""
    if "/" not in include:
        return PurePosixPath(parts[-1]).full_match(include)
    return PurePosixPath(*parts).full_match(include)


def _binary_file_message(relative_path: str) -> str:
    if Path(relative_path).suffix.lower() in _IMAGE_SUFFIXES:
        return (
            "This is an image, not text, so read_file cannot read it. describe_image "
            "answers a question about a PNG, JPEG or WebP image."
        )
    return "The file is not UTF-8 text, so read_file cannot read it."


def _excerpt_around(text: str, offset: int, length: int) -> Mapping[str, Any]:
    """The edited region with a little context, so the model sees what it wrote."""
    lines = text.splitlines()
    first = text.count("\n", 0, offset)
    last = text.count("\n", 0, offset + max(length - 1, 0))
    start = max(first - _EXCERPT_CONTEXT_LINES, 0)
    end = min(last + _EXCERPT_CONTEXT_LINES + 1, len(lines), start + _EXCERPT_MAX_LINES)
    return {"start_line": start + 1, "text": "\n".join(lines[start:end])}


class _ExpressionError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


# 9**9**9 finishes no faster than the heat death of the host, and it would hang
# the event loop long before any Turn limit could notice.
_MAX_EXPONENT = 1000


def _evaluate_expression(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError) as error:
        raise _ExpressionError(
            "invalid_tool_arguments", "The expression could not be parsed."
        ) from error
    return _evaluate_node(tree.body)


def _evaluate_node(node: ast.expr) -> float | int:
    """Arithmetic only: no name, call, attribute or subscript is ever resolved."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _ExpressionError(
                "invalid_tool_arguments", "Only numbers are allowed in an expression."
            )
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = _evaluate_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        match node.op:
            case ast.Add():
                return left + right
            case ast.Sub():
                return left - right
            case ast.Mult():
                return left * right
            case ast.Div():
                return left / right
            case ast.FloorDiv():
                return left // right
            case ast.Mod():
                return left % right
            case ast.Pow():
                if abs(right) > _MAX_EXPONENT:
                    raise _ExpressionError(
                        "expression_too_large", "The exponent exceeds what this tool computes."
                    )
                return left**right
            case _:
                raise _ExpressionError(
                    "invalid_tool_arguments", "The expression uses an unsupported operator."
                )
    raise _ExpressionError("invalid_tool_arguments", "The expression is not plain arithmetic.")


def _meta(
    *,
    truncated: bool,
    mutation: bool = False,
    producer: str = "local_filesystem",
) -> Mapping[str, Any]:
    meta: dict[str, Any] = {
        "producer": producer,
        "truncated": truncated,
        "taints": [],
    }
    if mutation:
        meta["limitations"] = ["external_filesystem_toctou"]
    return meta


def _error_result(
    call: ToolCall,
    status: ToolResultStatus,
    code: str,
    message: str,
    *,
    retryable: bool,
    mutation: bool = False,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        status=status,
        retryable=retryable,
        data=None,
        error={"code": code, "message": message},
        meta=_meta(truncated=False, mutation=mutation),
    )


def _atomic_replace(path: Path, content: bytes, *, mode: int | None) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            if mode is not None:
                os.fchmod(temp_file.fileno(), mode)
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()
