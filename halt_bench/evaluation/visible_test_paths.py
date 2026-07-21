"""Shared visible-test path resolution for validation and test execution.

The resolver intentionally preserves the linter's established lookup order:
exact repository-relative path, first suffix match, then first basename match.
Both input validation and check-2 execution use this module so they select the
same file for a given visible-test identifier.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

_JS_TS_LANGS = frozenset({"js", "ts", "javascript", "typescript", "jsx", "tsx", "mjs", "cjs"})
_GO_LANGS = frozenset({"go", "golang"})
_FILE_LEVEL_JS_TS_SELECTORS = frozenset({"test suite"})
_RESOLVED_MARKER = "HALT_BENCH_VISIBLE_PATH_RESOLVED:"
_MISSING_MARKER = "HALT_BENCH_VISIBLE_PATH_MISSING:"


@dataclass(frozen=True)
class VisibleTestPathResolution:
    """Result of resolving file-backed visible-test identifiers."""

    resolved_tests: list[str]
    missing_tests: list[str]


def extract_test_file_path(test_id: str, language: str) -> str | None:
    """Extract the file portion of a visible-test identifier."""
    lang = language.lower()
    if lang in _GO_LANGS:
        if "_test.go" in test_id:
            return test_id.split("::")[0] if "::" in test_id else test_id
        return None
    if lang in _JS_TS_LANGS:
        path_part = test_id.split("|")[0].strip() if "|" in test_id else test_id
        js_ts_exts = (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
        if "/" in path_part or any(path_part.endswith(ext) for ext in js_ts_exts):
            return path_part
        return None
    return test_id.split("::")[0] if "::" in test_id else test_id


def is_file_level_visible_test(test_id: str, language: str) -> bool:
    """Return True when a visible-test ID intentionally names a whole file.

    Some runners/parsers cannot address individual cases and represent a whole
    JS/TS spec file as ``<path> | test suite``.  Treat that as a framework-level
    convention, not as a literal test selector to find in source.
    """
    lang = language.lower()
    if lang in _JS_TS_LANGS:
        if "|" not in test_id:
            return extract_test_file_path(test_id, language) is not None
        selector = test_id.split("|", 1)[1].strip().lower()
        return selector in _FILE_LEVEL_JS_TS_SELECTORS
    if lang in _GO_LANGS:
        return "_test.go" in test_id and "::" not in test_id
    return "::" not in test_id and extract_test_file_path(test_id, language) is not None


def _replace_test_file_path(test_id: str, original_path: str, resolved_path: str) -> str:
    """Replace only the file portion, preserving the test selector verbatim."""
    if test_id.startswith(original_path):
        return resolved_path + test_id[len(original_path) :]
    return resolved_path


def _path_entries(
    visible_tests: list[str],
    language: str,
    repo_path: str,
) -> list[tuple[int, str]]:
    repo_prefix = repo_path.rstrip("/") + "/"
    entries: list[tuple[int, str]] = []
    for index, test_id in enumerate(visible_tests):
        path = extract_test_file_path(test_id, language)
        if path is None:
            continue
        normalized = path[len(repo_prefix) :] if path.startswith(repo_prefix) else path.lstrip("/")
        entries.append((index, normalized))
    return entries


def build_visible_test_path_resolution_script(
    visible_tests: list[str],
    language: str,
    repo_path: str,
) -> str:
    """Build the shared in-container path lookup script.

    The lookup order is deliberately identical to the original ``-iv``
    existence check. ``find | head -1`` behavior is retained unchanged.
    """
    entries = _path_entries(visible_tests, language, repo_path)
    if not entries:
        return ""

    quoted_repo = shlex.quote(repo_path.rstrip("/"))
    lines = [
        "_hb_resolve_visible_path() {",
        "  local idx=$1 rel=$2 hit resolved",
        f"  local repo={quoted_repo}",
        '  if [ -e "$repo/$rel" ]; then',
        '    hit="$repo/$rel"',
        "  else",
        '    hit=$(find "$repo" -type f -path "*/$rel" 2>/dev/null | head -1)',
        "  fi",
        '  if [ -z "$hit" ]; then',
        '    hit=$(find "$repo" -type f -name "$(basename "$rel")" 2>/dev/null | head -1)',
        "  fi",
        '  if [ -n "$hit" ]; then',
        '    resolved=${hit#"$repo"/}',
        f'    printf "{_RESOLVED_MARKER}%s:%s\\n" "$idx" "$resolved"',
        "  else",
        f'    printf "{_MISSING_MARKER}%s\\n" "$idx"',
        "  fi",
        "}",
    ]
    for index, relative_path in entries:
        lines.append(f"_hb_resolve_visible_path {index} {shlex.quote(relative_path)}")
    return "\n".join(lines)


def parse_visible_test_path_resolution(
    output: str,
    visible_tests: list[str],
    language: str,
    repo_path: str,
) -> VisibleTestPathResolution:
    """Apply resolver-script output to visible-test identifiers."""
    resolved_paths: dict[int, str] = {}
    missing_indexes: set[int] = set()

    for line in output.splitlines():
        if line.startswith(_RESOLVED_MARKER):
            payload = line[len(_RESOLVED_MARKER) :]
            index_text, separator, resolved_path = payload.partition(":")
            if separator and index_text.isdigit():
                resolved_paths[int(index_text)] = resolved_path
        elif line.startswith(_MISSING_MARKER):
            index_text = line[len(_MISSING_MARKER) :]
            if index_text.isdigit():
                missing_indexes.add(int(index_text))

    entries = dict(_path_entries(visible_tests, language, repo_path))
    resolved_tests = list(visible_tests)
    for index, original_path in entries.items():
        resolved_path = resolved_paths.get(index)
        if resolved_path is not None:
            resolved_tests[index] = _replace_test_file_path(
                visible_tests[index], original_path, resolved_path
            )
        elif index not in missing_indexes:
            # Missing/unparseable resolver output must not be treated as found.
            missing_indexes.add(index)

    return VisibleTestPathResolution(
        resolved_tests=resolved_tests,
        missing_tests=[
            test_id for index, test_id in enumerate(visible_tests) if index in missing_indexes
        ],
    )
