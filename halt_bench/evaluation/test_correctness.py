"""Shared test-correctness check: apply a patch and run the visible test suite.

Used by:
  - HaltBenchLinter (check 4): verifies golden_patch passes all visible tests
  - evaluate_task_run (runner.py): verifies the agent's patch passes all visible tests

Mirrors hil_bench_agent.py _validate_swe_fast exactly for the test execution
path: scripts at /root/, python (not python3), /tmp/output.json, bash -c for
git apply, SWEAP JSON parsed with three strategies, git safe.directory config.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path

from halt_bench.evaluation.schema import TestCorrectnessResult
from halt_bench.runtime.command import run_command

logger = logging.getLogger(__name__)

_REPO_PATH = "/app"

# Returned by _run_tests_in_container when tests were requested but none were
# collected AND we cannot make a definitive pass/fail call.  This covers:
#   - ImportError, SyntaxError, or missing module at import time (Python)
#   - run_script.sh -run pattern never matched any test function (e.g. Ginkgo
#     spec names used as -run pattern) AND the full-suite fallback also passed
#     (so the suite is not broken, but we can't verify specific test cases).
# The caller converts this into tc_inconclusive=True on the result object.
_COLLECTION_ERROR_SENTINEL = (
    "0 test(s) collected — likely a collection error "
    "(ImportError, missing module, or SyntaxError)"
)

_PATCH_FILTER_PATTERNS = [
    r"__pycache__/",
    r"node_modules/",
    r"\.egg-info/",
    r"diff --git a/\S+\.pyc ",
    r"diff --git a/\S+\.pyo ",
    r"diff --git a/\S+\.so ",
    r"diff --git a/\S+\.dll ",
    r"diff --git a/\S+\.dylib ",
    r"diff --git a/parser\.py b/parser\.py",
    r"diff --git a/run_script\.sh b/run_script\.sh",
    r"appendonlydir/",
    r"diff --git a/\S*dump\.rdb ",
    r"diff --git a/\S*appendonly\.aof ",
]

_JS_TS_LANGS = frozenset({"js", "ts", "javascript", "typescript", "jsx", "tsx", "mjs", "cjs"})
_GO_LANGS = frozenset({"go", "golang"})


def has_test_failures(output: str) -> bool:
    """Return True if raw test-runner output contains an explicit failure signal.

    Supports Go (go test + Ginkgo v2), Python (pytest), JS/TS (Jest/Vitest),
    and ospec (tutanota).
    Strips ANSI escape codes before checking so callers don't need to pre-clean.

    Go / Jest:   "FAIL\t<pkg>" or "FAIL <file>" at the start of a line.
    Go per-test: "--- FAIL: <name>".
    Ginkgo v2:   "<N> Failed" summary line (N > 0).
    pytest:      "FAILED <path>::<test>" at the start of a line.
    ospec:       "N failure(s)" summary line (N > 0).
    """
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    # Go / Jest file-level failure: "FAIL" as the first word on a line
    if re.search(r"^FAIL\b", clean, re.MULTILINE):
        return True
    # Go per-test marker
    if re.search(r"^--- FAIL:", clean, re.MULTILINE):
        return True
    # Ginkgo v2 summary: "N Failed" where N > 0
    m = re.search(r"(\d+) Failed", clean)
    if m and int(m.group(1)) > 0:
        return True
    # pytest compact-mode summary: "FAILED tests/test_foo.py::test_bar - ..."
    if re.search(r"^FAILED\b", clean, re.MULTILINE):
        return True
    # ospec (tutanota): "N failure(s)" summary where N > 0.
    # Pattern requires digit(s) starting 1-9 to exclude "0 failures" (clean pass).
    if re.search(r"\b[1-9]\d*\s+failures?\b", clean, re.IGNORECASE):
        return True
    return False


# Exit codes that mean "no tests were collected / filter matched nothing" OR
# "the test environment itself is broken" rather than "the agent's code is
# broken".  These are infra / task-config issues.
#
#   5  — pytest: "no tests were collected" (filter matched nothing)
#   126 — bash: command not executable (permission denied)
#   127 — bash: command not found (test runner binary missing from PATH)
#
# Exit 126/127 consistently affect ALL agent patches for the same task image
# which confirms they are infrastructure issues, not patch-caused failures.
# Retroactive validation against the ProtonMail TypeScript tasks confirmed this:
# every patch on that task image exits 127 regardless of patch content.
_INCONCLUSIVE_EXIT_CODES: frozenset[str] = frozenset({"5", "126", "127"})

# Patterns in test-runner output that indicate the filter matched zero tests
# rather than the tests themselves failing.  Language-agnostic.
_NO_TESTS_COLLECTED_PATTERNS: tuple[str, ...] = (
    r"no tests ran",
    r"collected 0 items",
    r"Your test suite must contain at least one test",
    r"0 examples,\s*0 failures",  # RSpec with no matching examples
    r"no tests? found",
    r"0 tests?\s+ran",
)


def _filter_patch(patch: str) -> str:
    if not patch:
        return patch
    file_diffs = re.split(r"(?=diff --git )", patch)
    filtered = []
    for diff in file_diffs:
        if not diff.strip():
            continue
        if not any(re.search(p, diff) for p in _PATCH_FILTER_PATTERNS):
            filtered.append(diff)
    return "".join(filtered)


def _is_proton_webclients_runner(run_script: str) -> bool:
    return "yarn workspace" in run_script and any(
        marker in run_script
        for marker in (
            "proton-mail",
            "@proton/components",
            "proton-drive",
            "proton-calendar",
            "proton-account",
            "proton-verify",
        )
    )


def _proton_visible_test_runner_script() -> str:
    return r"""#!/bin/bash
set -e
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}"

REPO_ROOT="$(pwd)"
JEST_BIN="$REPO_ROOT/node_modules/jest/bin/jest.js"
if [ ! -f "$JEST_BIN" ]; then
  JEST_BIN="$REPO_ROOT/node_modules/.bin/jest"
fi
if [ ! -e "$JEST_BIN" ]; then
  echo "Jest binary not found under $REPO_ROOT/node_modules" >&2
  exit 127
fi

workspace_for_file() {
  local file_path="$1"
  if [[ "$file_path" == packages/components/* ]] || [[ "$file_path" == components/* ]] || [[ "$file_path" == *components* ]]; then
    echo "packages/components"
  elif [[ "$file_path" == applications/drive/* ]] || [[ "$file_path" == src/app/store/_shares/* ]] || [[ "$file_path" == *drive* ]]; then
    echo "applications/drive"
  elif [[ "$file_path" == applications/calendar/* ]] || [[ "$file_path" == *calendar* ]]; then
    echo "applications/calendar"
  elif [[ "$file_path" == applications/account/* ]] || [[ "$file_path" == *account* ]]; then
    echo "applications/account"
  elif [[ "$file_path" == applications/verify/* ]] || [[ "$file_path" == *verify* ]]; then
    echo "applications/verify"
  else
    echo "applications/mail"
  fi
}

run_one() {
  local test_spec="$1"
  local file_path="$test_spec"
  local test_name=""

  if [[ "$test_spec" == *"|"* ]]; then
    file_path="$(echo "$test_spec" | cut -d'|' -f1 | xargs)"
    test_name="$(echo "$test_spec" | cut -d'|' -f2- | xargs)"
  fi

  local workspace_dir
  workspace_dir="$(workspace_for_file "$file_path")"
  local pattern="$file_path"
  if [[ "$pattern" == "$workspace_dir/"* ]]; then
    pattern="${pattern#"$workspace_dir/"}"
  fi

  echo "Running Jest in $workspace_dir: $pattern${test_name:+ | $test_name}"
  cd "$REPO_ROOT/$workspace_dir"
  if [ -n "$test_name" ]; then
    node "$JEST_BIN" --runInBand --ci --testPathPattern="$pattern" --testNamePattern="$test_name" --verbose
  else
    node "$JEST_BIN" --runInBand --ci --testPathPattern="$pattern" --verbose
  fi
  cd "$REPO_ROOT"
}

if [ $# -eq 0 ]; then
  echo "Proton visible-test wrapper requires explicit test paths." >&2
  exit 5
fi

if [[ "$1" == *","* ]]; then
  IFS=',' read -r -a TEST_FILES <<< "$1"
else
  TEST_FILES=("$@")
fi

for test_spec in "${TEST_FILES[@]}"; do
  run_one "$test_spec"
done
"""


def _extract_test_file_path(test_id: str, language: str) -> str | None:
    lang = language.lower()
    if lang in _GO_LANGS:
        if "_test.go" in test_id:
            return test_id.split("::")[0] if "::" in test_id else test_id
        return None
    elif lang in _JS_TS_LANGS:
        path_part = test_id.split("|")[0].strip() if "|" in test_id else test_id
        js_ts_exts = (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
        if "/" in path_part or any(path_part.endswith(e) for e in js_ts_exts):
            return path_part
        return None
    else:
        return test_id.split("::")[0] if "::" in test_id else test_id


def _build_test_args(
    tests: list[str],
    language: str,
    run_script: str,
) -> str:
    """Build shell-safe quoted argument string for run_script.sh.

    Mirrors custom_eval.py / hil_bench_agent.py _process_validation_test_args.
    """
    args: list[str] = list(tests)

    if not args:
        return ""

    lang = language.lower()

    if "ansible-test" in run_script and any("::" in t for t in args):
        args = list(
            {t.split("::")[0] for t in args if "::" in t} | {t for t in args if "::" not in t}
        )

    # Go test IDs are stored as "pkg/path/file_test.go::FunctionName".
    # `go test -run` only matches against the bare function name, so strip
    # everything up to and including the last "::".
    if lang in _GO_LANGS and any("::" in t for t in args):
        args = [t.split("::")[-1] if "::" in t else t for t in args]

    if lang in _JS_TS_LANGS:
        stripped: list[str] = []
        seen: set[str] = set()
        for t in args:
            p = t.split("|")[0].strip() if "|" in t else t
            if p not in seen:
                seen.add(p)
                stripped.append(p)
        args = stripped

    return " ".join(f"'{t.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for t in args)


def _parse_sweap_json(log: str) -> dict | None:
    """Extract the SWEAP JSON dict from test runner output.

    Three strategies (mirrors hil_bench_agent.py _parse_sweap_json_test_status):
      1. SWEAP_JSON_START / SWEAP_JSON_END markers
      2. JSON structure scan
      3. Entire log parse (last resort)
    """
    # Strategy 1
    start_idx = log.find("SWEAP_JSON_START")
    end_idx = log.find("SWEAP_JSON_END")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = log[start_idx + len("SWEAP_JSON_START") : end_idx].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 2
    for prefix in ('{\n  "tests"', '{"tests"'):
        jstart = log.find(prefix)
        if jstart != -1:
            section = log[jstart:]
            for end_pat in ("\n  ]\n}", "]\n}", "]}"):
                epos = section.rfind(end_pat)
                if epos != -1:
                    try:
                        return json.loads(section[: epos + len(end_pat)])
                    except json.JSONDecodeError:
                        continue

    # Strategy 3
    try:
        return json.loads(log.strip())
    except json.JSONDecodeError:
        pass

    return None


def _merge_status(previous: str | None, new: str) -> str:
    """Aggregate duplicate parser statuses without letting later rows hide failures."""
    failure_statuses = {"FAILED", "ERROR"}
    known_non_failures = {"PASSED", "SKIPPED", ""}
    previous = previous or ""
    new = new or ""

    if previous in failure_statuses or new in failure_statuses:
        return "FAILED" if "FAILED" in (previous, new) else "ERROR"

    # Unknown non-empty statuses are failure-like and should not be hidden by a
    # duplicate PASSED/SKIPPED row. Downstream logic rejects anything but PASSED.
    if previous not in known_non_failures:
        return previous
    if new not in known_non_failures:
        return new

    if previous == "PASSED" or new == "PASSED":
        return "PASSED"
    if previous == "SKIPPED" or new == "SKIPPED":
        return "SKIPPED"
    return ""


def _parse_sweap_json_with_required_tests(
    log: str,
    fail_to_pass: list[str],
) -> tuple[dict[str, str], list[dict], bool]:
    """Parse SWEAP JSON and map results to required test names.

    Ported directly from hil_bench_agent.py ``_parse_sweap_json_test_status``
    (which was itself copied from custom_eval.py ``parse_log_sweap_json``) so
    that our classification logic is identical to the reference implementation.

    Key behaviour (same as hil_bench):
    • Exact match first, then JS/TS pipe-format path+desc match, then
      pytest-style path::func match with parameter compatibility.
    • Parametrized variants: if FAIL_TO_PASS has a bare ``test_foo`` and the
      parser only reports ``test_foo[param]``, mark the bare name PASSED when
      ALL parametrized variants passed.
    • JS/TS extension pairs: parser may report ``.js`` for a ``.ts`` required
      test — accepted only for valid TS↔JS extension pairs.
    • NO_TESTS_FOUND_OR_PARSING_ERROR sentinel: not in fail_to_pass → no match
      → returns empty test_status_map → caller falls through to
      _classify_parser_no_results.

    Returns:
        (test_status_map, raw_tests, json_found)
        test_status_map : required_test_name → status string ("PASSED" …)
        raw_tests       : the raw ``tests`` list from the SWEAP JSON (for
                          human-readable formatting)
        json_found      : True if SWEAP JSON was successfully parsed
    """
    test_status_map: dict[str, str] = {}
    required_tests: set[str] = set(fail_to_pass) if fail_to_pass else set()

    data: dict | None = None

    # Strategy 1: SWEAP_JSON_START / SWEAP_JSON_END markers (most reliable)
    start_idx = log.find("SWEAP_JSON_START")
    end_idx = log.find("SWEAP_JSON_END")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_section = log[start_idx + len("SWEAP_JSON_START") : end_idx].strip()
        try:
            data = json.loads(json_section)
        except json.JSONDecodeError:
            pass

    # Strategy 2: look for JSON structure directly
    if data is None:
        for prefix in ('{\n  "tests"', '{"tests"'):
            jstart = log.find(prefix)
            if jstart != -1:
                section = log[jstart:]
                for end_pat in ("\n  ]\n}", "]\n}", "]}"):
                    epos = section.rfind(end_pat)
                    if epos != -1:
                        try:
                            data = json.loads(section[: epos + len(end_pat)])
                            break
                        except json.JSONDecodeError:
                            continue
            if data is not None:
                break

    # Strategy 3: try parsing entire log (last resort)
    if data is None:
        try:
            data = json.loads(log.strip())
        except json.JSONDecodeError:
            pass

    if data is None:
        return test_status_map, [], False

    # ------------------------------------------------------------------ helpers

    def _extract_pytest_components(test_name: str) -> tuple[str | None, str, str]:
        """Return (file_path, func_with_params, func_base) for pytest-style name."""
        file_path = None
        func_with_params = test_name
        if "::" in test_name:
            parts = test_name.split("::")
            file_path = parts[0]
            func_with_params = parts[-1]
        func_base = func_with_params.split("[")[0] if "[" in func_with_params else func_with_params
        return file_path, func_with_params, func_base

    def _paths_match(path1: str | None, path2: str | None) -> bool:
        """True if two file paths refer to the same file (different root prefixes OK)."""
        if path1 is None and path2 is None:
            return True
        if path1 is None or path2 is None:
            return False
        if path1 == path2:
            return True
        return (
            path1.endswith("/" + path2)
            or path2.endswith("/" + path1)
            or path1.endswith(path2)
            or path2.endswith(path1)
        )

    def _descriptions_match(required_desc: str, parser_desc: str) -> bool:
        """True when a parser-emitted test name adds only describe/context prefixes."""
        if required_desc == parser_desc:
            return True
        return (
            required_desc.endswith(" | " + parser_desc)
            or parser_desc.endswith(" | " + required_desc)
            or required_desc.endswith(" " + parser_desc)
            or parser_desc.endswith(" " + required_desc)
        )

    def _find_matching_required_tests(parser_test_name: str) -> list[str]:
        """Find all required tests that match this parser-emitted name."""
        matches: list[str] = []

        # 1. Exact match
        if parser_test_name in required_tests:
            matches.append(parser_test_name)

        # 2. JS/TS pipe format: "file/path | description"
        if " | " in parser_test_name:
            parser_path, parser_desc = parser_test_name.split(" | ", 1)
            for req_test in required_tests:
                if " | " in req_test:
                    req_path, req_desc = req_test.split(" | ", 1)
                    path_matches = (
                        req_path == parser_path
                        or req_path.endswith(parser_path)
                        or parser_path.endswith(req_path)
                    )
                    desc_matches = _descriptions_match(req_desc, parser_desc)
                    if path_matches and desc_matches and req_test not in matches:
                        matches.append(req_test)
                else:
                    if (
                        req_test == parser_path
                        or req_test.endswith(parser_path)
                        or parser_path.endswith(req_test)
                    ) and req_test not in matches:
                        matches.append(req_test)
            return matches

        # 3. Pytest format: optional path::func with optional [params]
        parser_path, parser_func_params, parser_func_base = _extract_pytest_components(
            parser_test_name
        )
        parser_func_base_lower = parser_func_base.lower()
        fallback_matches: list[str] = []

        for req_test in required_tests:
            if " | " in req_test:
                continue
            req_path, req_func_params, req_func_base = _extract_pytest_components(req_test)
            if parser_func_base_lower != req_func_base.lower():
                continue
            params_compatible = (
                parser_func_params == req_func_params
                or req_func_params == req_func_base
                or parser_func_params == parser_func_base
            )
            if not params_compatible:
                continue
            if parser_path is not None and req_path is not None:
                if _paths_match(parser_path, req_path):
                    if req_test not in matches:
                        matches.append(req_test)
                elif req_test not in fallback_matches:
                    fallback_matches.append(req_test)
            elif req_test not in matches:
                matches.append(req_test)

        return matches or fallback_matches

    # ------------------------------------------------------------------ main loop

    raw_parser_results: dict[str, str] = {}
    raw_tests: list[dict] = data.get("tests", [])

    for test in raw_tests:
        test_name = test.get("name", "")
        status_str = test.get("status", "").upper()
        if not test_name:
            continue
        raw_parser_results[test_name] = _merge_status(raw_parser_results.get(test_name), status_str)
        if required_tests:
            matched_tests = _find_matching_required_tests(test_name)
            if not matched_tests:
                continue
            for matched in matched_tests:
                test_status_map[matched] = _merge_status(test_status_map.get(matched), status_str)
        else:
            test_status_map[test_name] = _merge_status(test_status_map.get(test_name), status_str)

    # ---- Parametrized variants: bare test_foo + all test_foo[p] passed → PASSED
    if required_tests:
        for req_test in required_tests:
            if req_test in test_status_map:
                continue
            if " | " in req_test or "[" in req_test:
                continue
            req_path, _, req_func_base = _extract_pytest_components(req_test)
            req_func_base_lower = req_func_base.lower()
            parametrized = [
                st
                for st in test_status_map
                if "[" in st
                and _extract_pytest_components(st)[2].lower() == req_func_base_lower
                and (req_path is None or _paths_match(req_path, _extract_pytest_components(st)[0]))
            ]
            if parametrized and all(test_status_map[t] == "PASSED" for t in parametrized):
                test_status_map[req_test] = "PASSED"

    # ---- JS/TS extension and suite-level matching (handles .ts ↔ .js pairs and
    #      parsers that only emit "file | test suite" instead of individual tests)
    if required_tests:
        _VALID_TS_JS_PAIRS = {
            (".ts", ".js"),
            (".js", ".ts"),
            (".tsx", ".jsx"),
            (".jsx", ".tsx"),
            (".mts", ".mjs"),
            (".mjs", ".mts"),
            (".cts", ".cjs"),
            (".cjs", ".cts"),
        }

        def _get_ext(path: str) -> str | None:
            for ext in (".tsx", ".jsx", ".mts", ".mjs", ".cts", ".cjs", ".ts", ".js"):
                if path.endswith(ext):
                    return ext
            return None

        def _strip_ext(path: str) -> str:
            for ext in (".tsx", ".jsx", ".mts", ".mjs", ".cts", ".cjs", ".ts", ".js"):
                if path.endswith(ext):
                    return path[: -len(ext)]
            return path

        def _valid_ext_pair(e1: str | None, e2: str | None) -> bool:
            if e1 is None or e2 is None:
                return False
            return e1 == e2 or (e1, e2) in _VALID_TS_JS_PAIRS

        passing_suites: dict[str, tuple[str, str | None]] = {}
        passing_suites_by_base: dict[str, str] = {}

        for pkey, pstatus in raw_parser_results.items():
            if " | test suite" in pkey and pstatus == "PASSED":
                fp = pkey.split(" | ")[0]
                ext = _get_ext(fp)
                norm = _strip_ext(fp).lower()
                passing_suites[norm] = (pkey, ext)
                basename = fp.split("/")[-1]
                base_no_ext = _strip_ext(basename)
                for sfx in ("Test", ".test", ".spec", "_test", "_spec"):
                    if base_no_ext.endswith(sfx):
                        base_no_ext = base_no_ext[: -len(sfx)]
                        break
                passing_suites_by_base[base_no_ext.lower()] = pkey

        for req_test in required_tests:
            if req_test in test_status_map:
                continue
            req_ext = _get_ext(req_test)
            if "/" in req_test and req_ext is not None:
                norm_req = _strip_ext(req_test).lower()
                if norm_req in passing_suites:
                    pkey, pext = passing_suites[norm_req]
                    if _valid_ext_pair(req_ext, pext):
                        test_status_map[req_test] = "PASSED"
                continue
            if " | " in req_test:
                req_fp = req_test.split(" | ")[0]
                req_ext = _get_ext(req_fp)
                if req_ext is not None:
                    norm_req = _strip_ext(req_fp).lower()
                    if norm_req in passing_suites:
                        pkey, pext = passing_suites[norm_req]
                        if _valid_ext_pair(req_ext, pext):
                            test_status_map[req_test] = "PASSED"
                continue
            req_lower = req_test.lower()
            for suite_base, pkey in passing_suites_by_base.items():
                if req_lower.startswith(suite_base):
                    test_status_map[req_test] = "PASSED"
                    break

    return test_status_map, raw_tests, True


def _format_test_output_text(tests: list[dict]) -> str:
    """Format SWEAP test list into a human-readable per-test summary for a gist.

    Only PASSED and FAILED/ERROR tests are shown; SKIPPED tests are ignored
    (they are expected and were not part of the evaluation signal).
    """
    passed_names: list[str] = []
    failed_names: list[str] = []
    for t in tests:
        name = t.get("name", "?")
        status = t.get("status", "?").upper()
        if status == "PASSED":
            passed_names.append(name)
        elif status in ("FAILED", "ERROR"):
            failed_names.append(f"{name} ({status})")

    n_passed = len(passed_names)
    n_failed = len(failed_names)
    n_total = n_passed + n_failed
    lines: list[str] = [f"Test Results: {n_passed}/{n_total} passed", ""]
    if failed_names:
        lines.append("FAILED:")
        for name in failed_names:
            lines.append(f"  FAIL  {name}")
        lines.append("")
    if passed_names:
        lines.append("PASSED:")
        for name in passed_names:
            lines.append(f"  PASS  {name}")
    return "\n".join(lines).rstrip()


def _read_run_script_exit_code(container_id: str) -> str:
    """Read the saved run_script.sh exit code from /tmp/test_exit_code.

    Returns the raw string ("0", "1", etc.) or "" if the file doesn't exist.
    Written by test_cmd's `echo $? > /tmp/test_exit_code` line.
    """
    ec_tr = run_command(
        [
            "docker",
            "exec",
            container_id,
            "bash",
            "-c",
            "cat /tmp/test_exit_code 2>/dev/null || echo ''",
        ],
        check=False,
    )
    return ec_tr.stdout.strip()


def _try_inline_go_test_parse(
    content: str,
    visible_tests: list[str],
) -> tuple[bool, str, str | None] | None:
    """Fallback: extract Go ``go test -v`` results directly from raw output.

    Used when parser.py could not produce usable SWEAP JSON but the raw
    test output still contains Go per-test markers (``--- PASS:`` /
    ``--- FAIL:``).  This avoids relying solely on a masked exit code (e.g.
    ``|| true`` wrappers) by giving explicit per-test confirmation.

    Args:
        content:       Combined stdout + stderr from the container.
        visible_tests: Required test IDs (may use ``pkg::TestFunc`` format).

    Returns:
        ``(passed, detail, test_output)`` if at least one visible test was
        matched via the Go output, or ``None`` if no Go test markers were
        found or no visible tests could be matched.
    """
    if "--- PASS:" not in content and "--- FAIL:" not in content:
        return None

    # Build bare-name → status from every ``--- PASS/FAIL: TestName`` line.
    inline_map: dict[str, str] = {}
    for line in content.splitlines():
        m = re.match(r"^\s*---\s+(PASS|FAIL):\s+(\S+)", line)
        if m:
            status = "PASSED" if m.group(1) == "PASS" else "FAILED"
            test_name = m.group(2)
            # For a name seen multiple times keep the worst-case status.
            if test_name not in inline_map or status == "FAILED":
                inline_map[test_name] = status

    if not inline_map or not visible_tests:
        return None

    matched: dict[str, str] = {}
    for vt in visible_tests:
        # visible_test may be "pkg/path::TestFunc" — bare name is after last "::".
        bare = vt.split("::")[-1] if "::" in vt else vt

        if bare in inline_map:
            matched[vt] = inline_map[bare]
            continue

        # Subtest aggregation: "TestFoo" matches "TestFoo/subtest_name".
        # Use worst-case status across all sub-entries.
        agg_status: str | None = None
        for iname, istatus in inline_map.items():
            if iname == bare or iname.startswith(bare + "/") or iname.startswith(bare + "#"):
                if agg_status is None or istatus == "FAILED":
                    agg_status = istatus
        if agg_status is not None:
            matched[vt] = agg_status

    if not matched:
        return None

    raw_for_fmt = [{"name": vt, "status": st} for vt, st in matched.items()]
    test_output = _format_test_output_text(raw_for_fmt)

    failed = [vt for vt, st in matched.items() if st == "FAILED"]
    missing = [vt for vt in visible_tests if vt not in matched]

    if failed:
        parts = [
            f"{len(failed)} test(s) failed (inline Go parser): "
            + ", ".join(f"{t} (FAILED)" for t in failed[:10])
        ]
        if missing:
            parts.append(
                f"{len(missing)} test(s) not found in Go output "
                f"(treated as failed): " + ", ".join(missing[:10])
            )
        return False, "; ".join(parts), test_output

    if missing:
        return (
            False,
            f"{len(missing)} test(s) not found in Go output (treated as failed): "
            + ", ".join(missing[:10]),
            test_output,
        )

    return True, f"All {len(matched)} test(s) passed (inline Go parser)", test_output


def _classify_parser_no_results(
    container_id: str,
    context_label: str = "",
    visible_tests: list[str] | None = None,
    parser_had_unmatched: bool = False,
) -> tuple[bool, str, None]:
    """Classify a run where parser.py extracted zero test results.

    Used when:
    • parser.py emitted only NO_TESTS_FOUND_OR_PARSING_ERROR sentinel entries, OR
    • parser.py returned an empty tests list but stdout.log is non-empty
      (i.e. the test runner produced output, but the parser couldn't parse it).

    Args:
        container_id:        Running Docker container to inspect.
        context_label:       Human-readable label appended to detail strings.
        visible_tests:       Required test IDs; forwarded to the inline Go
                             parser fallback (Proposal C).
        parser_had_unmatched: True when parser.py produced ≥1 result but none
                             matched any visible test name (Proposal B).
                             In this case exit-code 0 is insufficient to
                             confirm a PASS — the IDs don't line up — so we
                             return INCONCLUSIVE rather than PASSED.

    Decision logic — in priority order:

    1. Explicit failure markers in raw output  → FAIL
       (has_test_failures: ^FAIL, --- FAIL:, ^FAILED, Ginkgo "N Failed", ospec "N failures")

    2. Exit code in _INCONCLUSIVE_EXIT_CODES   → INCONCLUSIVE
       e.g. pytest exit 5 = "no tests collected";
            bash exit 126 = "permission denied";
            bash exit 127 = "command not found" (test runner missing from PATH).

    3. "No tests" output patterns              → INCONCLUSIVE
       Checked BEFORE exit-0 so frameworks like RSpec that exit 0 on
       "0 examples, 0 failures" are correctly classified here rather than
       being returned as PASS.

    4. Exit code 0                             → PASS  (with sub-checks)
       4a. Inline Go parser (Proposal C): if stdout contains ``--- PASS/FAIL:``
           markers, try to confirm/deny individual visible tests explicitly
           rather than trusting exit-code alone.
       4b. parser_had_unmatched=True (Proposal B): exit-0 cannot be trusted
           because the parser DID produce results — just with names that don't
           match the visible tests — so we return INCONCLUSIVE.
       4c. Otherwise: trust exit-0 → PASS.

    5. Other non-zero exit                     → FAIL
       Runner exited with a non-standard failure code (crash, build error …).

    6. Exit code unknown (file missing)        → INCONCLUSIVE  (safe default)

    Returns a (passed, detail, None) triple matching _run_tests_in_container's
    return type.
    """
    # One docker exec: read combined output (stdout.log + stderr.log).
    content_tr = run_command(
        [
            "docker",
            "exec",
            container_id,
            "bash",
            "-c",
            "cat /tmp/stdout.log /tmp/stderr.log 2>/dev/null",
        ],
        check=False,
    )
    raw_content = (content_tr.stdout + content_tr.stderr).strip()

    suffix = f" ({context_label})" if context_label else ""

    # 1. Explicit failure markers in output — most reliable, language-agnostic.
    if has_test_failures(raw_content):
        return (
            False,
            f"test runner output contains failure markers but parser extracted no "
            f"individual results{suffix} — check the test output for details",
            None,
        )

    saved_ec = _read_run_script_exit_code(container_id)

    if saved_ec:
        # 2. Known "no tests collected" exit codes → filter/task issue, not agent fault.
        if saved_ec in _INCONCLUSIVE_EXIT_CODES:
            return False, _COLLECTION_ERROR_SENTINEL, None

        # 3. "No tests" output patterns — checked BEFORE exit-0 so that frameworks
        #    like RSpec which exit 0 on "0 examples, 0 failures" are caught here
        #    rather than incorrectly returning PASS.
        for pat in _NO_TESTS_COLLECTED_PATTERNS:
            if re.search(pat, raw_content, re.IGNORECASE):
                return False, _COLLECTION_ERROR_SENTINEL, None

        # 4. Exit 0 → runner succeeded; parser just couldn't read the output format.
        if saved_ec == "0":
            # 4a. Inline Go parser (Proposal C): if raw output has Go test markers,
            #     try to verify individual visible tests explicitly rather than
            #     trusting a potentially-masked exit code.
            if visible_tests:
                go_result = _try_inline_go_test_parse(raw_content, visible_tests)
                if go_result is not None:
                    passed, detail, test_out = go_result
                    return passed, detail, None

            # 4b. Parser produced results but none matched visible test names
            #     (Proposal B): exit-0 is untrustworthy when the parser clearly
            #     ran and found *something* — the mismatch means we cannot
            #     confirm which tests actually ran.
            if parser_had_unmatched:
                return (
                    False,
                    f"parser extracted result(s) but none matched any visible test "
                    f"name{suffix}; exit-code 0 alone is insufficient to confirm "
                    f"a pass when test IDs do not align",
                    None,
                )

            # 4c. Parser found nothing at all and exit was 0 — trust the exit code.
            return (
                True,
                f"All tests passed (run_script.sh exit-code 0; parser could not "
                f"extract individual results{suffix} — test runner output format "
                f"unrecognised)",
                None,
            )

        # 5. Other non-zero exit → genuine runner failure (build error, crash, etc.).
        return (
            False,
            f"run_script.sh exited {saved_ec} but parser extracted no results{suffix} "
            f"(likely a build error, test failure, or crash — check the test output)",
            None,
        )

    # 6. Exit code file missing → genuinely uncertain → INCONCLUSIVE.
    return False, _COLLECTION_ERROR_SENTINEL, None


def _run_tests_in_container(
    container_id: str,
    test_args: str,
    parser_script: str,
    repo_path: str = _REPO_PATH,
    visible_tests: list[str] | None = None,
) -> tuple[bool, str, str | None]:
    """Run run_script.sh (at /root/run_script.sh) and classify results.

    Classification mirrors hil_bench_agent.py _validate_swe_fast exactly:
      - Scripts at /root/  (not /tmp/)
      - Uses 'python' (not 'python3')
      - Output to /tmp/output.json
      - bash -c for exec
      - SWEAP JSON parsed with three strategies
      - Missing required test = FAILED (same as hil_bench / swebench grading)

    When ``visible_tests`` are supplied the function uses the hil_bench approach:
    each required test must appear in the parser output as PASSED; a test that
    is absent from the results is treated as FAILED (not as inconclusive).

    Fallback chain when parser output contains no matches for visible tests:
      1. stdout.log empty + test_args → Ginkgo full-suite fallback
      2. stdout.log non-empty (or no test_args) → _classify_parser_no_results
         which uses saved exit code + raw output pattern analysis.

    Returns (passed, detail_string, test_output_text).
    test_output_text is a human-readable per-test breakdown for gist upload,
    or None when the parser produced no structured output.
    """
    if parser_script:
        # Save run_script.sh exit code immediately after it runs, before
        # parser.py, so _classify_parser_no_results can read it later.
        if test_args:
            test_cmd = (
                f"bash /root/run_script.sh {test_args} > /tmp/stdout.log 2> /tmp/stderr.log; "
                "echo $? > /tmp/test_exit_code; "
                "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
                "echo 'SWEAP_JSON_START'; cat /tmp/output.json; echo 'SWEAP_JSON_END'"
            )
        else:
            test_cmd = (
                "bash /root/run_script.sh > /tmp/stdout.log 2> /tmp/stderr.log; "
                "echo $? > /tmp/test_exit_code; "
                "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
                "echo 'SWEAP_JSON_START'; cat /tmp/output.json; echo 'SWEAP_JSON_END'"
            )
        tr = run_command(
            ["docker", "exec", container_id, "bash", "-c", f"cd {repo_path} && {test_cmd}"],
            check=False,
        )
        raw = tr.stdout + tr.stderr

        # Parse SWEAP JSON using the same fuzzy name-matching logic as
        # hil_bench_agent.py / custom_eval.py.  Returns:
        #   test_status_map : required_test_name → status string
        #   raw_tests       : raw tests list (for human-readable formatting)
        #   json_found      : whether valid SWEAP JSON was present at all
        test_status_map, raw_tests, json_found = _parse_sweap_json_with_required_tests(
            raw, visible_tests or []
        )

        if not json_found:
            logger.warning(
                "[%s] SWEAP JSON not found or unparseable — using saved exit code "
                "and raw output signals",
                container_id,
            )
            # Fall through to the empty-map path below (no extra docker exec needed
            # because _classify_parser_no_results reads /tmp/test_exit_code).

        if test_status_map:
            # ---- hil_bench grading: every required test must be PASSED ----
            # A test that is absent from parser output is treated as FAILED,
            # matching swebench's grading philosophy (missing == not passing).
            test_output = _format_test_output_text(raw_tests)

            if visible_tests:
                explicitly_failed = [
                    t
                    for t in visible_tests
                    if t in test_status_map and test_status_map[t] not in ("PASSED",)
                ]
                missing = [t for t in visible_tests if t not in test_status_map]
                not_passed = explicitly_failed + missing
            else:
                # No required test list — fall back to checking for any failure
                not_passed = [
                    t for t, s in test_status_map.items() if s not in ("PASSED", "SKIPPED")
                ]
                missing = []

            if not_passed:
                parts: list[str] = []
                if explicitly_failed if visible_tests else not_passed:
                    shown = explicitly_failed if visible_tests else not_passed
                    parts.append(
                        f"{len(shown)} test(s) failed: "
                        + ", ".join(f"{t} ({test_status_map[t]})" for t in shown[:10])
                    )
                if visible_tests and missing:
                    parts.append(
                        f"{len(missing)} test(s) not found in parser output "
                        f"(treated as failed): " + ", ".join(missing[:10])
                    )
                return False, "; ".join(parts), test_output

            passed_count = (
                sum(1 for t in visible_tests if test_status_map.get(t) == "PASSED")
                if visible_tests
                else sum(1 for s in test_status_map.values() if s == "PASSED")
            )
            return True, f"All {passed_count} test(s) passed", test_output

        # ---- test_status_map is empty: parser found nothing for visible tests ----
        # This covers:
        #   • SWEAP JSON unparseable (json_found=False)
        #   • parser returned only NO_TESTS_FOUND_OR_PARSING_ERROR sentinel
        #     (sentinel name never matches a visible test → map stays empty)
        #   • parser returned results but none matched visible test names

        # ---- Proposal D: Safety-net — restore global failure detection ----
        # If parser.py found FAILED/ERROR tests that didn't match any visible
        # test name, those failures are still real.  The per-visible_test
        # name-matching refactor inadvertently removed this safety net; this
        # restores it.  The NO_TESTS_FOUND_OR_PARSING_ERROR sentinel is
        # deliberately excluded — it is injected by parser.py to signal
        # "I found no tests at all", not an actual test failure.
        if json_found and raw_tests:
            unmatched_failures = [
                t["name"]
                for t in raw_tests
                if t.get("status", "").upper() in ("FAILED", "ERROR")
                and not t.get("name", "").startswith("NO_TESTS_FOUND_OR_PARSING_ERROR")
            ]
            if unmatched_failures:
                logger.warning(
                    "[%s] parser found %d failing test(s) with no visible-test name match: %s",
                    container_id,
                    len(unmatched_failures),
                    ", ".join(unmatched_failures[:5]),
                )
                return (
                    False,
                    f"parser found {len(unmatched_failures)} failing test(s) that did "
                    f"not match any visible test name: " + ", ".join(unmatched_failures[:10]),
                    _format_test_output_text(raw_tests),
                )

        # Distinguish Ginkgo (empty stdout.log) from parser-format bug (non-empty):
        if test_args:
            stdout_size_tr = run_command(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    "wc -c < /tmp/stdout.log 2>/dev/null || echo 0",
                ],
                check=False,
            )
            stdout_empty = stdout_size_tr.stdout.strip() in ("0", "")
            if stdout_empty:
                # Go / Ginkgo: -run pattern never matched any test function.
                # The runner emitted "[no tests to run]" which the awk filter
                # in run_script.sh discarded → stdout.log is empty.
                # Fall back to the full suite to get a definitive pass/fail.
                logger.info(
                    "[%s] run_script.sh produced empty stdout for test_args=%r; "
                    "falling back to full-suite run for failure detection",
                    container_id,
                    test_args[:120],
                )
                full_tr = run_command(
                    [
                        "docker",
                        "exec",
                        container_id,
                        "bash",
                        "-c",
                        f"cd {repo_path} && bash /root/run_script.sh 2>&1",
                    ],
                    check=False,
                )
                full_output = (full_tr.stdout + full_tr.stderr).strip()
                if has_test_failures(full_output) or full_tr.returncode != 0:
                    return (
                        False,
                        "full-suite run reported test failure(s) "
                        "(specific test filtering produced no output — "
                        "run_script.sh -run pattern may not match test function names)",
                        None,
                    )
                return (
                    True,
                    "All tests passed (suite exit-code 0; "
                    "test framework does not support -run filtering — "
                    "individual spec names cannot be verified)",
                    None,
                )

        # stdout.log is non-empty (parser had output) or test_args is empty:
        # use saved exit code + raw output signals.
        context = (
            "parser extracted 0 results matching visible tests from non-empty output"
            if test_args
            else "no-filter run: parser extracted 0 results"
        )
        # Proposal B: flag when parser produced results but none matched visible tests.
        parser_had_unmatched = json_found and len(raw_tests) > 0
        return _classify_parser_no_results(
            container_id,
            context_label=context,
            visible_tests=visible_tests,
            parser_had_unmatched=parser_had_unmatched,
        )

    else:
        # No parser script — use exit code directly (no SWEAP JSON expected)
        if test_args:
            cmd = f"cd {repo_path} && bash /root/run_script.sh {test_args}"
        else:
            cmd = f"cd {repo_path} && bash /root/run_script.sh"
        tr = run_command(
            ["docker", "exec", container_id, "bash", "-c", cmd],
            check=False,
        )
        combined = (tr.stdout + tr.stderr).strip()
        if tr.returncode != 0:
            tail = combined[-3000:] if len(combined) > 3000 else combined
            return False, f"run_script.sh exited {tr.returncode}.\n{tail}", None
        return True, "All tests passed", None


def _create_keepalive_container(image_ref: str) -> str:
    for shell in ["/bin/bash", "/bin/sh", "bash", "sh"]:
        res = run_command(
            [
                "docker",
                "create",
                "--label",
                f"haltbench_owner_pid={os.getpid()}",
                "--entrypoint",
                shell,
                image_ref,
                "-lc",
                "while true; do sleep 3600; done",
            ],
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    raise RuntimeError(f"Failed to create keepalive container from {image_ref}")


def _copy_text_to_container(content: str, container_id: str, dest: str) -> None:
    suffix = Path(dest).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as fp:
        fp.write(content if content.endswith("\n") else content + "\n")
        tmp = fp.name
    try:
        run_command(["docker", "cp", tmp, f"{container_id}:{dest}"])
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_test_correctness(
    *,
    image_ref: str,
    agent_patch: str,
    visible_tests: list[str],
    run_script: str,
    parser_script: str,
    language: str,
    repo_path: str = _REPO_PATH,
    timeout_seconds: float = 600,
    support_setup_patch: bool = False,
    setup_script: str = "",
) -> TestCorrectnessResult:
    """Apply agent_patch to a fresh container and run the visible test suite.

    Pass ``visible_tests=[]`` to skip the test suite entirely (result is
    passed=True, visible_passed=True — no tests to fail).

    Flow (agent patch → run tests):
      1. Create a fresh container from image_ref.
      2. Reconcile git state to match the exact HEAD the agent worked against
         (support_setup_patch governs which strategy is used — see below).
      3. Apply agent_patch (the agent's changes) via ``git apply``.
      4. Copy run_script.sh and parser.py to /root/.
      5. If visible_tests is non-empty: run them via run_script.sh → visible_passed.
         If visible_tests is empty: skip; visible_passed=True (no tests to fail).

    Git reconciliation strategies
    ──────────────────────────────
    support_setup_patch=False (permanent pipeline):
      setup_script.sh committed everything at image-build time; HEAD is already
      clean.  We still run ``git reset HEAD && git checkout HEAD -- . &&
      git clean -fd`` to handle any residual staging artefacts, but it is
      effectively a no-op.

    support_setup_patch=True (legacy pipeline):
      The image was built with setup_patch.diff applied via ``git apply`` but NOT
      committed.  Those working-tree changes (e.g. test stubs) must be present
      when tests run, but ``git checkout HEAD -- .`` would destroy them.

      We mirror opencode_agent.py exactly:
        Dirty working tree → nuclear reset (wipe .git, reinit, commit
          everything as "initial state") then re-run setup_script.sh if
          non-empty.  This matches the HEAD the agent actually worked against.
        Clean working tree → current approach (no-op); setup_script.sh
          already committed everything at image-build time.

    Returns:
        TestCorrectnessResult with:
          - passed:        True iff the visible suite passed (or was skipped).
          - visible_passed: True/False. True when no visible tests were requested.

    Args:
        image_ref:            Docker image to create the container from.
        agent_patch:          The patch to evaluate (agent_patch.diff or golden_patch.diff).
        visible_tests:        Tests visible to the agent.  Pass [] to skip.
        run_script:           Contents of run_script.sh.
        parser_script:        Contents of parser.py.
        language:             Repository language (go, python, js, etc.).
        repo_path:            Path to the repo root inside the container.
        timeout_seconds:      Per-command timeout (currently unused; Docker handles it).
        support_setup_patch:  Whether this task uses the legacy setup_patch pipeline
                              (True → nuclear reset on dirty working tree).
        setup_script:         Contents of setup_script.sh; re-run after nuclear reset
                              when support_setup_patch=True and working tree is dirty.
    """
    if not image_ref:
        return TestCorrectnessResult(
            passed=False,
            detail="skipped: no image_ref available",
        )

    container_id: str | None = None
    try:
        container_id = _create_keepalive_container(image_ref)
        run_command(["docker", "start", container_id])

        # Configure git safe.directory (matches hil_bench_agent.py)
        run_command(
            [
                "docker",
                "exec",
                container_id,
                "git",
                "config",
                "--global",
                "--add",
                "safe.directory",
                repo_path,
            ],
            check=False,
        )

        # Reconcile git state to match the HEAD the agent worked against.
        #
        # Two strategies depending on support_setup_patch:
        #
        # support_setup_patch=False (permanent pipeline):
        #   setup_script.sh committed everything at image-build time; HEAD is clean.
        #   git reset HEAD + git checkout HEAD -- . + git clean -fd handles any residual
        #   staging artefacts from setup_script.sh (primer stubs staged but not committed,
        #   unstaged modifications, untracked files that conflict with git apply).
        #
        # support_setup_patch=True (legacy pipeline):
        #   setup_patch.diff was applied via `git apply` at image-build time but NOT
        #   committed.  Those working-tree changes include test stubs that must survive
        #   to test time.  `git checkout HEAD -- .` would silently destroy them.
        #   We mirror opencode_agent.py: check `git status --porcelain`.
        #     Dirty working tree → nuclear reset (rm .git, reinit, git add -A, commit
        #       "initial state") then re-run setup_script.sh (if non-empty).
        #       This recreates the exact HEAD the agent patched against.
        #     Clean working tree → fall through to the standard reset below
        #       (setup_script.sh already committed everything at image-build time).
        if support_setup_patch:
            quoted_repo = shlex.quote(repo_path)
            status_result = run_command(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    f"cd {quoted_repo} && git status --porcelain 2>/dev/null",
                ],
                check=False,
            )
            if status_result.stdout.strip():
                # Dirty working tree: nuclear reset to commit everything (including
                # setup_patch stubs) and recreate the git state the agent saw.
                logger.info(
                    "[%s] support_setup_patch=True with dirty working tree; "
                    "running nuclear reset at %s",
                    container_id,
                    repo_path,
                )
                nuclear_script = f"""\
set -euo pipefail
cd {quoted_repo}
rm -rf .git
git init
git symbolic-ref HEAD refs/heads/master
git config gc.auto 0
git config user.email "haltbench@eval.internal"
git config user.name "HaltBench"
git add -A
git commit --no-gpg-sign -m "initial state"
echo "[haltbench] Nuclear reset complete; HEAD=$(git rev-parse --short HEAD)"
"""
                run_command(
                    ["docker", "exec", container_id, "bash", "-c", nuclear_script],
                    check=False,
                )
                if setup_script.strip():
                    # Re-run setup_script.sh with a git-commit wrapper that exits 0
                    # on "nothing to commit" (mirrors opencode_agent.py exactly).
                    real_git_result = run_command(
                        ["docker", "exec", container_id, "bash", "-c", "command -v git"],
                        check=False,
                    )
                    real_git = real_git_result.stdout.strip() or "git"
                    # /halt_bench_task is only bind-mounted in the solver container;
                    # the keepalive test container has no bind mounts, so we must
                    # create the directory before copying into it.
                    run_command(
                        ["docker", "exec", container_id, "mkdir", "-p", "/halt_bench_task"],
                        check=False,
                    )
                    _copy_text_to_container(
                        setup_script, container_id, "/halt_bench_task/setup_script.sh"
                    )
                    run_command(
                        [
                            "docker",
                            "exec",
                            container_id,
                            "chmod",
                            "+x",
                            "/halt_bench_task/setup_script.sh",
                        ],
                        check=False,
                    )
                    setup_cmd = f"""\
set -euo pipefail
REAL_GIT={shlex.quote(real_git)}
mkdir -p /tmp/_hb_git_wrap
cat > /tmp/_hb_git_wrap/git << 'GITEOF'
#!/bin/bash
for arg in "$@"; do
  if [ "$arg" = "commit" ]; then
    PLACEHOLDER "$@"
    ec=$?
    if [ $ec -ne 0 ]; then
      PLACEHOLDER status 2>/dev/null | grep -qE "nothing to commit" && exit 0
    fi
    exit $ec
  fi
done
exec PLACEHOLDER "$@"
GITEOF
sed -i "s|PLACEHOLDER|$REAL_GIT|g" /tmp/_hb_git_wrap/git
chmod +x /tmp/_hb_git_wrap/git
cd {quoted_repo} && PATH=/tmp/_hb_git_wrap:$PATH bash /halt_bench_task/setup_script.sh {quoted_repo}
"""
                    run_command(
                        ["docker", "exec", container_id, "bash", "-c", setup_cmd],
                        check=False,
                    )
                # After nuclear reset the working tree is clean; skip the standard reset.
                _reconcile_done = True
            else:
                _reconcile_done = False
        else:
            _reconcile_done = False

        if not _reconcile_done:
            # Standard reconciliation for support_setup_patch=False (permanent pipeline)
            # or support_setup_patch=True with an already-clean working tree.
            run_command(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    f"cd {repo_path} && git reset HEAD && git checkout HEAD -- . && git clean -fd",
                ],
                check=False,
            )

        # Apply agent patch
        agent_filtered = _filter_patch(agent_patch)
        if agent_filtered.strip():
            _copy_text_to_container(agent_filtered, container_id, "/tmp/agent_patch.diff")
            ap = run_command(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    f"cd {repo_path} && git apply -v /tmp/agent_patch.diff 2>&1",
                ],
                check=False,
            )
            if ap.returncode != 0:
                detail = (
                    f"agent_patch.diff did not apply cleanly:\n{(ap.stdout + ap.stderr).strip()}"
                )
                return TestCorrectnessResult(
                    passed=False,
                    detail=detail,
                    visible_passed=False if visible_tests else True,
                )

        if not visible_tests:
            return TestCorrectnessResult(
                passed=True,
                detail="skipped: no visible tests",
                visible_passed=True,
            )

        # Copy run_script.sh and parser.py to /root/ (matches reference).
        # Proton webclient images have broken Yarn shims; use a direct Jest
        # wrapper that runs from the resolved workspace and preserves
        # "file | test name" visible IDs.
        effective_run_script = (
            _proton_visible_test_runner_script()
            if language.lower() in _JS_TS_LANGS and _is_proton_webclients_runner(run_script)
            else run_script
        )
        _copy_text_to_container(effective_run_script, container_id, "/root/run_script.sh")
        run_command(["docker", "exec", container_id, "chmod", "+x", "/root/run_script.sh"])
        if parser_script:
            _copy_text_to_container(parser_script, container_id, "/root/parser.py")

        visible_args = _build_test_args(visible_tests, language, run_script)
        if not visible_args:
            return TestCorrectnessResult(
                passed=True,
                detail="skipped: visible tests produced no runnable args",
                visible_passed=True,
            )

        vis_ok, vis_detail, test_output_text = _run_tests_in_container(
            container_id,
            visible_args,
            parser_script,
            repo_path,
            visible_tests=visible_tests,
        )
        is_collection_error = not vis_ok and vis_detail == _COLLECTION_ERROR_SENTINEL
        return TestCorrectnessResult(
            passed=vis_ok,
            detail=f"visible: {vis_detail}",
            visible_passed=vis_ok,
            tc_inconclusive=is_collection_error,
            test_output_text=test_output_text,
        )

    except Exception as exc:
        logger.exception("Unexpected error in run_test_correctness")
        return TestCorrectnessResult(passed=False, detail=f"error: {exc}", tc_inconclusive=True)
    finally:
        if container_id:
            run_command(["docker", "rm", "-fv", container_id], check=False)
