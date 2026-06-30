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
# collected (ImportError, SyntaxError, or missing module at import time).
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


def _run_tests_in_container(
    container_id: str,
    test_args: str,
    parser_script: str,
    repo_path: str = _REPO_PATH,
) -> tuple[bool, str]:
    """Run run_script.sh (at /root/run_script.sh) and parse results.

    Mirrors hil_bench_agent.py _validate_swe_fast step 6 exactly:
      - Scripts at /root/  (not /tmp/)
      - Uses 'python' (not 'python3')
      - Output to /tmp/output.json  (not /tmp/test_output.json)
      - bash -c for exec  (not -lc)
      - Combined stdout+stderr for SWEAP JSON search

    Returns (passed, detail_string).
    """
    if parser_script:
        if test_args:
            test_cmd = (
                f"bash /root/run_script.sh {test_args} > /tmp/stdout.log 2> /tmp/stderr.log; "
                "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
                "echo 'SWEAP_JSON_START'; cat /tmp/output.json; echo 'SWEAP_JSON_END'"
            )
        else:
            test_cmd = (
                "bash /root/run_script.sh > /tmp/stdout.log 2> /tmp/stderr.log; "
                "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
                "echo 'SWEAP_JSON_START'; cat /tmp/output.json; echo 'SWEAP_JSON_END'"
            )
        tr = run_command(
            ["docker", "exec", container_id, "bash", "-c", f"cd {repo_path} && {test_cmd}"],
            check=False,
        )
        raw = tr.stdout + tr.stderr

        data = _parse_sweap_json(raw)
        if data is not None:
            tests = data.get("tests", [])
            failed = [
                f"{t.get('name', '?')} ({t.get('status', '?')})"
                for t in tests
                if t.get("status", "").upper() in ("FAILED", "ERROR")
            ]
            if failed:
                return False, (
                    f"parser.py reports {len(failed)} failing test(s):\n"
                    + "\n".join(f"  - {f}" for f in failed[:20])
                )
            # If specific tests were requested but none were collected, the test
            # module likely failed to import (ImportError, SyntaxError, etc.).
            # Signal this as a special sentinel string so the caller can distinguish
            # a collection error from a genuine agent test failure.
            if not tests and test_args:
                return False, _COLLECTION_ERROR_SENTINEL
            passed_count = sum(1 for t in tests if t.get("status", "").upper() == "PASSED")
            return True, f"All {passed_count} test(s) passed"

        logger.warning("SWEAP_JSON unparseable — falling back to exit-code check")

        # Fallback: re-run without parser to get a clean exit code
        fallback_tr = run_command(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                f"cd {repo_path} && bash /root/run_script.sh {test_args}",
            ],
            check=False,
        )
        tail = (fallback_tr.stdout + fallback_tr.stderr).strip()
        if fallback_tr.returncode != 0:
            return False, (
                f"parser.py output unparseable; fallback exit code {fallback_tr.returncode}.\n"
                + tail[-3000:]
            )
        return True, "All tests passed (exit code fallback)"

    else:
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
            return False, f"run_script.sh exited {tr.returncode}.\n{tail}"
        return True, "All tests passed"


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
        fp.write(content)
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

        # Copy run_script.sh and parser.py to /root/ (matches reference)
        _copy_text_to_container(run_script, container_id, "/root/run_script.sh")
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

        vis_ok, vis_detail = _run_tests_in_container(
            container_id, visible_args, parser_script, repo_path
        )
        is_collection_error = not vis_ok and vis_detail == _COLLECTION_ERROR_SENTINEL
        return TestCorrectnessResult(
            passed=vis_ok,
            detail=f"visible: {vis_detail}",
            visible_passed=vis_ok,
            tc_inconclusive=is_collection_error,
        )

    except Exception as exc:
        logger.exception("Unexpected error in run_test_correctness")
        return TestCorrectnessResult(passed=False, detail=f"error: {exc}")
    finally:
        if container_id:
            run_command(["docker", "rm", "-fv", container_id], check=False)
