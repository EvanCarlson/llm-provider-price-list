"""Grade model-written HumanEval completions by running them against the task's tests.

Model output is untrusted code. Each completion runs in its own interpreter
(python -I) inside a fresh temp directory, wrapped in macOS sandbox-exec with
network access denied, writes allowed only inside that temp directory, and
reads denied for .env files and common credential directories. The process gets
a CPU limit, a memory limit where the OS honours it, a wall-clock timeout, and
is killed as a process group. This is process isolation, not a hardened
sandbox: do not run it on a machine where a determined attacker would matter.

A completion passes only if the program exits 0 and prints a per-run random
sentinel after check() returns, so a stray exit(0) cannot count as a pass.
"""

import os
import re
import resource
import secrets
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 10.0
CPU_LIMIT_SECONDS = 10
MEMORY_LIMIT_BYTES = 2 * 1024**3
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
CREDENTIAL_DIRS = [".ssh", ".aws", ".claude", ".config", ".gnupg"]
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OUTPUT_TAIL_BYTES = 200


def extract_code(*, text: str) -> str:
    """The first fenced block that defines a function, else the first block, else the text."""
    blocks = CODE_BLOCK.findall(text)
    if not blocks:
        return text.strip("\n").rstrip()
    with_def = [b for b in blocks if "def " in b]

    # Strip only blank lines: a body-only completion needs its leading indent.
    return (with_def or blocks)[0].strip("\n").rstrip()


def build_program(
    *, prompt: str, completion: str, test: str, entry_point: str, sentinel: str
) -> str:
    """Prompt, then the completion (a redefinition or a body continuation), then the tests.

    HumanEval prompts can define helper functions the tests rely on, so the
    prompt is kept and the completion is appended after it.
    """
    return (
        f"{prompt}\n{completion}\n\n{test}\n\n"
        f"check({entry_point})\nprint({sentinel!r})\n"
    )


def sandbox_profile(*, workdir: Path) -> str:
    home = Path.home()
    credentials = " ".join(f'(subpath "{home / d}")' for d in CREDENTIAL_DIRS)

    return (
        "(version 1)(allow default)(deny network*)"
        f'(deny file-write* (require-not (require-any (subpath "{workdir}") '
        '(literal "/dev/null") (literal "/dev/tty"))))'
        f'(deny file-read* {credentials} (regex #"\\.env$"))'
    )


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        pass


def _clean_env() -> dict[str, str]:
    """No inherited environment, so no API keys reach the program."""
    return {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}


def run_program(*, program: str, sentinel: str, timeout: float) -> bool:
    if not SANDBOX_EXEC.exists():
        raise RuntimeError(
            f"{SANDBOX_EXEC} not found; refusing to run untrusted code unsandboxed"
        )
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp).resolve()
        (workdir / "prog.py").write_text(program)
        out_path = workdir / "out.txt"
        with out_path.open("w") as out:
            proc = subprocess.Popen(
                [
                    str(SANDBOX_EXEC),
                    "-p",
                    sandbox_profile(workdir=workdir),
                    sys.executable,
                    "-I",
                    "prog.py",
                ],
                cwd=workdir,
                env=_clean_env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=_limit_resources,  # noqa: PLW1509 (grader is single-threaded)
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                returncode = None
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        if returncode != 0:
            return False
        tail = out_path.read_bytes()[-OUTPUT_TAIL_BYTES:].decode(errors="replace")

        return tail.strip().endswith(sentinel)


def passes_tests(
    *,
    text: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    sentinel = secrets.token_hex(16)
    program = build_program(
        prompt=prompt,
        completion=extract_code(text=text),
        test=test,
        entry_point=entry_point,
        sentinel=sentinel,
    )

    return run_program(program=program, sentinel=sentinel, timeout=timeout)
