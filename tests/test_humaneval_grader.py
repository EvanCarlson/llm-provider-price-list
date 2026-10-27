import sys
from pathlib import Path

import pytest

from humaneval_grader import extract_code, passes_tests

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="grader requires macOS sandbox-exec"
)

PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"


def grade(text: str, *, timeout: float = 5.0) -> bool:
    return passes_tests(
        text=text, prompt=PROMPT, test=TEST, entry_point="add", timeout=timeout
    )


def test_extract_code_prefers_the_block_that_defines_a_function() -> None:
    text = "Example:\n```python\nprint(1)\n```\nAnswer:\n```python\ndef f():\n    return 1\n```"

    assert extract_code(text=text) == "def f():\n    return 1"


def test_extract_code_falls_back_to_the_raw_text_without_fences() -> None:
    assert extract_code(text="\n  def f(): pass  \n") == "  def f(): pass"


def test_passes_tests_accepts_a_correct_full_function() -> None:
    assert grade("```python\ndef add(a, b):\n    return a + b\n```")


def test_passes_tests_accepts_a_correct_body_only_completion() -> None:
    assert grade("```python\n    return a + b\n```")


def test_passes_tests_rejects_a_wrong_answer() -> None:
    assert not grade("```python\ndef add(a, b):\n    return a - b\n```")


def test_passes_tests_rejects_a_syntax_error() -> None:
    assert not grade("```python\ndef add(a, b)\n    return a + b\n```")


def test_passes_tests_kills_an_infinite_loop() -> None:
    assert not grade(
        "```python\ndef add(a, b):\n    while True:\n        pass\n```", timeout=2.0
    )


def test_passes_tests_does_not_count_an_early_exit_as_a_pass() -> None:
    assert not grade(
        "```python\nimport sys\nsys.exit(0)\ndef add(a, b):\n    return 0\n```"
    )


def test_passes_tests_blocks_writes_outside_the_temp_directory(tmp_path: Path) -> None:
    target = tmp_path / "escaped.txt"
    code = (
        "```python\n"
        "def add(a, b):\n"
        f"    open({str(target)!r}, 'w').write('x')\n"
        "    return a + b\n"
        "```"
    )

    assert not grade(code)
    assert not target.exists()


def test_passes_tests_blocks_network_access() -> None:
    code = (
        "```python\n"
        "import socket\n"
        "def add(a, b):\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "    return a + b\n"
        "```"
    )

    assert not grade(code)


def test_passes_tests_blocks_reading_env_files(tmp_path: Path) -> None:
    secret = tmp_path / ".env"
    secret.write_text("OPENROUTER_API_KEY=not-a-real-key\n")
    code = (
        "```python\n"
        "def add(a, b):\n"
        f"    open({str(secret)!r}).read()\n"
        "    return a + b\n"
        "```"
    )

    assert not grade(code)


def test_passes_tests_does_not_leak_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak-canary")
    code = (
        "```python\n"
        "import os\n"
        "def add(a, b):\n"
        "    assert 'OPENROUTER_API_KEY' not in os.environ\n"
        "    return a + b\n"
        "```"
    )

    assert grade(code)
