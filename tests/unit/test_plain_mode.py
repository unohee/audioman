# Created: 2026-05-11
# Purpose: --plain / AUDIOMAN_PLAIN 출력 모드 회귀 테스트.
# LLM agent 후기 #1 대응: --help가 ANSI/색상 토큰 없이 영어로 출력돼야 한다.

import os
import re
import subprocess
import sys

import pytest


ANSI_RE = re.compile(r"\x1b\[")


def _run(args, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "audioman", *args],
        env=env,
        capture_output=True,
        text=True,
    )


class TestPlainMode:
    def test_plain_flag_strips_ansi_from_help(self):
        result = _run(["--plain", "--help"])
        assert result.returncode == 0
        assert ANSI_RE.search(result.stdout) is None, (
            "Plain help should not contain ANSI escape sequences"
        )

    def test_plain_help_is_english(self):
        result = _run(["--plain", "--help"])
        assert result.returncode == 0
        assert "Available commands" in result.stdout

    def test_env_var_alone_enables_plain(self):
        result = _run(["--help"], env_extra={"AUDIOMAN_PLAIN": "1"})
        assert result.returncode == 0
        assert ANSI_RE.search(result.stdout) is None
        assert "Available commands" in result.stdout
