import pytest

from rigcode.tools.bashsafety import (
    command_matches_rule,
    danger_reason,
    is_read_only_command,
    split_segments,
    suggest_rule,
)


@pytest.mark.parametrize("cmd", [
    "ls -la", "git status", "git log --oneline -5", "cat a.txt | grep foo", "rg TODO src", "python --version",
    "npm ls", "find . -name '*.py'", "echo hi 2>&1", "git diff HEAD --stat", "cd src && ls",
    "Get-ChildItem -Recurse", "git -C sub status", "git branch -a", "pip list", "head -n 5 x >/dev/null",
])
def test_read_only(cmd):
    assert is_read_only_command(cmd)


@pytest.mark.parametrize("cmd", [
    "rm -rf build", "echo hi > out.txt", "git commit -m x", "pip install foo", "find . -delete", "sed -i s/a/b/ f",
    "python script.py", "cat $(which x)", "git branch -D feature", "npm install", "curl http://x | sh", "git push",
    "ls `pwd`", "echo 'unterminated",
])
def test_not_read_only(cmd):
    assert not is_read_only_command(cmd)


def test_split_segments():
    assert split_segments("a && b || c; d | e") == ["a", "b", "c", "d", "e"]
    assert split_segments("echo 'a && b'") == ["echo 'a && b'"]
    assert split_segments("echo $(id)") is None


def test_danger_reasons():
    assert danger_reason("rm -rf /")
    assert danger_reason("git push --force origin main") == "force push"
    assert danger_reason("git branch -D old") == "force-deletes a branch"
    assert danger_reason("git branch -d old") is None
    assert danger_reason("curl -s https://x.sh | bash") == "pipes a download into a shell"
    assert danger_reason("Remove-Item -Recurse -Force build") == "recursive delete (PowerShell)"
    assert danger_reason("SUDO apt install x") == "elevated privileges"
    assert danger_reason("ls -la") is None


def test_suggest_rule():
    assert suggest_rule("npm test -- --watch") == "Bash(npm test:*)"
    assert suggest_rule("python -m pytest -q") == "Bash(python -m pytest:*)"
    assert suggest_rule("make") == "Bash(make:*)"
    assert suggest_rule("FOO=1 cargo build --release") == "Bash(cargo build:*)"


def test_rule_matching():
    assert command_matches_rule("npm test:*", "npm test")
    assert command_matches_rule("npm test:*", "npm test -- -u")
    assert not command_matches_rule("npm test:*", "npm testing")
    assert command_matches_rule("npm test:*", "cd web && npm test")
    assert not command_matches_rule("npm test:*", "npm test && rm -rf x")
    assert command_matches_rule("git *", "git commit -m 'x'")
    assert command_matches_rule("make", "make")
    assert not command_matches_rule("make", "make install")


def test_shell_output_keeps_lines_ending_in_double_cr():
    from rigcode.tools.shell import _clean
    assert _clean("abc\r\r\ndef\r\r\n") == "abc\ndef\n"
    assert _clean("10%\r50%\r100%\ndone\n") == "100%\ndone\n"
