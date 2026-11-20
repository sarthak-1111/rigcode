"""Run the real CLI (`python -m rigcode -p ...`) as a subprocess against the mock server."""
import json
import os
import subprocess
import sys
from pathlib import Path

from mock_server import Reply

ROOT = Path(__file__).resolve().parent.parent


def rigcode(args, cwd, home, stdin=None, timeout=60, **extra_env):
    env = {**os.environ, "RIGCODE_HOME": str(home), "PYTHONPATH": str(ROOT), **extra_env}
    return subprocess.run([sys.executable, "-m", "rigcode", *args], cwd=cwd, env=env, input=stdin,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def test_print_mode_json(server, project, isolated_home):
    server.add(Reply(tool_calls=[("Write", {"file_path": "out.txt", "content": "made by rigcode\n"})]),
               Reply(content="Created out.txt."))
    proc = rigcode(["-p", "create out.txt", "--base-url", server.base_url, "--mode", "acceptEdits",
                  "--output-format", "json"], project, isolated_home)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["result"] == "Created out.txt." and data["tool_calls"] == 1
    assert (project / "out.txt").read_text(encoding="utf-8") == "made by rigcode\n"


def test_print_mode_needs_approval_is_reported(server, project, isolated_home):
    server.add(Reply(tool_calls=[("Bash", {"command": "npm install"})]), Reply(content="Could not install."))
    proc = rigcode(["-p", "install deps", "--base-url", server.base_url], project, isolated_home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Could not install."
    tool_msg = [m for m in server.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "needs the user's approval" in tool_msg["content"]


def test_stream_json_and_stdin(server, project, isolated_home):
    server.add(Reply(content="Summary of input."))
    proc = rigcode(["-p", "summarize:", "--base-url", server.base_url, "--output-format", "stream-json"],
                 project, isolated_home, stdin="some piped text")
    assert proc.returncode == 0, proc.stderr
    events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "system" and kinds[-1] == "result" and "assistant" in kinds
    assert "some piped text" in server.requests[0]["messages"][-1]["content"]


def test_doctor_unreachable(project, isolated_home):
    proc = rigcode(["doctor", "--quick"], project, isolated_home, RIGCODE_BASE_URL="http://127.0.0.1:9/v1")
    assert proc.returncode == 1
    assert "Model server" in proc.stdout and "TCP 127.0.0.1:9" in proc.stdout


def test_doctor_against_mock(server, project, isolated_home):
    proc = rigcode(["doctor", "--quick"], project, isolated_home, RIGCODE_BASE_URL=server.base_url)
    assert proc.returncode == 0, proc.stdout
    assert "llama.cpp" in proc.stdout


def test_doctor_takes_server_flags(server, project, isolated_home):
    proc = rigcode(["doctor", "--quick", "--base-url", server.base_url, "-m", "mock-model"], project, isolated_home)
    assert proc.returncode == 0, proc.stdout
    assert f"Model server: {server.base_url}" in proc.stdout and "✓ models: mock-model" in proc.stdout
    proc = rigcode(["doctor", "--model-typo", "x"], project, isolated_home)
    assert proc.returncode == 2 and "unrecognized arguments" in proc.stderr


def test_sessions_and_config_subcommands(project, isolated_home):
    proc = rigcode(["config", "set", "provider.reasoning_effort", "high"], project, isolated_home)
    assert proc.returncode == 0, proc.stderr
    proc = rigcode(["config", "get", "provider.reasoning_effort"], project, isolated_home)
    assert proc.stdout.strip() == '"high"'
    proc = rigcode(["sessions"], project, isolated_home)
    assert proc.returncode == 0
