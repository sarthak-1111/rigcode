from rigcode.config import load_settings
from rigcode.permissions import PermissionManager, normalize_mode
from rigcode.tools.base import Tool
from rigcode.tools.files import EditTool, ReadTool
from rigcode.tools.interact import ExitPlanModeTool
from rigcode.tools.shell import BashTool, ShellSpec
from rigcode.tools.web import WebFetchTool

BASH = BashTool(ShellSpec("bash", "bash", "bash"))


class FakeMcp(Tool):
    name = "mcp__github__create_issue"
    kind = "mcp"


def pm(project, **perms):
    settings = load_settings(project, {f"permissions.{k}": v for k, v in perms.items()})
    return PermissionManager(settings, project)


def verdict(p, tool, args, project):
    return p.evaluate(tool, args, project)[0]


def test_default_mode(project):
    p = pm(project)
    assert verdict(p, ReadTool(), {"file_path": "a.py"}, project) == "allow"
    assert verdict(p, EditTool(), {"file_path": "a.py"}, project) == "ask"
    assert verdict(p, BASH, {"command": "ls -la"}, project) == "allow"
    assert verdict(p, BASH, {"command": "npm install"}, project) == "ask"
    assert verdict(p, WebFetchTool(), {"url": "https://example.com"}, project) == "ask"


def test_accept_edits_mode(project, tmp_path):
    p = pm(project, mode="acceptEdits")
    assert verdict(p, EditTool(), {"file_path": "src/a.py"}, project) == "allow"
    outside = tmp_path / "elsewhere.py"
    v, reason = p.evaluate(EditTool(), {"file_path": str(outside)}, project)
    assert v == "ask" and "outside the workspace" in reason
    assert verdict(p, EditTool(), {"file_path": ".rigcode/settings.json"}, project) == "ask"


def test_plan_mode(project):
    p = pm(project, mode="plan")
    assert verdict(p, EditTool(), {"file_path": "a.py"}, project) == "deny"
    assert verdict(p, BASH, {"command": "rm x"}, project) == "deny"
    assert verdict(p, BASH, {"command": "git status"}, project) == "allow"
    assert verdict(p, ExitPlanModeTool(), {"plan": "x"}, project) == "allow"


def test_bypass_and_deny_rules(project):
    p = pm(project, mode="bypass", deny=["Bash(rm:*)"])
    assert verdict(p, BASH, {"command": "npm install"}, project) == "allow"
    assert verdict(p, BASH, {"command": "rm -rf x"}, project) == "deny"


def test_allow_and_ask_rules(project):
    p = pm(project, allow=["Bash(npm test:*)", "Read"], ask=["Read(secrets/**)"])
    assert verdict(p, BASH, {"command": "npm test -- -u"}, project) == "allow"
    assert verdict(p, ReadTool(), {"file_path": "secrets/key.txt"}, project) == "ask"
    assert verdict(p, ReadTool(), {"file_path": "src/a.py"}, project) == "allow"


def test_webfetch_domain_rule(project):
    p = pm(project, allow=["WebFetch(domain:python.org)"])
    assert verdict(p, WebFetchTool(), {"url": "https://docs.python.org/3/"}, project) == "allow"
    assert verdict(p, WebFetchTool(), {"url": "https://example.com"}, project) == "ask"


def test_mcp_server_rule(project):
    p = pm(project, allow=["mcp__github"])
    assert verdict(p, FakeMcp(), {}, project) == "allow"


def test_session_rules_and_modes(project):
    p = pm(project)
    p.remember("Bash(make:*)", persist=False)
    assert verdict(p, BASH, {"command": "make all"}, project) == "allow"
    assert p.cycle_mode() == "acceptEdits"
    assert p.cycle_mode() == "plan"
    assert p.cycle_mode() == "default"
    assert normalize_mode("yolo") == "bypass"
    assert normalize_mode("acceptedits") == "acceptEdits"
