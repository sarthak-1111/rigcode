import json

from rigcode.agent.session import SessionStore
from rigcode.config import add_permission_rule, load_settings, settings_paths
from rigcode.extensions import load_agents, load_commands, load_memory, load_skills
from rigcode.ui.render import BlockSplitter, render_diff
from rigcode.utils.jsonrepair import loads_lenient


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_settings_layering(project, isolated_home, monkeypatch):
    paths = settings_paths(project)
    write_json(paths["user"], {"provider": {"model": "user-model", "reasoning_effort": "high"},
                               "permissions": {"allow": ["Bash(ls:*)"]}})
    write_json(paths["project"], {"provider": {"model": "project-model"}, "permissions": {"allow": ["Read"]}})
    write_json(paths["local"], {"permissions": {"deny": ["Bash(rm:*)"]}})
    write_json(project / ".mcp.json", {"mcpServers": {"fs": {"command": "x"}}})
    monkeypatch.setenv("RIGCODE_EFFORT", "max")
    s = load_settings(project, {"provider.temperature": 0.5})
    assert s.get("provider.model") == "project-model"
    assert s.get("provider.reasoning_effort") == "max"
    assert s.get("provider.temperature") == 0.5
    assert s.get("permissions.allow") == ["Bash(ls:*)", "Read"]
    assert s.get("permissions.deny") == ["Bash(rm:*)"]
    assert "fs" in s.get("mcp_servers")
    assert s.get("provider.base_url") == "auto"


def test_profiles_and_rule_persistence(project):
    write_json(settings_paths(project)["project"], {"profiles": {"fast": {"model": "small", "reasoning_effort": "low"}}})
    s = load_settings(project, {"profile": "fast"})
    assert s.get("provider.model") == "small"
    path = add_permission_rule(project, "Bash(npm test:*)")
    assert json.loads(path.read_text())["permissions"]["allow"] == ["Bash(npm test:*)"]


def test_session_roundtrip_and_repair(project):
    store = SessionStore(project)
    store.meta(id=store.id, title="t")
    store.message({"role": "user", "content": "hello"})
    store.message({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]})
    with open(store.path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "message", "message": {"role": "tool"')    # torn last line
    meta, msgs = SessionStore.load(store.path)
    assert meta["id"] == store.id
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[-1]["tool_call_id"] == "c1"
    store.compact([{"role": "system", "content": "s"}, {"role": "user", "content": "summary"}])
    _, msgs = SessionStore.load(store.path)
    assert [m["content"] for m in msgs] == ["s", "summary"]
    assert SessionStore.resolve(project, store.id[-6:]) == store.path
    assert SessionStore.list(project)[0].id == store.id


def test_extensions(project, isolated_home):
    (project / ".claude" / "agents").mkdir(parents=True)
    (project / ".claude" / "agents" / "docs.md").write_text(
        "---\nname: docs\ndescription: Writes docs\ntools: Read, Grep\nmodel: sonnet\n---\nYou write docs.", encoding="utf-8")
    (project / ".rigcode" / "skills" / "deploy").mkdir(parents=True)
    (project / ".rigcode" / "skills" / "deploy" / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: How to deploy\n---\nRun make deploy.", encoding="utf-8")
    (project / ".rigcode" / "commands" / "git").mkdir(parents=True)
    (project / ".rigcode" / "commands" / "git" / "sync.md").write_text(
        "---\ndescription: Sync\n---\nSync branch $1 with $ARGUMENTS", encoding="utf-8")
    (project / "notes.md").write_text("imported note", encoding="utf-8")
    (project / "AGENTS.md").write_text("Use tabs.\n@notes.md", encoding="utf-8")
    s = load_settings(project)
    agents = load_agents(s, project)
    assert {"explore", "plan", "coder", "reviewer", "debugger", "researcher", "verifier"} <= set(agents)
    assert agents["docs"].read_only and agents["docs"].tools == ["Read", "Grep"] and agents["docs"].model is None
    skills = load_skills(s, project)
    assert skills["deploy"].load_body() == "Run make deploy."
    cmds = load_commands(s, project)
    assert cmds["git:sync"].render("main now", project) == "Sync branch main with main now"
    mem = load_memory(s, project, project)
    assert len(mem) == 1 and "Use tabs." in mem[0].content and "imported note" in mem[0].content


def test_block_splitter_and_diff():
    sp = BlockSplitter()
    assert sp.feed("Para one") == []
    assert sp.feed(" continues\n\nPara two\n") == ["Para one continues"]
    assert sp.feed("```py\nx = 1\n\ny = 2\n") == ["Para two"]
    assert sp.feed("```\n") == ["```py\nx = 1\n\ny = 2\n```"]
    assert sp.flush() == ""
    diff = "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c"
    text = render_diff(diff).plain
    assert "- b" in text and "+ B" in text and "2 " in text


def test_json_repair():
    assert loads_lenient('{"a": 1,}')[0] == {"a": 1}
    assert loads_lenient('```json\n{"a": "x\ny"}\n```')[0] == {"a": "x\ny"}
    assert loads_lenient('{"a": [1, 2')[0] == {"a": [1, 2]}
    value, err = loads_lenient("not json")
    assert value is None and err


def test_repo_overview_is_stable_when_files_are_added(project):
    from rigcode.agent.prompts import repo_overview
    (project / "src" / "pkg").mkdir(parents=True)
    for i in range(42):
        (project / "src" / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
    (project / "src" / "pkg" / "a.py").write_text("a = 1\n", encoding="utf-8")
    before = repo_overview(project)
    (project / "src" / "pkg" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert repo_overview(project) == before          # rounded counts keep the prompt prefix (and cache) intact
    assert "src/ (~40 files)" in before and "  pkg/" in before
