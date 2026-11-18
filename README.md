# rigcode

**A terminal coding agent for the model on your own rig.**

rigcode reads your code, edits files, runs commands and tests, and delegates to subagents, the way commercial coding agents do. The model it uses runs on your hardware: llama.cpp, Ollama, LM Studio, vLLM, SGLang, or any OpenAI-compatible server. Your code never leaves your machines, and rigcode has no telemetry.

It is built for the case other agents handle badly: **a big model on modest hardware**. When your server reads prompts at 30 tokens a second, re-reading a 4,000-token conversation costs two minutes, and most agents re-read it on every turn. rigcode is designed around how llama.cpp's prompt cache actually works, so each step re-reads only the new tokens.

<!-- Demo GIF goes here: docs/demo.gif -->

```bash
pipx install rigcode          # or: uv tool install rigcode
cd your-project
rigcode                       # finds a local llama.cpp, Ollama, LM Studio or vLLM server on its own
```

## Why rigcode

- **Fast on slow hardware.** The prompt prefix stays fixed for the whole session and the history is append-only, so the server's cache always applies. Subagents fork the cached conversation instead of starting cold. A background warm-up reads the prompt while you type. The details and measurements are below.
- **Zero setup for local servers.** With no configuration, rigcode checks the default ports of llama.cpp (8080), Ollama (11434), LM Studio (1234) and vLLM/SGLang (8000), and uses the model the server has loaded. `rigcode doctor` explains anything that's wrong, down to the firewall command to run.
- **A full agent, not a chat window.** Read, Edit, Write, Grep, Glob and Bash tools; background shells; seven built-in subagents; plan mode; a todo list; `/undo`; session resume; web search and fetch.
- **Safe by default.** It asks before editing files, running commands or using the web. Read-only commands such as `git status` and `ls` run without asking. Allow and deny rules let you tune it per project.
- **Bring your existing setup.** It reads `AGENTS.md` and `CLAUDE.md`, plus custom commands, agents and skills in `.claude/`. Hooks use the same event names and JSON format, and MCP servers can come from `.mcp.json`. Permission rules use the familiar `Bash(npm test:*)` syntax.
- **Works with imperfect models.** If the server rejects native tool calls, rigcode falls back to text-based ones. It recovers Kimi-, Hermes- and Qwen-style tool calls that leak into the reply text, and it repairs broken JSON arguments.

## Benchmarks

Both agents used the same model and server: Kimi K3 (2.78T parameters) on llama.cpp, on one machine on the local network. The tasks are six SWE-bench Verified issues from Django and SymPy, graded with the maintainers' hidden tests at maximum reasoning effort. Every run was audited for internet use and for reading files outside its own checkout.

| | rigcode | Claude Code (same model) |
|---|---|---|
| Issues resolved, clean runs | **4 / 6** | **4 / 5** |
| Runs excluded | 0 | 1 (read a later Django release that contains the fix) |

On six tasks that's a tie, not a win for either. In an earlier test on five small tasks at low effort, both solved all five, and rigcode took 30% less time (41 vs 59 min) and generated 34% fewer tokens. [Method, per-task results and caveats](docs/benchmarks.md).

## Quick start

Requires Python 3.10+. Install with `pipx install rigcode`, `uv tool install rigcode`, or `pip install rigcode`.

**Ollama**

```bash
ollama pull gpt-oss:20b                 # any model that supports tool calling
OLLAMA_CONTEXT_LENGTH=32768 ollama serve   # Ollama's default context is too small for an agent
rigcode -m gpt-oss:20b
```

**llama.cpp**

```bash
llama-server -m model.gguf --jinja -c 65536 --port 8080   # --jinja enables tool calling
rigcode
```

**Any other OpenAI-compatible server**, local or remote:

```bash
rigcode --base-url http://gpu-box:8000/v1 -m qwen3-coder --api-key "$KEY"
```

To make that permanent: `rigcode config set provider.base_url http://gpu-box:8000/v1 --user`.

Optional: install [ripgrep](https://github.com/BurntSushi/ripgrep) for faster search. On Windows, rigcode uses Git Bash when it's installed and PowerShell otherwise.

## Use

```bash
rigcode                                 # interactive session in the current project
rigcode "fix the failing test in tests/test_api.py"   # start with a prompt
rigcode -c                              # continue the last session here; rigcode -r [id] to pick one
rigcode -p "explain src/auth.py" --output-format json # headless: run one task and print the result
git diff | rigcode -p "review this diff"             # stdin is appended to the prompt
rigcode -e high                         # reasoning effort: low | medium | high | max | none
rigcode --mode plan                     # read-only until you approve a plan
rigcode doctor                          # diagnose the server connection and local setup
```

### Keys and input

| Key / input | Action |
|---|---|
| Enter | Send. While the agent works, your message is **queued and delivered at its next step**, so you can steer it mid-task. |
| Alt+Enter, Ctrl+J, `\` + Enter | New line |
| Esc or Ctrl+C | Interrupt the agent. Ctrl+C twice on an empty prompt (or Ctrl+D) exits. |
| Shift+Tab | Cycle the permission mode: default → accept edits → plan |
| Ctrl+O / Ctrl+T | Toggle the full thinking display / the todo list |
| `@path` | Attach a file or directory listing to your message (Tab completes). This saves the model a tool round trip. |
| `!command` | Run a shell command yourself; the agent sees the output next turn |
| `#note` | Append a line to the project's `AGENTS.md` |

### Slash commands

`/help` `/clear` `/compact [focus]` `/context` `/cost` `/status` `/model [profile]` `/effort [level]` `/mode [mode]` `/plan [task]` `/permissions [allow|deny|ask RULE]` `/add-dir PATH` `/tools` `/agents` `/skills` `/mcp` `/memory` `/init` `/review [focus]` `/commit` `/diff` `/undo` `/todos` `/resume [id]` `/export [path]` `/thinking [collapsed|full|hidden]` `/config [key [value]]` `/doctor` `/bashes` `/exit`

Custom commands are Markdown files in `.rigcode/commands/` or `.claude/commands/`. They support `$ARGUMENTS`, `$1`…`$9`, and `` !`cmd` `` when the file's `allowed-tools` includes Bash.

## How it stays fast on slow hardware

These measurements come from Kimi K3 on llama.cpp, where prompt processing runs at about 10–35 tokens/s and generation at about 5 tokens/s.

- **Reuse happens only at checkpoints.** Hybrid models such as Kimi K3 can only rewind the cache to a checkpoint, and llama.cpp makes those at the start of user messages (plus two at the end of each prompt). So rigcode keeps everything before the first user message fixed for the whole session: the tool definitions and one system message with the core prompt and stable project context. Volatile facts (date, git status, recent commits) go in a reminder in your first message, and the history is append-only.
- **No slot pinning.** With its default unified KV cache, llama.cpp saves every idle slot to a RAM cache and clears the slot when a request starts. Only an unpinned request gets its saved state loaded back, so rigcode lets the server choose the slot. Measured: a 1,052-token prompt was restored in 1.2 s after another slot had been used.
- **Subagents fork the conversation.** A subagent continues from the main agent's cached prompt, and its brief arrives as the result of the Task call. When it finishes, the main agent continues from the subagent's state, which still begins with the whole main conversation. Measured: the subagent's first request re-read about 500 of 4,147 tokens, and the main agent's next one re-read 124 of 3,770. A subagent starts fresh only when it can't fork (a custom agent with its own model or effort, or a conversation over half the context window).
- **Background warm-up.** While you type your first message, rigcode processes the prompt prefix, or the whole history after `--continue`. Your first request waits for it rather than racing it on another slot.
- **Compaction triggers on an absolute token count** (`context.compact_at_tokens`, default 100k). The summary request reuses the cached conversation.
- **Live progress.** The status line shows prompt processing (`Reading prompt 45% … ~1m 10s left`), thinking, generation speed and tool activity. Read timeouts are never retried automatically, because disconnecting cancels the server's work.
- Tool output is capped and middle-truncated, with the full output saved to a file. Independent read-only tool calls run in parallel.

If you start llama-server with `--no-cache-idle-slots`, idle slots keep their cache in VRAM. You can then set `provider.llama_cpp.slot_pinning: true` to give the main agent and each fresh subagent its own slot.

## Permissions

Modes: `default` asks before edits, commands and web access. `acceptEdits` auto-approves edits inside the workspace. `plan` is read-only until you approve a plan. `bypass` allows everything except deny rules.

Read-only shell commands (`ls`, `git status`, `rg`, …) are approved automatically; anything with output redirection or command substitution asks first. Rules live in `.rigcode/settings.local.json` (personal) or `.rigcode/settings.json` (shared with the repo):

```json
{ "permissions": {
    "allow": ["Bash(npm test:*)", "Bash(python -m pytest:*)", "Edit(src/**)", "WebFetch(domain:docs.python.org)", "mcp__github"],
    "deny":  ["Bash(git push:*)", "Read(.env)"],
    "ask":   ["Bash(git commit:*)"] } }
```

Choosing "don't ask again" in a permission prompt saves the suggested rule. `/undo` reverts the files changed in the last turn.

## Configuration

Layers, later wins: built-in defaults < `~/.rigcode/settings.json` < `.rigcode/settings.json` < `.rigcode/settings.local.json` < environment (`RIGCODE_BASE_URL`, `RIGCODE_MODEL`, `RIGCODE_API_KEY`, `RIGCODE_EFFORT`, `RIGCODE_CONTEXT_WINDOW`, `RIGCODE_TOOL_MODE`) < command-line flags. `rigcode config` shows the result, and `rigcode config set provider.reasoning_effort high [--user|--project|--local]` changes it.

```json
{
  "provider": {
    "base_url": "auto", "model": "auto", "api_key": "none",
    "reasoning_effort": "low", "max_output_tokens": 32768, "tool_mode": "auto",
    "llama_cpp": { "slot_pinning": false }
  },
  "profiles": { "deep": { "reasoning_effort": "max" }, "gpu-box": { "base_url": "http://gpu-box:8000/v1", "model": "qwen3-coder" } },
  "agents": { "context": "auto", "max_concurrency": 1, "max_turns": 40 },
  "context": { "compact_at_tokens": 100000, "max_tool_output_chars": 12000 },
  "ui": { "thinking": "collapsed", "notify": "bell" }
}
```

Switch profiles with `rigcode -m deep` or `/model gpu-box`. `tool_mode: auto` uses native tool calls and falls back to text-based `<tool_call>` blocks if the server rejects tools.

`agents.context` controls how subagents start. `auto` (the default) forks them from the main conversation while it is below `agents.fork_max_tokens` (by default the compaction threshold or half the context window, whichever is smaller) and starts them fresh beyond that. `fork` always forks and `fresh` never does. Parallel subagents (`max_concurrency` above 1) need `fresh`, because parallel forks would each re-read the conversation.

## Extending

- **Memory:** `AGENTS.md`, `CLAUDE.md` or `RIGCODE.md`, found by walking up from the working directory, plus `~/.rigcode/AGENTS.md`. `@path` imports work. `/init` writes an `AGENTS.md` for you.
- **Subagents:** seven are built in: `explore`, `plan`, `coder`, `reviewer`, `debugger`, `researcher` and `verifier`. To add your own, put a Markdown file with `name`, `description`, `tools` and optionally `effort`, `max_turns` or `model` (a profile name) frontmatter in `.rigcode/agents/` or `.claude/agents/`.
- **Skills:** `.rigcode/skills/<name>/SKILL.md` (or `.claude/skills/`). The model sees each skill's description and loads the full instructions when it needs them.
- **Hooks:** `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `SessionStart`, `SessionEnd` and `PreCompact`. Hooks get JSON on stdin; exit code 2 blocks the action, and stderr is fed back to the model.
- **MCP servers:** add them under `"mcp_servers"` in settings or in a `.mcp.json`. Both stdio and streamable HTTP work, and tools appear as `mcp__<server>__<tool>`.

## Troubleshooting

- **`rigcode doctor`** checks the configuration, the shell, TCP reachability, the server type, the context size and slots, and makes a tiny chat request. If the TCP check times out, a firewall is blocking the port; doctor prints the `ufw` command for the server.
- **The model forgets things or ignores instructions on Ollama:** the context window is probably too small. Ollama's is set with `OLLAMA_CONTEXT_LENGTH`; rigcode needs at least 16k tokens, and 32k or more is better. rigcode warns when it detects a small window.
- **`rigcode --debug`** logs a summary of every request and response, including prompt and cached token counts, to `~/.rigcode/logs/<session>.jsonl`.
- **The first session after a change is slower.** Changing anything in the prompt prefix means it gets re-read once: upgrading rigcode, adding an MCP server, agent or skill, editing a memory file, changing effort, or adding a top-level directory. Later sessions in the same project reuse it.
- Sessions are stored as append-only JSONL in `~/.rigcode/projects/<project>/` and survive crashes. `rigcode sessions` lists them.

## Contributing

Bug reports with `rigcode doctor` output are the most useful thing you can send. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup.

## License

[MIT](LICENSE)
