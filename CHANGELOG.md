# Changelog

## 0.1.1 - 2026-11-20

- Finding a local server is faster on Windows, where a closed localhost port takes about 2 seconds to refuse a connection. Ports are now checked with a quick TCP connect first, so startup no longer waits on servers that aren't running.
- Permission and question dialogs no longer have a bright blue background. Their style names clashed with prompt_toolkit's built-in `dialog` style.
- Permission prompts read naturally: "Allow this edit?" and "Allow creating this file?" instead of "Allow this edit file?" and "Allow this create file?".
- `rigcode doctor` accepts `-m`, `--base-url` and `--api-key`, and rejects unknown flags instead of silently ignoring them.
- The README has a demo recording.

## 0.1.0 - 2026-11-19

First public release.

- Terminal coding agent for OpenAI-compatible servers: llama.cpp, Ollama, LM Studio, vLLM and SGLang.
- Finds a local server and its loaded model automatically when none is configured, and warns when Ollama's or LM Studio's context window is too small for an agent.
- Prompt-cache-aware design for slow hardware: fixed prompt prefix, append-only history, forked subagents, background warm-up.
- Tools: Read, Edit, MultiEdit, Write, Grep, Glob (files and directories), Bash with background shells, WebFetch, WebSearch, TodoWrite, Task, Skill, AskUser.
- Permission modes, allow/deny/ask rules, hooks, MCP servers, skills, custom agents and commands. Reads `AGENTS.md`, `CLAUDE.md` and `.claude/` setups.
- Edits are allowed after Grep has shown the exact lines; overwriting a file with Write still requires a full Read.
