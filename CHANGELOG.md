# Changelog

## 0.1.0 - unreleased

First public release.

- Terminal coding agent for OpenAI-compatible servers: llama.cpp, Ollama, LM Studio, vLLM and SGLang.
- Finds a local server and its loaded model automatically when none is configured, and warns when Ollama's or LM Studio's context window is too small for an agent.
- Prompt-cache-aware design for slow hardware: fixed prompt prefix, append-only history, forked subagents, background warm-up.
- Tools: Read, Edit, MultiEdit, Write, Grep, Glob (files and directories), Bash with background shells, WebFetch, WebSearch, TodoWrite, Task, Skill, AskUser.
- Permission modes, allow/deny/ask rules, hooks, MCP servers, skills, custom agents and commands. Reads `AGENTS.md`, `CLAUDE.md` and `.claude/` setups.
- Edits are allowed after Grep has shown the exact lines; overwriting a file with Write still requires a full Read.
