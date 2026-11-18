# Contributing

Thanks for helping. Bug reports, fixes and reports of which models and servers work well are all welcome.

## Reporting a bug

Open an issue with the output of `rigcode doctor`, the command you ran, and what you expected. If the problem involves the model server, run with `--debug` and attach the relevant lines from `~/.rigcode/logs/<session>.jsonl`, with anything private removed.

## Development setup

```bash
git clone https://github.com/sarthak-1111/rigcode
cd rigcode
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
pytest -q                 # unit tests plus end-to-end runs against a scripted mock server
ruff check rigcode tests
```

The tests don't need a real model server. `tests/mock_server.py` plays back scripted streaming replies, including tool calls.

## Code layout

| Path | What's there |
|---|---|
| `rigcode/llm` | HTTP client, streaming, server detection, text-mode tool calls |
| `rigcode/agent` | the agent loop, runtime, prompts, sessions, file and todo state |
| `rigcode/tools` | Read, Edit, Write, Grep, Glob, Bash, web, Task, and the other tools |
| `rigcode/ui` | the full-screen terminal UI, the plain/headless UI, slash commands |
| `rigcode/prompts` | the system prompt and built-in subagents |

## Ground rules

- **Keep the prompt prefix stable.** Anything that changes the system message or tool schemas mid-session forces the server to re-read the whole prompt, which can cost minutes on slow hardware. Put volatile information in reminders instead.
- **Keep changes focused.** A bug fix doesn't need surrounding refactors. Add a test that fails without your change.
- **Only original code.** Don't copy code or prompts from other agents unless their license allows it and you credit them.
