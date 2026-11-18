# Benchmarks

These are small, honest comparisons, not a leaderboard entry. The question they answer: with the same model on the same hardware, does rigcode do as well as a mature commercial agent?

## SWE-bench Verified, six tasks (October 2026)

**Setup**

- **Model:** Kimi K3 (2.78T parameters) on llama.cpp, on one machine on the local network, with 4 slots and a 262,144-token context. Prompt processing runs at about 10–35 tokens/s and generation at about 5 tokens/s.
- **Agents:** rigcode 0.1 (then called kcode) and Claude Code 2.1, both pointed at the same server. Claude Code reached it through a small local proxy that set the reasoning effort.
- **Effort:** maximum reasoning effort for both, with a 1M-token compaction threshold so neither compacted.
- **Tasks:** six SWE-bench Verified issues: four from Django and two from SymPy.
- **Grading:** the maintainers' hidden tests. A task counts as resolved only if every FAIL_TO_PASS test passes and no PASS_TO_PASS test breaks.
- **Isolation:** agents were told not to use the network. After one run downloaded a released package (see below), the network was also blocked with a dead proxy, and pip was limited to local wheels.
- **Audit:** every transcript was audited for network use and for reading files outside the agent's own checkout.

**Results**

| Issue | rigcode | Claude Code |
|---|---|---|
| django__django-15957 | not resolved (0/4), 480 min, hit the time cap | resolved, 427 min (rerun, see below) |
| sympy__sympy-24066 | resolved, 132 min | resolved, 169 min |
| django__django-16631 | not resolved (0/1), 120 min | not resolved (0/1), 188 min |
| django__django-15503 | resolved, 199 min | resolved, 337 min |
| sympy__sympy-24661 | resolved, 139 min | resolved, 126 min |
| django__django-15128 | resolved, 433 min | *excluded*: resolved after reading the fix (see below) |
| **Clean resolves** | **4 / 6** | **4 / 5** |

For reference, Claude Code with Claude Opus 5.5 on Anthropic's servers resolved all six, in about 20 minutes in total.

**Integrity problems found and how they were handled**

- **Claude Code's first run on django-15957 ran `pip download django==4.2`** and used the released code, which contains the fix. That run was discarded. Both agents then reran the task side by side with the network blocked, and those reruns are the results above. rigcode failed the task in both its runs.
- **Claude Code's run on django-15128 read other tasks' source snapshots**, including Django 4.2, which already contains the official fix for this issue. It printed the fixed `Query.combine()` and `Query.bump_prefix()` before writing its patch, so the result is excluded. rigcode's run on the same task did not read any snapshot.
- **The Django test-log parser misread failing subtests.** On Python 3.10, a test whose subtests fail prints no status on its own line. All results were regraded using the `FAIL:`/`ERROR:` summary blocks.

**Caveats**

- Six tasks is a small sample: one task flips the comparison. Each task ran once per agent, and agent runs are not deterministic.
- The two agents often ran at the same time and shared the server's four slots, so the wall times are only roughly comparable.
- rigcode was designed against this exact model and server, while Claude Code was designed for Anthropic's models.

## Five small tasks at low effort (October 2026)

The tasks were a bug fix, a feature, a crash fix, a question about the code, and a rename, each graded by hidden tests. Both agents ran on Kimi K3 at low effort, and Claude Code's prompt was pre-warmed.

| | rigcode | Claude Code |
|---|---|---|
| Tasks passed | 5 / 5 | 5 / 5 |
| Total time | **41.4 min** | 59.1 min |
| Model calls | 25 | 31 |
| Context per call | ~5k tokens | ~18k tokens |

rigcode generated 34% fewer tokens. The tasks were too easy to separate the agents on correctness. They do show what the smaller prompt and cache reuse save on slow hardware.

## Ollama smoke test (November 2026)

rigcode with no configuration found Ollama 0.34 on its default port, picked the loaded model (gpt-oss:120b), fixed a one-line bug and ran the test to confirm. It took 9.5 s and 5 model calls, and 97% of prompt tokens came from Ollama's cache.
