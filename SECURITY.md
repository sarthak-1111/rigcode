# Security

rigcode runs shell commands and edits files on your machine when the model asks and you, or your permission rules, allow it. Treat its permission settings like any other access control.

## Reporting a vulnerability

Please report security problems privately through GitHub's **Report a vulnerability** button on the Security tab of this repository, not in a public issue. Examples: a way for model output or file contents to bypass a permission check, or a command that the read-only rules approve but that can change things.

You should get a reply within a week.

## Things to know

- `bypass` mode runs everything except deny rules without asking. Use it only in a sandbox or a throwaway checkout.
- Tool results and file contents are untrusted input to the model. A file in a repository you don't control can try to steer the agent (prompt injection). Permission prompts are the safeguard, so read them.
- rigcode sends no telemetry. It connects only to your model server, the MCP servers you configure, and websites the model fetches or searches when you allow it.
