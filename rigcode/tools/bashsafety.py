"""Shell command analysis: read-only classification, danger detection, rule
suggestions. Conservative by design: anything unrecognized needs approval."""
from __future__ import annotations

import fnmatch
import re
import shlex

READ_ONLY = {
    "ls", "dir", "pwd", "cat", "head", "tail", "wc", "echo", "printf", "grep", "egrep", "fgrep", "rg", "ag",
    "tree", "which", "where", "whereis", "type", "file", "stat", "du", "df", "uname", "whoami", "id", "date",
    "hostname", "ps", "sort", "uniq", "cut", "basename", "dirname", "realpath", "readlink", "nl", "diff",
    "cmp", "comm", "column", "jq", "true", "false", "test", "[", "env", "printenv", "md5sum", "sha1sum",
    "sha256sum", "cksum", "od", "hexdump", "strings", "free", "uptime", "nproc", "arch", "lscpu", "seq",
    "findstr", "ver", "tasklist", "systeminfo", "wsl.exe-version",
    # PowerShell read-only cmdlets (and aliases)
    "get-childitem", "gci", "get-content", "gc", "select-string", "sls", "get-location", "gl", "get-item",
    "gi", "test-path", "get-command", "gcm", "resolve-path", "measure-object", "measure", "get-process",
    "gps", "format-table", "ft", "format-list", "fl", "select-object", "select", "where-object", "sort-object", "out-string", "get-date", "get-itemproperty", "get-filehash", "get-help", "write-output",
    "write-host",
}

GIT_READ_ONLY = {
    "status", "diff", "log", "show", "rev-parse", "ls-files", "ls-tree", "blame", "describe", "shortlog",
    "grep", "cat-file", "show-ref", "name-rev", "merge-base", "whatchanged", "count-objects", "for-each-ref",
    "var", "help", "version", "--version", "fetch --dry-run",
}

VERSION_ONLY = {"python", "python3", "py", "node", "ruby", "go", "cargo", "rustc", "java", "javac", "gcc",
                "clang", "dotnet", "deno", "bun", "npm", "pnpm", "yarn", "pip", "pip3", "uv", "conda", "make",
                "cmake", "perl", "php", "tsc", "kubectl", "docker", "terraform", "gh", "git"}

PKG_READ_ONLY = {
    "npm": {"ls", "list", "view", "info", "outdated", "why", "explain", "config"},
    "pnpm": {"ls", "list", "why", "outdated"},
    "yarn": {"list", "info", "why", "outdated"},
    "pip": {"list", "show", "freeze", "check"},
    "pip3": {"list", "show", "freeze", "check"},
    "conda": {"list", "info"},
    "docker": {"ps", "images", "logs", "inspect", "version", "info"},
    "kubectl": {"get", "describe", "logs", "version"},
    "go": {"version", "env", "list", "doc", "vet"},
    "cargo": {"tree", "metadata"},
    "gh": {"status"},
}

MULTI_WORD = {"npm", "pnpm", "yarn", "bun", "git", "cargo", "go", "uv", "pip", "pip3", "docker", "kubectl",
              "make", "poetry", "dotnet", "gh", "conda", "pytest", "mvn", "gradle", "./gradlew", "terraform"}

DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\*|\.\.?)(\s|$)", "recursive delete of a broad path"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r", "rm -rf"),
    (r"\b(rmdir|rd)\s+/s\b|\bdel\s+/[sfq]", "recursive delete (cmd)"),
    (r"remove-item\b.*-recurse", "recursive delete (PowerShell)"),
    (r"\bgit\s+push\b.*(--force\b|-f\b|--force-with-lease)", "force push"),
    (r"\bgit\s+reset\s+--hard", "git reset --hard (discards changes)"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*[fdx]", "git clean (deletes untracked files)"),
    (r"\bgit\s+(checkout|restore)\s+(--\s+)?\.(\s|$)", "discards working-tree changes"),
    (r"\bgit\s+branch\s+-D\b", "force-deletes a branch"),
    (r"\bsudo\b|\bsu\s+-", "elevated privileges"),
    (r"\bmkfs\b|\bdd\s+if=|>\s*/dev/sd|\bformat\s+[a-z]:", "disk-level operation"),
    (r"\b(shutdown|reboot|halt|poweroff)\b|stop-computer|restart-computer", "shuts down or reboots"),
    (r"(curl|wget|iwr|invoke-webrequest)\b[^|]*\|\s*(sh|bash|zsh|python|iex|invoke-expression)", "pipes a download into a shell"),
    (r"\bchmod\s+(-R\s+)?777\b", "world-writable permissions"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\b(npm|pnpm|yarn|cargo|twine|uv)\s+publish\b", "publishes a package"),
    (r"\bdocker\s+(system|volume|image)\s+prune\b", "docker prune"),
    (r"\bdrop\s+(table|database)\b|\btruncate\s+table\b", "destructive SQL"),
]
_CASE_SENSITIVE = {"force-deletes a branch"}     # git branch -D (unlike -d) is the dangerous one
_DANGER = [(re.compile(p, 0 if reason in _CASE_SENSITIVE else re.I), reason) for p, reason in DANGEROUS_PATTERNS]


def split_segments(command: str) -> list[str] | None:
    """Split on unquoted && || ; | & and newlines. Returns None if the command
    uses constructs we refuse to auto-approve (substitution, output redirection)."""
    segs: list[str] = []
    cur: list[str] = []
    i, n = 0, len(command)
    quote = None
    while i < n:
        c = command[i]
        if quote == "'":
            cur.append(c)
            if c == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if c == "\\" and i + 1 < n:
                cur.append(command[i:i + 2])
                i += 2
                continue
            if c == "`" or command.startswith("$(", i):
                return None
            cur.append(c)
            if c == '"':
                quote = None
            i += 1
            continue
        if c in "'\"":
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(command[i:i + 2])
            i += 2
            continue
        if c == "`" or command.startswith("$(", i) or command.startswith("<(", i) or command.startswith(">(", i):
            return None
        if c == ">":
            rest = command[i:]
            ok = re.match(r">&\s*[12]\b|>\s*/dev/null\b|>\s*\$null\b|>\s*nul\b", rest, re.I)
            prev = "".join(cur).rstrip()
            if ok or (prev.endswith("2") and re.match(r">\s*/dev/null|>&1", rest)):
                m = ok or re.match(r">\s*/dev/null|>&1", rest)
                cur.append(m.group(0))
                i += len(m.group(0))
                continue
            return None
        if command.startswith("&&", i) or command.startswith("||", i):
            segs.append("".join(cur))
            cur = []
            i += 2
            continue
        if c in ";|&\n":
            segs.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if quote:
        return None
    segs.append("".join(cur))
    return [s.strip() for s in segs if s.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        toks = shlex.split(segment, posix=True)
    except ValueError:
        toks = segment.split()
    while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
        toks.pop(0)
    return toks


def _base(word: str) -> str:
    w = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in (".exe", ".cmd", ".bat"):
        if w.endswith(ext):
            w = w[: -len(ext)]
    return w


def segment_is_read_only(segment: str) -> bool:
    toks = _tokens(segment)
    if not toks:
        return True
    cmd = _base(toks[0])
    args = toks[1:]
    low_args = [a.lower() for a in args]
    if cmd in ("cd", "set-location", "sl", "pushd", "popd"):
        return True
    if cmd == "find":
        return not any(a in ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls") for a in args)
    if cmd == "sed":
        return not any(a.startswith("-i") or a == "--in-place" for a in args)
    if cmd == "git":
        while args and args[0].startswith("-"):
            opt = args.pop(0)
            if opt in ("-C", "-c") and args:
                args.pop(0)
        if not args:
            return True
        sub = args[0]
        rest = args[1:]
        if sub in GIT_READ_ONLY:
            return True
        if sub == "branch":
            return not any(a in ("-d", "-D", "-m", "-M", "--delete", "--move", "-c", "-C", "--copy", "-f", "--force",
                                 "--set-upstream-to", "-u", "--unset-upstream", "--edit-description") for a in rest) and \
                all(a.startswith("-") for a in rest)
        if sub == "tag":
            return not rest or all(a in ("-l", "--list", "-n") or a.startswith("--sort") or a.startswith("--contains") for a in rest)
        if sub == "remote":
            return not rest or rest[0] in ("-v", "show", "get-url")
        if sub == "stash":
            return bool(rest) and rest[0] in ("list", "show")
        if sub == "config":
            return any(a in ("--get", "--list", "-l", "--get-all", "--get-regexp") for a in rest)
        if sub == "reflog":
            return not rest or rest[0] == "show"
        return False
    if cmd in READ_ONLY:
        return True
    if cmd in VERSION_ONLY and low_args and low_args[0] in ("--version", "-v", "-V", "version"):
        return True
    if cmd in PKG_READ_ONLY and args and args[0] in PKG_READ_ONLY[cmd]:
        if cmd == "npm" and args[0] == "config":
            return len(args) > 1 and args[1] in ("get", "list", "ls")
        return True
    if cmd == "uv" and args[:2] in (["pip", "list"], ["pip", "show"], ["pip", "freeze"]) or (cmd == "uv" and args[:1] == ["tree"]):
        return True
    return False


def is_read_only_command(command: str) -> bool:
    segs = split_segments(command)
    if segs is None:
        return False
    return all(segment_is_read_only(s) for s in segs)


def danger_reason(command: str) -> str | None:
    for rx, reason in _DANGER:
        if rx.search(command):
            return reason
    return None


def command_prefix(segment: str) -> str:
    toks = _tokens(segment)
    if not toks:
        return segment.strip()
    word = toks[0]
    base = _base(word)
    if base in ("python", "python3", "py") and len(toks) > 2 and toks[1] == "-m":
        return " ".join(toks[:3])
    if base in ("npm", "pnpm", "yarn", "bun", "uv", "poetry") and len(toks) > 2 and toks[1] == "run":
        return " ".join(toks[:3])
    if base in MULTI_WORD and len(toks) > 1 and not toks[1].startswith("-"):
        return " ".join(toks[:2])
    return word


def suggest_rule(command: str) -> str:
    segs = split_segments(command) or [command]
    return f"Bash({command_prefix(segs[0])}:*)"


def rule_matches_segment(rule_body: str, segment: str) -> bool:
    """rule_body: 'npm test:*' (prefix), 'git *' (glob) or an exact command."""
    seg = " ".join(segment.split())
    body = " ".join(rule_body.split())
    if body.endswith(":*"):
        prefix = body[:-2].rstrip()
        return seg == prefix or seg.startswith(prefix + " ")
    if "*" in body or "?" in body:
        return fnmatch.fnmatchcase(seg, body)
    return seg == body


def command_matches_rule(rule_body: str, command: str) -> bool:
    """Every segment of a compound command must match (or be read-only)."""
    segs = split_segments(command)
    if segs is None:
        return rule_matches_segment(rule_body, command)
    return all(rule_matches_segment(rule_body, s) or segment_is_read_only(s) for s in segs) and \
        any(rule_matches_segment(rule_body, s) for s in segs)
