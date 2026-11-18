Create a git commit for the current changes.
1. Run `git status`, `git diff --staged`, `git diff` and `git log --oneline -8` (together, in one reply) to see the changes and this repository's commit message style.
2. If nothing is staged, stage the relevant files explicitly by path. Never use `git add -A` or `git add .` blindly; leave out unrelated files, build artifacts and anything that looks like a secret.
3. Write a commit message in the repository's style: a short summary line (imperative mood, at most 72 characters) and, only if useful, a brief body explaining why.
4. Commit, then show `git log --oneline -1`. Do not push.
