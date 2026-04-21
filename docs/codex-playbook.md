# Codex Playbook

Use these prompt patterns when you want faster, more reliable Codex sessions on this repo.

## Good Default Framing

When asking for work, include:

- the target files or area
- whether behavior may change
- whether tests should be added or updated
- whether live IMAP/OpenAI calls are allowed
- whether you want architectural or design feedback included with the implementation
- the intended branch or PR type if you want GitHub naming to follow a repo convention

## Prompt Recipes

### Explore before editing

```text
Inspect the classifier and cache flow in this repo before changing anything.
Summarize the relevant files, identify risks, and do not edit yet.
```

### Fix a bug with tests first

```text
Reproduce and fix the bug in `src/ish/app.py` where ...
Add or update the smallest relevant pytest coverage first, then implement the fix.
Avoid live IMAP or OpenAI calls.
```

### Safe refactor

```text
Refactor `src/ish/imap.py` for readability without changing behavior.
Keep the diff tight, preserve current tests, and run the relevant pytest file after the change.
```

### Review a risky area

```text
Review the move and cache logic for regressions.
Focus on message movement, folder-to-hash mapping, migration behavior, and threshold handling.
List findings first with file references. Do not edit code yet.
```

### Add a feature without touching live services

```text
Add support for ...
Keep all validation at the unit-test level with mocks.
Do not run commands that talk to a real IMAP server or OpenAI.
```

### Ask for implementation plus architectural feedback

```text
Make the requested change, pair it with practical unit or regression tests, and
also call out any architectural or design observations that the change reveals.
Keep the feedback short and action-oriented.
```

### Docs sync

```text
Update the repo docs to match the current code paths and CLI behavior.
If the README is outdated, fix it in the same change.
```

### Ask for a clean GitHub-ready change

```text
Make this change as a clearly scoped `<type>` change, use the repo branch naming
convention, and propose a PR title that cleanly communicates whether this is a
feature, fix, refactor, docs, test, or chore change.
```

## Repo-Specific Reminders

- Ask Codex to use `python -m ish.app` if you want runtime command examples.
- Ask for `--dry-run` unless you explicitly want live mailbox operations.
- Ask for focused tests first, then the full suite if the change grows.
- Mention `AGENTS.md` if you want Codex to follow the repo conventions strictly.
- Ask explicitly for architecture feedback if you want Codex to surface design pressure, not just deliver the patch.
- Ask explicitly for a branch and PR type if you want Codex to keep GitHub hygiene tight.

## Example High-Quality Request

```text
Inspect `src/ish/app.py` and `src/ish/db.py`, explain how cached embeddings are reused,
then patch the bug causing stale folder mappings. Add the smallest regression test that
covers the change and run only the relevant pytest target. Also note any architectural
issues the change reveals. Do not use live IMAP or OpenAI.
```
