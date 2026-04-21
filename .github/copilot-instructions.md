# Copilot Instructions

Use [../AGENTS.md](../AGENTS.md) as the canonical project guide.

For this repository:

- avoid live IMAP and OpenAI calls unless the user explicitly asks for them
- prefer unit tests with mocks over live mailbox runs
- use `pytest tests` for the full validation path
- use `python -m ish.app` as the current entry point, not the older `python3 ish.py` wording from the README
- be careful with changes to message movement, SQLite cache compatibility, and move-threshold defaults
- prefer explicit change-type naming for branches and PRs such as `feature/...`, `fix/...`, or `refactor/...`

Helpful context lives in:

- [../AGENTS.md](../AGENTS.md)
- [../docs/architecture.md](../docs/architecture.md)
- [../docs/codex-playbook.md](../docs/codex-playbook.md)
