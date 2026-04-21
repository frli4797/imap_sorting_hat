# AGENTS.md

This is the main repo guidance for Codex and similar coding agents.

## Project Summary

`ish` is an IMAP mail sorter that:

- reads configuration from `ISH_CONFIG_PATH` or `~/.ish/settings.yaml`
- fetches message content from IMAP folders
- caches message text and embeddings in `cache.sqlite`
- trains a `RandomForestClassifier` on destination folders
- classifies unread messages from source folders and moves them to predicted folders

## High-Signal Rules

- Treat IMAP actions and OpenAI embedding calls as live side effects. Do not run them unless the user explicitly asks for live verification.
- Prefer unit tests and mocks over commands that touch a real mailbox, real config, or real API keys.
- If you need to demonstrate runtime usage, prefer `--dry-run`.
- Do not edit files under the user's external config directory such as `~/.ish` unless the user asks for it.
- The worktree may already contain user changes. Never revert unrelated edits.

## Working Commands

- Python target: `3.13`
- Install dependencies: `python -m pip install -r requirements.txt`
- Install package: `python -m pip install -e .`
- Run all tests: `pytest tests`
- Run focused tests:
  - `pytest tests/test_ish.py`
  - `pytest tests/test_imap.py`
  - `pytest tests/test_settings.py tests/test_metrics.py`
- Current code entry point: `python -m ish.app`

Notes:

- GitHub Actions currently runs `pytest tests` on Python `3.13`.
- The README still contains some legacy `python3 ish.py` wording. Prefer the module entry point from the code.
- There is no dedicated formatter config in the repo today. Match the existing style and keep diffs tight.

## Git and PR Conventions

- Prefer branch names that make the change type obvious.
- Branch format: `<type>/<short-kebab-summary>`
- Recommended branch types:
  - `feature/`
  - `fix/`
  - `refactor/`
  - `docs/`
  - `test/`
  - `chore/`
- Examples:
  - `feature/rest-api-skeleton`
  - `fix/cache-hash-collision-handling`
  - `refactor/classifier-service-extraction`
- Keep unrelated fixes and refactors out of a feature branch unless they are required for the change.
- PRs should also make the change type explicit in the title.
- Preferred PR title format: `<type>: <concise summary>`
- Recommended PR title types:
  - `feature:`
  - `fix:`
  - `refactor:`
  - `docs:`
  - `test:`
  - `chore:`
- If a change contains both a feature and incidental refactor, title it by the primary outcome and mention the refactor in the body.
- When practical, apply matching GitHub labels such as `feature`, `fix`, `refactor`, `docs`, `test`, or `chore`.
- Keep PRs scoped so a reviewer can tell whether it is primarily a feature, bug fix, refactor, or documentation change.

## Code Map

- `src/ish/app.py`: main orchestration, CLI args, training loop, classification loop, move threshold handling
- `src/ish/imap.py`: IMAP connection management, folder search, message fetch, move/copy behavior, message parsing
- `src/ish/db.py`: SQLite cache for message content, folder/uid mapping, embeddings, and move history
- `src/ish/settings.py`: YAML-backed configuration and data directory setup
- `src/ish/metrics.py`: logging configuration and optional Prometheus metrics
- `src/ish/message.py`: immutable message model and content hash
- `tests/`: unit-style coverage built mostly around mocks

For longer-lived context, see:

- `docs/architecture.md`
- `docs/codex-playbook.md`

## Change Guidance

- When practical, pair code changes with unit tests or non-regression tests that cover the new behavior or the bug being fixed.
- If a code change is not paired with tests, explain why the test was skipped or why it was not practical.
- For mail movement behavior, think through `ISH.move_messages()` and `ImapHandler.move()` together.
- For fetch and embedding work, preserve the cache-first flow in `ISH.get_msgs()` and `ISH.get_embeddings()`.
- For persistence changes, account for both `cache.sqlite` and the legacy shelve migration path.
- If you touch threshold defaults, do it intentionally in both places. `ISH.__init__()` and `RuntimeOptions.from_args()` currently use different defaults.
- If you change behavior that a user will notice, update the related docs in the same change.
- New business logic should stay reusable outside the CLI so it can later be exposed via API and UI.
- Keep stateful operations callable from future service layers instead of coupling them tightly to terminal flows.
- Design with per-user isolation in mind, even if the current implementation is still effectively single-user.

## Safe Validation

- Docs-only changes: a manual review is enough.
- Unit-level code changes: run the narrowest relevant `pytest` target first.
- Broader code changes: run `pytest tests`.
- Avoid end-to-end runs unless the user explicitly wants live verification and has provided safe config.

## Testing and Mocking Expectations

- Default expectation: substantive code changes should come with tests when practical.
- Mock `ImapHandler` instead of connecting to a real IMAP server.
- Mock the OpenAI client instead of making embedding calls.
- Keep new tests in `tests/` close to the module they exercise.
- Prefer regression tests around behavior boundaries: cache hits, cache misses, move thresholds, migration behavior, and dry-run safety.
- Treat tests as both functional verification and quality assurance, not just a box-checking exercise.

## Risk Areas

- `Message.hash()` is content-based and does not include UID. Cache and move history behavior depend on that design.
- `SQLiteCache.store_message()` rewrites folder associations for a hash before inserting the current mapping.
- `ISH.run()` can trigger both training and classification in one session.
- `move_messages()` is the last step before side effects on a mailbox. Changes there need careful tests.

## Future Direction

These points are intentional product direction, not claims about current implementation.

- Planned interface direction: `ish` is expected to grow a REST API around core operations.
- Planned deployment direction: `ish` is expected to evolve toward a multi-user, single-tenant application model.
- Planned UX direction: `ish` is expected to gain a user interface for configuration, retraining, re-indexing, and operational visibility.

Implications for current changes:

- Prefer separating core application logic from CLI-only concerns.
- Avoid embedding terminal interaction into business logic that will later need to be called from API or UI layers.
- Prefer service boundaries and data models that can be reused across CLI, REST, and UI entry points.
- When changing persistence or user-scoped state, avoid assumptions that make future per-user isolation harder.

## Architecture Feedback

- Agents should actively reflect on architecture and design while working, not only on local code correctness.
- When a change exposes architectural friction, unclear boundaries, duplication, coupling, or roadmap tension, call that out in the final response.
- Keep that feedback concise and actionable: identify the issue, why it matters, and what direction it suggests.
- Use those observations to help steer future architectural and functional decisions, even when the immediate task stays narrowly scoped.
