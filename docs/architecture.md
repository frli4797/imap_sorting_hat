# Architecture

This note gives AI agents and contributors a stable overview of how `ish` works today.

It also documents a small amount of intended future direction so implementation choices can lean the right way without pretending those features already exist.

## Main Runtime Flow

1. `Settings` loads configuration from `ISH_CONFIG_PATH` or `~/.ish/settings.yaml`.
2. `ISH` initializes logging, metrics, IMAP/OpenAI clients, and the SQLite cache in the configured data directory.
3. `ISH.run()` connects to IMAP and OpenAI, then ensures a classifier exists.
4. Training uses destination folders as labels:
   - fetch folder UIDs
   - reuse cached message text and embeddings when possible
   - request missing embeddings from OpenAI in batches
   - train a `RandomForestClassifier`
   - write the model to `model.pkl`
5. Classification scans unread messages in source folders:
   - fetch unread UIDs
   - resolve embeddings from cache or OpenAI
   - predict destination folder and probability
   - move only when probability exceeds the configured threshold
6. Move history and cache state are persisted in SQLite.

## Module Responsibilities

- `src/ish/app.py`
  - owns application lifecycle
  - trains and loads the classifier
  - groups predicted moves by destination folder
  - records metrics and move history
- `src/ish/imap.py`
  - connects to IMAP
  - searches folders
  - fetches raw messages
  - converts message bodies to plain text
  - performs server-side move when supported, otherwise copy/delete fallback
- `src/ish/db.py`
  - stores canonical message content by hash
  - maps `(folder, uid)` to `msg_hash`
  - stores embeddings as pickled numpy arrays
  - stores move history with confidence scores
- `src/ish/settings.py`
  - loads YAML config
  - resolves config and data directories
- `src/ish/metrics.py`
  - configures logging
  - exposes optional Prometheus metrics

## Persistent Files

By default the app works under `~/.ish` unless `ISH_CONFIG_PATH` overrides it.

- `settings.yaml`: IMAP, folder, data-dir, and OpenAI configuration
- `data/cache.sqlite`: messages, folder mappings, embeddings, and move history
- `data/model.pkl`: trained classifier

## SQLite Model

The cache currently has four main tables:

- `messages_content`: canonical message fields keyed by `msg_hash`
- `folder_messages`: current `(folder, uid) -> msg_hash` mapping
- `embeddings`: pickled embedding vectors keyed by `msg_hash`
- `moves`: move history with confidence and timestamp

Important implementation detail:

- `Message.hash()` is based on message fields, not UID.
- `SQLiteCache.store_message()` deletes prior folder mappings for the same hash before inserting the current one.

## External Side Effects

Be careful around these boundaries:

- IMAP login, search, fetch, move, copy, flagging, and expunge
- OpenAI embedding requests
- filesystem writes to `cache.sqlite` and `model.pkl`
- optional metrics server startup through env vars

For local development and agent work, unit tests with mocks are preferred over live runs.

## Testing Shape

The current tests are mostly unit tests with mocked collaborators:

- `tests/test_ish.py`: orchestration, training/classification behavior, dry-run behavior
- `tests/test_imap.py`: parsing, reconnection paths, and move fallback logic
- `tests/test_settings.py`: config and data-directory behavior
- `tests/test_metrics.py`: logging and metrics helpers

CI runs `pytest tests` on Python `3.13`.

## Current Quirks Worth Preserving Intentionally

- The direct `ISH(...)` constructor and CLI runtime options currently have different default move thresholds.
- Legacy shelve data is migrated automatically only when `cache.sqlite` is missing and old cache files exist.
- Classification only acts on unread messages from configured source folders.
- Training requires at least two samples and at least two destination folders with usable data.

## Planned Direction

These are roadmap-level goals, not current architecture guarantees:

- add a REST API around core operations
- evolve toward a multi-user, single-tenant deployment model
- add a user interface for configuration, retraining, re-indexing, and operational visibility

Design pressure this creates now:

- keep domain logic reusable outside the CLI entry point
- avoid mixing terminal interaction with application services
- prefer interfaces that could later support API and UI callers
- avoid persistence assumptions that hard-code a forever single-user model
