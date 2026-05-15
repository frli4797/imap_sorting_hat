# Development Environment

This document collects setup notes that are mainly useful while developing `ish`.

## Docker Compose

Docker Compose can run the checkout directly with a mounted config directory. The default `ish` service command uses `--dry-run`.

Prepare a local config:

```shell
cp .ish-dev/settings.yaml.example .ish-dev/settings.yaml
```

Edit `.ish-dev/settings.yaml` with test credentials and folders. The real settings file and `.ish-dev/data` are ignored by git.

Build and run the dry-run app:

```shell
docker compose run --rm ish
```

Run tests in the container:

```shell
docker compose run --rm test
```

If your Docker install uses the legacy Compose binary, replace `docker compose` with `docker-compose`.

To test a specific runtime command, override the Compose command explicitly:

```shell
docker compose run --rm ish python -m ish.app --dry-run --learn-folders
```

Only remove `--dry-run` when you intentionally want to move messages in the configured mailbox.

## Local Tests

The repo target is Python 3.13. Run unit tests with an isolated config path so local runs do not touch `~/.ish`:

```shell
ISH_CONFIG_PATH=/tmp/ish-test-config PYTHONPATH=src pytest tests
```
