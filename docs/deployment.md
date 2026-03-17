# Synapse Deployment Guide

[Back to README](../README.md)

## Deployment model

Synapse is designed to run locally first.

You can use it in three ways:

1. as a CLI-only tool
2. as a running MCP server
3. as a user-level background service

## Server modes

Start startup checks only:

```bash
python -m synapse serve
```

Start the actual server:

```bash
python -m synapse serve --run-server
```

The CLI no longer exposes transport selection. Synapse starts its single server runtime through the same entrypoint each time.

## macOS daemonization

Synapse supports user-level `launchd` integration.

Install:

```bash
python -m synapse install --service
```

Generated manifest path:

- `~/Library/LaunchAgents/com.synapse.memory.plist`

The generated service uses:

- `KeepAlive`
- `RunAtLoad`
- the current Python interpreter
- `-m synapse serve --run-server`
- `SYNAPSE_CONFIG_PATH`
- service log files under `.synapse/.logs/`

## Linux daemonization

Synapse supports `systemd --user` integration.

Install:

```bash
python -m synapse install --service
```

Generated manifest path:

- `~/.config/systemd/user/synapse-memory.service`

The unit uses:

- `Restart=on-failure`
- `RestartSec=5`
- the current Python interpreter
- `-m synapse serve --run-server`
- `SYNAPSE_CONFIG_PATH`

For headless persistence across logout, Linux systems may also need:

```bash
loginctl enable-linger $USER
```

## Service management commands

```bash
python -m synapse install --service
python -m synapse uninstall --service
python -m synapse restart
python -m synapse logs --lines 50
python -m synapse status
```

## Service logs

Synapse routes service output to:

- `.synapse/.logs/service.log`
- `.synapse/.logs/service-error.log`

## Status output

`python -m synapse status` reports both:

- memory/index health
- daemon install/runtime information

## Security note

If you expose the server beyond localhost, set `auth_token` in `config.toml`.

## Practical deployment advice

- Start `llama-server` for embedding and reranking before starting Synapse (or configure launchd/systemd to manage those processes).
- Keep `config.toml` under your project root or set `SYNAPSE_CONFIG_PATH` explicitly.
- Treat SQLite as derived state and Markdown as canonical state.
- Use service installation only after `python -m synapse serve --run-server` works manually.
