# TODO-11: Deployment & OS Daemonization

## Status: COMPLETED
## Priority: P2 (Quality of life — not blocking core functionality)
## Design Doc Section: §8

---

## Summary

实现 Synapse 的 OS 级守护进程集成——macOS（`launchd`）和 Linux（`systemd`）的自动启动服务配置生成与安装。让 Synapse 作为 "always-on" 的背景记忆代理运行，自动随系统启动。

---

## Detailed Requirements

### 1. macOS Daemon: `launchd` (§8.1)

#### 1.1 Plist 生成

`synapse install --service` 自动生成 plist 文件：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.synapse.memory</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>-m</string>
        <string>synapse</string>
        <string>serve</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/synapse/project</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/.synapse/logs/service.log</string>
    <key>StandardErrorPath</key>
    <string>~/.synapse/logs/service-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>SYNAPSE_CONFIG_PATH</key>
        <string>/path/to/config.toml</string>
    </dict>
</dict>
</plist>
```

#### 1.2 Installation Flow

```bash
$ synapse install --service
# 1. Detect OS (macOS)
# 2. Generate plist → ~/Library/LaunchAgents/com.synapse.memory.plist
# 3. Load service: launchctl load ~/Library/LaunchAgents/com.synapse.memory.plist
# 4. Start service: launchctl start com.synapse.memory
# 5. Print: "✓ Synapse daemon installed and running"
```

#### 1.3 Service Features
- **KeepAlive**: Crash 后自动重启
- **RunAtLoad**: 用户登录后自动启动
- **日志路由**: stdout/stderr → `~/.synapse/logs/service.log`

### 2. Linux Daemon: `systemd` (§8.2)

#### 2.1 Service File 生成

```ini
[Unit]
Description=Synapse Memory Agent
After=network.target

[Service]
Type=simple
ExecStart=/path/to/python -m synapse serve
WorkingDirectory=/path/to/synapse/project
Environment=SYNAPSE_CONFIG_PATH=/path/to/config.toml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

#### 2.2 Installation Flow

```bash
$ synapse install --service
# 1. Detect OS (Linux)
# 2. Generate service file → ~/.config/systemd/user/synapse-memory.service
# 3. Reload: systemctl --user daemon-reload
# 4. Enable + Start: systemctl --user enable --now synapse-memory.service
# 5. Print: "✓ Synapse daemon installed and running"
```

#### 2.3 Headless Support
- 提示用户运行 `loginctl enable-linger $USER` 以支持无登录启动
- 适用于 NAS、云 VM 等无头环境

### 3. CLI Commands

```bash
synapse install --service    # 安装 OS daemon
synapse uninstall --service  # 卸载 OS daemon
synapse status               # 查看 daemon 运行状态
synapse logs                 # tail 日志文件
synapse restart              # 重启 daemon
```

### 4. Service Management Utilities

```python
class ServiceManager:
    def detect_os(self) -> str:
        """Detect macOS vs Linux."""
    
    def install(self) -> None:
        """Generate and install service file for current OS."""
    
    def uninstall(self) -> None:
        """Remove service file and stop daemon."""
    
    def status(self) -> ServiceStatus:
        """Check if daemon is running."""
    
    def restart(self) -> None:
        """Restart the daemon."""
    
    def get_python_path(self) -> str:
        """Detect the correct Python interpreter path (venv-aware)."""
```

### 5. Path Resolution

- 自动检测当前 Python 解释器路径（支持 virtualenv）
- 自动检测项目工作目录
- 自动检测 config.toml 路径

### 6. Idempotent Installation

- 重复运行 `synapse install --service` 不报错（覆盖已有配置）
- 版本升级后重新安装以更新路径

---

## Dependencies
- **TODO-01**: CLI framework, config system
- **TODO-07**: MCP server (`synapse serve` command)

## Blocks
- None (independent quality-of-life feature)

## Acceptance Criteria
- [x] macOS: plist 文件正确生成到 `~/Library/LaunchAgents/`
- [x] macOS: `launchctl` best-effort install/status/restart wrappers integrated with graceful degradation
- [x] Linux: service 文件正确生成到 `~/.config/systemd/user/`
- [x] Linux: `systemctl --user` best-effort enable/start/status wrappers integrated with graceful degradation
- [x] `synapse status` 正确报告 daemon 状态
- [x] `synapse uninstall --service` 清理干净
- [x] KeepAlive / Restart policies are encoded in generated launchd/systemd manifests
- [x] 日志正确路由到 `.synapse/.logs/service.log` 与 `.synapse/.logs/service-error.log`
- [x] Python 路径检测正确（包括 venv）
- [x] 重复安装幂等

---

## Implementation Notes

- Added `synapse.deployment.service_manager.ServiceManager` as the cross-platform abstraction for:
    - OS detection (`macOS` vs `Linux`)
    - venv-aware interpreter resolution via `sys.executable`
    - working directory and config path resolution
    - launchd plist generation
    - systemd user unit generation
    - best-effort install / uninstall / restart / status / log helpers
- Extended the CLI with:
    - `synapse install --service`
    - `synapse uninstall --service`
    - `synapse restart`
    - `synapse logs`
    - daemon-aware `synapse status`
- Added focused unit tests for plist rendering, systemd unit rendering, idempotent install/uninstall behavior, and CLI output using monkeypatched/fake service managers.
