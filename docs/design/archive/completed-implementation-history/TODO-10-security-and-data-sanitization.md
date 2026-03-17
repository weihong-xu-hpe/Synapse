# TODO-10: Security & Data Sanitization Pipeline

## Status: COMPLETED
## Priority: P1 (Privacy protection — founding design goal)
## Design Doc Section: §7

---

## Summary

实现 Synapse 的安全与隐私保护层——包括数据净化管道（Regex Redaction、Sensitivity Classifier）、传输审计日志、以及本地文件保护。确保发送到**任何远端 provider**（Cloud LLM、远端 embedding API、远端 rerank API）的数据经过清洗，敏感信息永远不离开本地。调用本地 Ollama（loopback）默认不视为远端外发。

---

## Detailed Requirements

### 1. Three-Stage Sanitization Pipeline (§7.2)

```
Raw Markdown Nodes → Regex Redaction Engine → Sensitivity Classifier
  → SAFE: 发送到 Cloud LLM
  → SENSITIVE: 阻断 + 记录 Warning
```

#### 1.1 Stage 1: Regex Redaction Engine

在任何远端传输前应用模式替换：

```python
DEFAULT_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]'),           # API keys
    (r'[^@\s]+@[^@\s]+\.[^@\s]+', '[REDACTED_EMAIL]'),        # Email addresses
    (r'\b\d{1,3}(\.\d{1,3}){3}\b', '[REDACTED_IP]'),          # IPv4 addresses
    (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
     '[REDACTED_UUID]'),                                         # UUIDs
    (r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
     '[REDACTED_JWT]'),                                          # JWT tokens
]
```

- 用户可通过 `config.toml → [sanitization] custom_patterns` 添加自定义模式
- 替换不可逆——只在临时传输 payload 上操作，**不修改原始 Markdown 文件**

```python
class RedactionEngine:
    def redact(self, text: str) -> tuple[str, list[str]]:
        """Apply all redaction patterns. Returns (redacted_text, list_of_redaction_types_applied)."""
```

#### 1.2 Stage 2: Sensitivity Classifier

基于 frontmatter 的 `sensitivity` 字段：

| Level | 行为 |
|-------|------|
| `public` | 无限制发送到远端 provider |
| `internal` | Regex 脱敏后发送，记录审计日志 |
| `private` | **永远不发送到远端 provider**。Janitor 跳过这些节点。仅限本地检索。 |

```python
class SensitivityFilter:
    def can_transmit(self, node: Node) -> bool:
    """Returns True if node can be sent to a remote provider."""
        return node.metadata.sensitivity != "private"
    
    def prepare_for_cloud(self, node: Node) -> str:
    """Prepare node content for remote transmission.
        Applies redaction for 'internal' nodes, passes through 'public' nodes."""
```

#### 1.3 Stage 3: Transmission Audit Log (§7.2)

每次远端传输记录到本地审计日志：

```
.synapse/.audit/2026-03-01T10:30:00Z_janitor_batch.json
```

内容：
```json
{
  "timestamp": "2026-03-01T10:30:00Z",
  "operation": "condense_batch",
  "node_ids_sent": ["mem_001", "mem_002"],
  "redacted_payload_hash": "sha256:abc123...",
  "llm_response_summary": "Synthesized 2 nodes into 1 reference node",
  "redactions_applied": ["API_KEY: 2", "EMAIL: 1"]
}
```

### 2. Integration Points

Sanitization Pipeline 需要集成到所有远端通信点：

| 调用场景 | Design Doc Section | 触发 |
|---------|-------------------|------|
| Session summarization (Write path) | §6 | LLM 消费节点内容时 |
| Manual condensation (`synapse condense`) | §5.3 | Janitor 推送归档批次时 |
| Conflict detection (LLM Judge) | §5.4 | 发送节点摘要给 LLM 判断时 |
| Remote embedding / rerank API | §4.2 | 使用 `provider = "remote_api"` 时 |

为每个调用点提供统一的 `sanitize_for_cloud(nodes: list[Node]) -> list[str]` 接口。

### 3. Local File Protection (§7.3)

- **文件权限**：`.synapse/` 目录设置为 `700`（owner-only）
- **MCP daemon 运行在用户 UID 下**
- 在 `synapse serve` 启动时检查并设置权限

### 4. Archive Hygiene (§7.3)

- `.archive/` 中超过配置保留期（默认 90 天）的文件永久删除
- **Note**: 删除逻辑的调度在 TODO-09 Janitor 中实现，此处提供删除工具函数

### 5. Encryption at Rest (§7.3)

- **不在 Synapse 中实现**
- 文档说明：依赖操作系统级全磁盘加密（FileVault / LUKS）
- 在 README/docs 中记录推荐配置

---

## Dependencies
- **TODO-01**: Config (sanitization patterns, audit log path)
- **TODO-02**: Node model (sensitivity field)

## Blocks
- TODO-08 (Write path uses sanitization before LLM calls)
- TODO-09 (Condensation uses sanitization before LLM calls)

## Acceptance Criteria
- [x] Regex Redaction 正确替换 API keys, emails, IPs, UUIDs, JWTs
- [x] 自定义 patterns 从 config.toml 加载并生效
- [x] `sensitivity: private` 节点绝不发送到 Cloud
- [x] `sensitivity: internal` 节点经过 redaction 后发送
- [x] `sensitivity: public` 节点直接发送
- [x] 审计日志记录完整（timestamp, node IDs, hash, redaction types）
- [x] `.synapse/` 目录权限设为 700
- [x] 原始 Markdown 文件不被修改（redaction 仅在传输 payload 上）
- [x] 单元测试覆盖所有 redaction 模式
- [x] 集成测试：包含敏感信息的节点 → 验证 redacted 输出

## Implementation Notes

- Added regex redaction, sensitivity filtering, audit-log writing, and archive hygiene helpers under `synapse/security/sanitization.py`
- Added config-driven custom redaction patterns via `SynapseConfig.sanitization.custom_patterns`
- Reused existing runtime bootstrap directory permissions for `.synapse/` owner-only setup
