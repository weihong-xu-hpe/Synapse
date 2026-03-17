# TODO-14 代码架构审计 — 零智能原则合规检查

> **状态**: 待修复（3 处问题，均属轻量级）
> **审计日期**: 2026-03-09
> **审计范围**: `server/`、`retrieval/`、`models/`、`config.py`、`lifecycle/`（共 11 个文件）
> **架构原则**: Synapse 是纯执行层，零智能——不做任何内容判断或意图推断。

---

## 1. 审计结论摘要

代码整体高度符合"零智能"设计规范。**主要写路径完全干净**：`integrate_knowledge` 机械执行外部传入的 `action`，`search_existing_nodes` 纯搜索返回，`write_path.py` 无任何遗留的 `CONFLICT_UNCLEAR` 代码。

发现 **1 处潜在违反**、**2 处次要问题**，均集中在 `condensation.py` 和 `config.py`，与写路径和 MCP 工具层无关。

| 编号 | 文件 | 类型 | 严重性 | Scope |
|------|------|------|--------|-------|
| V-1  | `lifecycle/condensation.py` | 潜在违反：协议预留了 LLM 注入接口 | Medium | Small |
| M-1  | `config.py` | 次要问题：死字段 `similarity_threshold` | Low | Very Small |
| M-2  | `server/mcp.py` | 次要问题：工具描述混入了 Skill 级操作指南 | Very Low | Very Small |

---

## 2. 违反 V-1：`ArchiveCondenser` 协议预留了 LLM 智能注入接口

### 位置

`synapse/lifecycle/condensation.py`：
- 第 51–63 行：`ArchiveCondenser` Protocol 定义
- 第 75 行：`DeterministicArchiveCondenser.requires_sanitized_payloads = False`
- 第 323–331 行：`_attempt_condensation()` 中的条件分支

### 问题说明

`ArchiveCondenser` Protocol 包含 `requires_sanitized_payloads: bool` 字段：

```python
# condensation.py, lines 51–63
class ArchiveCondenser(Protocol):
    """Protocol for pluggable archive condensation implementations."""

    name: str
    requires_sanitized_payloads: bool   # ← 此字段暗示设计支持外部 LLM condenser

    def synthesize(
        self,
        nodes: Sequence[Node],
        *,
        now: datetime,
        sanitized_payloads: SanitizedPayloadBatch | None = None,
    ) -> CondensationDraft:
        """Produce a single synthesized draft from archived nodes."""
```

`_attempt_condensation()` 中，当 `condenser.requires_sanitized_payloads is True` 时，会调用 `sanitize_nodes_for_cloud()` 并将 sanitized payloads 传给 condenser：

```python
# condensation.py, lines 323–331
if condenser.requires_sanitized_payloads:
    payload_batch = sanitize_nodes_for_cloud(
        nodes,
        config=self.config,
        audit_dir=self.runtime_paths.audit,
        operation="condense_archive",
        llm_response_summary=f"Preparing {len(nodes)} archived nodes for condensation",
        timestamp=self._utc_now(),
    )
return condenser.synthesize(nodes, now=self._utc_now(), sanitized_payloads=payload_batch), None, attempts
```

这意味着设计显式预留了：**在 Synapse 内部生命周期流程中调用外部 LLM API 对归档节点做语义整合**，违反了 "Synapse 零智能" 原则。设计文档（`external-memory-skill-design.md`）明确要求所有内容判断逻辑属于外部 LLM Agent 的职责。

### 当前状态

**当前代码未实际触发违反**：目前唯一的实现 `DeterministicArchiveCondenser` 将 `requires_sanitized_payloads = False`（第 75 行），是纯确定性的模板填充，不做语义分析。但 Protocol 架构明确支持未来注入 LLM condenser，属于潜在违反和技术债。

### 期望修复

**选项 A（推荐）**：彻底删除 LLM condenser 接口，Protocol 仅允许确定性实现：

```python
# condensation.py — 修改后
class ArchiveCondenser(Protocol):
    """Protocol for pluggable deterministic archive condensation.

    Implementations MUST be fully deterministic. Synapse does not invoke
    any external LLM or inference service inside the lifecycle system.
    """

    name: str

    def synthesize(
        self,
        nodes: Sequence[Node],
        *,
        now: datetime,
    ) -> CondensationDraft:
        """Produce a single synthesized draft from archived nodes."""
```

同步清理 `_attempt_condensation()` 中的 `sanitize_nodes_for_cloud` 分支：

```python
# condensation.py — _attempt_condensation() 修改后
def _attempt_condensation(
    self,
    condenser: ArchiveCondenser,
    nodes: Sequence[Node],
    *,
    max_attempts: int | None = None,
) -> tuple[CondensationDraft | None, str | None, int]:
    ...
    for attempt_index in range(allowed_attempts):
        attempts += 1
        try:
            return condenser.synthesize(nodes, now=self._utc_now()), None, attempts
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            ...
```

同步清理头部导入：删除 `from synapse.security import SanitizedPayloadBatch, sanitize_nodes_for_cloud`（第 14 行），以及 `synthesize()` 方法签名中的 `sanitized_payloads` 参数（`DeterministicArchiveCondenser` 第 86 行、第 89 行）。

**选项 B（保守，不推荐）**：保留协议，但在 `ArchiveCondensationService.__init__` 加入运行时守卫，显式拒绝注入 LLM condenser：

```python
if condenser is not None and getattr(condenser, "requires_sanitized_payloads", False):
    raise ValueError(
        "Synapse lifecycle does not support LLM-based condensers. "
        "This would violate the zero-intelligence architecture principle."
    )
```

推荐选项 A：彻底清理，不留死路，代码意图一目了然。

### 是否需要更新测试

需检查 `tests/test_lifecycle.py`，确认是否存在对 `requires_sanitized_payloads=True` 的测试覆盖；若有，需删除相关测试 case。`sanitize_nodes_for_cloud` 的独立单元测试（`tests/test_sanitization.py`）不受影响。

---

## 3. 次要问题 M-1：`config.py` 中 `RetrievalSettings.similarity_threshold` 为死字段

### 位置

`synapse/config.py`，`RetrievalSettings` 类，第 96 行：

```python
class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: RetrievalEngine = "sqlite"
    rrf_k: int = Field(default=60, ge=1)
    similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)  # ← 死字段
    top_k: int = Field(default=3, ge=1)
```

### 问题说明

`config.retrieval.similarity_threshold`（默认值 0.80）在整个代码库中从未被读取：

- `retrieval/pipeline.py` 的 `search()` 方法不按此阈值过滤结果，直接返回 top-k。
- `service.py` 的 `search_existing_nodes()` 从请求参数中接收 `similarity_threshold`（默认 0.5），与此配置字段无关。
- 全局 `grep retrieval.similarity_threshold` 无任何命中。

**误导性**：字段名暗示 Synapse 会在内部按相似度自动过滤检索结果——这与"零智能/零决策"原则冲突。实际上，阈值由调用方在 `search_existing_nodes` 请求参数中决定，external Agent 自行决策阈值高低。`config.toml` 示例中存在此字段（见 `tests/test_write_path.py` 第 38 行的嵌入配置字符串），可能会误导后续维护者。

### 期望修复

确认无任何运行时路径读取此字段后，从 `RetrievalSettings` 中删除：

```python
class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: RetrievalEngine = "sqlite"
    rrf_k: int = Field(default=60, ge=1)
    # similarity_threshold 已删除 — 阈值由调用方通过 API 参数决定，Synapse 不做内部过滤
    top_k: int = Field(default=3, ge=1)
```

同步更新：
1. `config.toml`（项目根目录）中的示例配置，删除 `similarity_threshold` 行
2. `docs/configuration.md` 中如有对此字段的说明，一并删除
3. 测试文件嵌入配置字符串（`tests/test_write_path.py` 第 38 行等）中删除此行

### 是否需要更新测试

需检查全部测试配置字符串：`grep -r "similarity_threshold" tests/`，逐一删除配置行（不影响测试逻辑，因为该字段从未被断言）。

---

## 4. 次要问题 M-2：`mcp.py` 工具描述嵌入了 Skill 级操作指南

### 位置

`synapse/server/mcp.py`，`integrate_knowledge` 工具定义，第 58–64 行：

```python
MCPToolDefinition(
    name="integrate_knowledge",
    description=(
        "Write a knowledge node with an explicit action decided by the caller. "
        "Use search_existing_nodes first to find similar nodes, then call this "  # ← Skill 级指南
        "with action='create' (no targets), 'supersede' (replace targets), or "
        "'complement' (cross-link with targets)."
    ),
    ...
)
```

### 问题说明

`"Use search_existing_nodes first to find similar nodes"` 是外部 Agent 的调用编排指南，属于 Skill Prompt 的职责范围（`external-memory-skill-design.md` 第 3 节已覆盖），不应当嵌入 MCP 工具的技术描述。工具描述应仅说明"此工具做什么"，而非"调用者应如何编排调用顺序"。

**注意**：这是文档文本，不是可执行代码，**无功能性违反**。严重性极低。

### 期望修复

将 description 精简为纯工具功能说明：

```python
description=(
    "Execute a write action on the knowledge graph. "
    "action must be 'create' (new node, no targets), "
    "'supersede' (new node replaces target_node_ids), or "
    "'complement' (new node cross-links with target_node_ids). "
    "The caller decides action and target_node_ids before invoking this tool."
),
```

操作编排指南（"先调用 search_existing_nodes"）保留在外部 Skill Prompt（`external-memory-skill-design.md`）中。

### 是否需要更新测试

不需要。工具描述字符串不影响功能，相关测试不应硬编码 description 内容。

---

## 5. 通过确认 — 无违反组件

以下组件经逐行检查，**无架构违反**：

### `server/service.py` ✅

- `integrate_knowledge()`（第 105–205 行）：纯机械执行外部传入的 `action`，无内部相似度搜索，无冲突检测逻辑。注释明确声明："The caller is responsible for running search_existing_nodes first and deciding which action to take. Synapse does not infer intent here."
- `search_existing_nodes()`（第 274–282 行）：纯搜索返回，`similarity_threshold` 由调用方传入，不做任何额外过滤或分类决策。
- 写路径中无任何embedding搜索调用：`integrate_knowledge` 完全不调用 `_search_existing_nodes_payload`，两条路径严格分离。
- 无任何 `CONFLICT_UNCLEAR` 遗留代码。

### `server/write_path.py` ✅

- `IntegrateAction` 枚举仅有 `CREATE`、`SUPERSEDE`、`COMPLEMENT`，不存在 `CONFLICT_UNCLEAR`。
- 类文档注释明确："The decision of which action to use is made externally (by the calling agent or skill) after inspecting results from search_existing_nodes. Synapse only executes the action — it does not infer intent."

### `server/schemas.py` ✅

- `IntegrateKnowledgeRequest` 要求 `action` 由调用方显式提供，默认值 `CREATE` 是保守兜底，不是自动推断。
- 注释："Explicit decision from the external agent/skill — Synapse only executes."
- 无任何隐式智能字段。

### `server/app.py` ✅

- 纯路由分发，无业务逻辑，无内容判断。
- `write_node` REST 端点存在（`POST /api/v1/nodes`）但未暴露为 MCP 工具，符合设计文档第 6 节约定。

### `server/mcp.py`（功能部分）✅

- 工具路由纯净，仅做参数校验和分发，无决策逻辑。
- 5 个 MCP 工具均为纯执行/查询接口，无内部智能。

### `retrieval/pipeline.py` ✅

- 全部为确定性计算：RRF 融合、rerank 分数、时间 decay 乘子（tier 参数化）、status 惩罚乘子。
- `STATUS_MULTIPLIERS`（`SUPERSEDED: 0.1`, `DISPUTED: 0.5`）是硬编码的固定权重，不是内容判断。
- `[DISPUTED]` 前缀（`_format_context_block()` 第 210 行）是格式化注解，不是决策逻辑。
- 检索管道与写路径完全隔离：无任何写操作调用或冲突检测。

### `models/node.py` ✅

- `NodeStatus` 枚举：`ACTIVE`、`SUPERSEDED`、`DISPUTED`，无 `CONFLICT_UNCLEAR`。
- `TIER_WORD_LIMITS` 仅用于外部主动校验（`word_count_validation()`），不触发自动分类或自动降级。
- `importance` 在写路径中由 `service.py` 硬编码为 `0.5`，不作为推断输入或分类依据。

### `config.py` ✅

- 无"智能"配置项：所有 embedding/reranker 设置均为技术参数（模型名、超时、维度），非决策参数。
- decay 阈值（`note_factor`、`memory_factor`、`*_janitor_days`）是机械衰减系数，不是内容决策参数。

### `lifecycle/janitor.py` ✅

- 孤儿归档、superseded 归档、archive 清理均基于配置的**时间阈值**，不做任何内容分析。
- `disputed_count > 5` 检查（第 186–191 行）仅生成被动警告消息，不采取任何动作，不修改节点状态。
- 无从 `disputed` 到其他状态的自动转换逻辑；`disputed` 状态只能由外部 Agent 通过 `update_node_status` 显式设置。

### `lifecycle/condensation.py`（当前实现部分）✅

- `DeterministicArchiveCondenser.synthesize()`：纯确定性模板填充——列出节点 ID、标题、`_summarize()` 首行截取（180 字符上限）、`_collect_common_tags()` 频率统计（前 3 名）。
- 无语义聚类，无 LLM 调用，无相似度分析。
- `_summarize()`（第 117–123 行）：返回 content 中第一个非空行（截断 180 字符），非摘要生成。
- `_collect_common_tags()`（第 109–115 行）：简单频率统计，非语义分类。

---

## 6. 关联工作项

| 编号 | 优先级 | 文件 | 操作 |
|------|--------|------|------|
| V-1 | Medium | `lifecycle/condensation.py` | 删除 `requires_sanitized_payloads` 字段及 `sanitize_nodes_for_cloud` 分支（选项 A） |
| M-1 | Low | `config.py`、`config.toml`、测试配置 | 删除死字段 `RetrievalSettings.similarity_threshold` |
| M-2 | Very Low | `server/mcp.py` | 精简 `integrate_knowledge` 工具描述 |

---

*审计者：GitHub Copilot（Claude Sonnet 4.6）*
*文档版本：2026-03-09*
