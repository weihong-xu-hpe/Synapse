# Synapse Design Docs

本目录现在只保留**当前有效的活跃设计文档**。

如果一份文档讨论的是：

- 旧的双线架构
- REST 公开接口主线
- stdio transport 主线
- HTTP/SSE sampling 过渡方案

它应该进入 `archive/`，而不应继续留在 active design 集中。

---

## Active Docs

### 1. `streamable-mcp-single-path-architecture.md`

当前总设计入口。

回答：

- Synapse 以后是什么系统
- 为什么只保留 Streamable MCP 一条线
- 为什么 REST / stdio / legacy SSE 不再保留
- internal canonical execution layer 在新架构中的位置

### 2. `streamable-mcp-implementation-plan.md`

当前实施拆分文档。

回答：

- 先删什么
- 再建什么
- 如何把唯一主路径做通
- 如何做安全、审计和兼容性验收

### 3. `TODO-sampling-only-cutover.md`

当前执行跟踪文档。

回答：

- 在不修改 skill 文件的前提下，如何把正式执行面收口到 sampling 高层工具
- 当前哪些收口动作已经完成
- 检索一致性、public surface 与文档口径的剩余工作在哪里

### 4. `TODO-agent-validation-followup.md`

基于一次真实 agent 视角验收形成的跟进 TODO。

回答：

- 为什么服务端“已经能做”的能力，在 agent 侧仍可能表现为不可见或不可编排
- 为什么 MCP tool result 需要收口到标准 `structuredContent` 语义
- lifecycle 工具在 `no_op` / `keep_as_note` / `recommend_manual_review` 场景下还需要补哪些显式回执
- 接下来优先修什么，怎样用回归测试把行为钉住

---

## Archive Policy

`archive/` 下的文档默认表示：

- 历史设计
- 已 superseded 的设计
- 过渡性技术说明
- 已完成但不再作为当前方向依据的实施材料

这些文档可以参考，但**不能再作为当前实现依据**。

另外，根目录下的下列文件也只应视为历史注记，而不是当前架构说明：

- `docs/http-sampling-compatibility.md`
- `docs/http-mcp-full-test-playbook.md`

---

## Reading Order

如果你是第一次看当前设计，只需要按这个顺序读：

1. `streamable-mcp-single-path-architecture.md`
2. `streamable-mcp-implementation-plan.md`
3. `TODO-sampling-only-cutover.md`
4. `TODO-agent-validation-followup.md`

读完这三份，再决定是否需要查看 `archive/` 里的历史材料。
