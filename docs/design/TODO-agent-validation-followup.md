# Synapse TODO — Agent 视角验收后的修复跟进

> **文档状态**: 执行 TODO  
> **日期**: 2026-03-16  
> **背景**: 基于一次真实 agent 视角的 MCP / sampling 工具验收整理

## 1. 这份 TODO 解决什么

这份 TODO 不再讨论架构方向是否正确。

当前结论已经很明确：

- sampling 高层写入链路是通的
- `create` / `complement` / `supersede` 的核心执行语义已成立
- confidence gate 已经工作
- Streamable runtime 的 timeout / duplicate / close 语义已有服务端证据

但从 **agent 真正使用** 的角度，仍有两类问题阻碍“可依赖性”：

1. **结果可见性不足** — 服务端有结果，但 agent 侧可能看到空输出或缺少可消费回执
2. **lifecycle 回执不够清晰** — `review` / `condense` / `promote` 在 no-op、skip、未执行时语义不够显式

这份 TODO 的目标是：

> **把“后端已经能做事”推进到“agent 能稳定看见、理解并继续编排”。**

---

## 2. 本轮验收后的修正判断

## 2.1 已确认成立的点

- `write_memory` 能完成 sampling → canonical write 的闭环
- `create` / `complement` / `supersede` 已真实落盘
- SQLite `nodes` / `nodes_fts` / `nodes_vec` / `edges` 已反映写入结果
- confidence threshold 会阻止低置信度决策直接执行

## 2.2 重新检查后的修正

此前一次人工观察曾怀疑 `supersede` 没有正确回写旧 markdown 文件。

复查后发现：

- 旧节点 frontmatter 已更新为 `status: superseded`
- `superseded_by` 已回写
- superseded banner 已存在

因此该项当前不再视为“已确认 bug”，而应转为：

> **补强一致性回归测试，防止该行为未来回退。**

---

## 3. P0 — 先修 agent 可见性

## 3.1 标准化 MCP tool result 输出

### 问题

当前 MCP tool result 采用了非标准的 `content.type = "json"` 形式。

这在仓库内测试可以工作，但真实 MCP host / agent 可能忽略这类 content item，导致出现：

- 服务端日志显示 response payload 非空
- agent 侧却看到空输出

### 目标

将 tool result 改为更标准、更兼容的形式：

- `content` 使用 `text`
- `structuredContent` 承载结构化 JSON 结果

### 验收标准

- `search_memory`、`write_memory`、`run_dreamer` 的结果能被标准 MCP host 正常看见
- 不再依赖 `content.type = "json"` 作为唯一结构化返回方式
- 现有测试迁移到读取 `structuredContent`（必要时兼容 text）

---

## 4. P1 — 让 lifecycle 工具对 agent 真正可编排

## 4.1 给 lifecycle execution 返回显式语义

### 问题

当前 `_execute_lifecycle_plan(...)` 在以下场景下语义不够清楚：

- `plan_only`
- `no_op`
- `keep_as_note`
- `recommend_manual_review`
- outcome 需要 draft，但 sampling 未提供 draft

agent 很难区分：

- 没执行是否合理
- 是跳过、阻塞、建议人工处理，还是实际失败

### 目标

统一 execution payload，至少明确：

- 是否执行
- 为什么没有执行
- 当前属于哪种 outcome / action
- 是否有 warning

### 建议最小字段

- `executed`
- `tool`
- `action_taken`
- `result`
- `warnings`
- `skipped_reason`

### 验收标准

- `plan_only` 返回显式未执行原因
- `no_op` / `keep_as_note` / `recommend_manual_review` 不再误报“缺少可执行 draft”
- `condense` / `promote` 的 no-op 对 agent 可解释

---

## 4.2 修正 no-op 与 draft 校验顺序

### 问题

当前 `_execute_lifecycle_plan(...)` 先检查 `draft is None`，再判断 outcome 是否属于 no-op / keep。

这会导致某些本应“正常不执行”的情况，返回误导性的 warning：

- `No executable draft was produced for this lifecycle plan.`

### 目标

优先根据 outcome 判断是否本来就不该执行，再决定是否需要 draft。

### 验收标准

- `no_op`
- `keep_as_note`
- `recommend_manual_review`

这些 outcome 在 `execute_safe_actions` 模式下，返回合理的 skipped 语义，而不是 draft 缺失警告。

---

## 5. P2 — 用回归测试把行为钉住

## 5.1 MCP tool result 暴露层测试

### 目标

验证 public MCP tool surface 采用标准返回格式后，结果仍可被消费。

### 至少覆盖

- `search_memory`
- `write_memory`
- `run_dreamer`

### 验收标准

- 测试从 `structuredContent` 读取结果成功
- `content` 中至少保留可读 text

---

## 5.2 lifecycle no-op / keep 行为测试

### 目标

确保 lifecycle planner 给出 no-op / keep 类 outcome 时，execution payload 语义稳定。

### 至少覆盖

- `run_dreamer` triage with empty candidates (no-op)
- `run_dreamer` triage producing skip/keep decisions
- `run_dreamer` conflict resolution with no disputed nodes (no-op)

### 验收标准

- 不再出现误导性 draft warning
- `action_taken` / `skipped_reason` 断言成立

---

## 5.3 supersede 文件一致性测试

### 目标

虽然复查后 supersede 文件回写目前是正确的，但必须补测试防止回退。

### 至少覆盖

- 旧节点 frontmatter 变为 `superseded`
- `superseded_by` 正确
- old node banner 存在
- 新节点 `supersedes` 正确
- DB 与 markdown 一致

---

## 6. 当前建议执行顺序

1. **标准化 MCP tool result**
2. **修正 lifecycle execution / no-op 语义**
3. **补 MCP 暴露层测试**
4. **补 lifecycle no-op 测试**
5. **补 supersede 一致性测试**
6. **视结果决定是否继续处理 test data 清理与文档扩写**

---

## 7. 一句话结论

这轮修复不是再去证明 sampling 主路径是否成立，而是：

> **把已经成立的服务端能力，改造成 agent 看得见、能理解、能继续编排的正式工作流。**
