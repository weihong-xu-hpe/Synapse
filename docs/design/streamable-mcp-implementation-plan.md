# Synapse — Streamable MCP 单线实施拆分

> **文档状态**: 执行计划（持续更新）  
> **日期**: 2026-03-16  
> **主文档**: `docs/design/streamable-mcp-single-path-architecture.md`

---

## 1. 本文目的

这份文档不再讨论“要不要双线”——那件事已经在总设计里收口了。

这里回答的是：

> **如果我们接受 Streamable MCP 单线架构，具体先删什么、再建什么、最后怎样验收。**

实施原则非常明确：

- 先砍掉错误方向
- 再打通唯一主路径
- 最后补强安全、审计和 host compatibility

而不是继续让 fallback 与目标路径并存。

### 1.1 当前进度快照（2026-03-16）

- Phase 0：已完成，active design docs 已收口为单线架构 + 实施计划 + 执行 TODO
- Phase 1：对 public surface 与 active guidance 的收口已基本完成；低层工具不再公开暴露
- Phase 3 的关键依赖项已完成一部分：高层写入候选检索与读取检索已统一，`query_hint` 已改为增强式联合召回
- 剩余重点是继续清理历史叙事、补齐运行时收尾与真实 host compatibility 记录

---

## 2. 总体阶段

建议拆成五个阶段：

1. **Phase 0 — 文档与接口去歧义**
2. **Phase 1 — 删除旧路径与旧叙事**
3. **Phase 2 — 建立 Streamable transport runtime**
4. **Phase 3 — 接通 sampling 高层工具闭环**
5. **Phase 4 — 安全、审计、兼容性验收**

---

## 3. Phase 0 — 文档与接口去歧义

### 目标

让文档先只剩一个北极星，避免实现阶段还被旧叙事拖拽。

### 要做的事

1. 建立新的 active 设计入口
2. 归档双线/REST/stdio/SSE sampling 旧文档
3. 将“single path = Streamable MCP”写成新的唯一设计基准
4. 把 implementation plan 与 architecture 分离

### 交付物

- 新总设计文档
- 新实施拆分文档
- 归档目录与 active design index

### 验收标准

- active 设计目录中不再存在相互冲突的产品叙事
- 团队可以只读两份 active 文档就理解未来方向

---

## 4. Phase 1 — 删除旧路径与旧叙事

### 目标

在代码与文档层同时清除会误导未来实现的旧路径。

### 必须删除或关闭的内容

#### transport

- stdio transport
- legacy SSE sampling transport
- 基于普通 HTTP 的 sampling bridge 路径

#### public interface

- REST API 公开入口
- 任何把 REST 视作正式集成面的文档与测试

#### docs

- external-skills path 作为正式产品线的表述
- dual-path / fallback / recommended alternative 表述

### 注意

这一阶段不是“功能下线以后先空着”，而是：

- 旧路径删除
- 目标路径明确为唯一待完成主路径

### 验收标准

- 代码中不存在继续鼓励使用旧 transport 的默认入口
- 文档中不存在“你也可以先用别的”的主叙事

---

## 5. Phase 2 — 建立 Streamable transport runtime

### 目标

把 transport 层做成真正服务于 sampling 的 first-class runtime。

### 核心工作

1. 建立 session-aware runtime
2. 能力协商（sampling capability、initialized lifecycle）
3. request id / correlation / timeout 管理
4. Streamable transport adapter
5. 把 transport-specific state 从旧 MCP runtime 中剥离

### 设计要求

- 不再围绕 stdio 生命周期设计 runtime
- 不再围绕 SSE bridge 设计 message flow
- 直接按 Streamable MCP 的长期语义建模

### 交付物

- Streamable session manager
- transport runtime abstraction
- capability-aware session state

### 验收标准

- 一个 Streamable MCP client 可以完成基本会话建立与工具调用
- runtime 结构不再依赖旧 transport 假设

---

## 6. Phase 3 — 接通 sampling 高层工具闭环

### 目标

让单线架构真正变成可用系统，而不是只有 transport 壳子。

### 需要接通的高层工具

- `write_memory`
- `run_dreamer`

### 工作内容

1. 将 sampling request / response 生命周期接到 Streamable runtime
2. 将当前 service 层 sampling 流程迁移到新 transport 抽象
3. 保留 internal canonical execution layer 作为编译落点
4. 确保 `decision / evidence / execution` 结构不退化

### 要求

- 不允许为“先跑起来”而继续偷偷依赖旧 fallback 路径
- 不允许通过禁用高层工具来掩盖 transport 未完成

### 验收标准

至少要在真实 server 下完整跑通：

- sampling-backed create
- complement
- supersede
- lifecycle plan_only
- lifecycle execute_safe_actions

---

## 7. Phase 4 — 安全、审计、兼容性验收

### 目标

把主路径从“能跑”提升为“可以被信任地运行”。

### 工作内容

#### 安全

- auth-bound session
- duplicate response 防护
- expired response 防护
- session close cancellation

#### 审计

- sampling request created
- sampling response received
- timeout
- cancelled
- resume success / failure

#### 兼容性

- 建立 host compatibility matrix
- 区分 server implemented 与 host verified

### 验收标准

必须完成三类验证：

1. synthetic client 端到端验证
2. failure matrix 验证
3. 至少一个真实 host 的兼容性验证

---

## 8. 代码层面的推荐拆分

为了支持单线架构，代码实现建议按下面拆：

### A. Transport Runtime

负责：

- session
- capability negotiation
- stream lifecycle
- request correlation

### B. Sampling Bridge

负责：

- sampling request emission
- response matching
- timeout / cancellation / duplicate handling

### C. High-Level Tool Orchestration

负责：

- evidence assembly
- prompt construction
- normalized structured result
- compile to internal execution layer

### D. Internal Canonical Execution Layer

负责：

- low-level write / retrieval / status mutation
- markdown persistence
- sqlite sync

### 原则

transport、sampling、tool orchestration、execution 四层必须解耦。

否则未来很容易再次滑回：

- transport-specific behavior 侵入 service 层
- fallback logic 侵入工具语义

---

## 9. 需要显式移除的测试与文档

为了防止旧方向反复回潮，实施时建议同步删掉或改掉以下内容：

### 测试

- stdio transport 主路径测试
- HTTP/SSE sampling bridge 主路径测试
- REST 公开接口主路径测试

### 文档

- usage/configuration 中关于多模式选择的叙事
- compatibility 文档中关于 stdio fallback 的推荐语
- old playbooks 中围绕 HTTP/SSE sampling bridge 的主测试剧本

这里的原则不是“测试越多越好”，而是：

> **测试也必须服务于唯一目标路径，而不是继续把历史兼容层固化成永远要维护的行为。**

---

## 10. 推荐的审批后执行顺序

如果你批准这套单线方案，我建议实际施工按下面顺序开始：

1. **完成文档归档与 design index 整理**
2. **删除或封存旧 transport / REST 对外入口**
3. **重构 runtime，建立 Streamable session 模型**
4. **迁移 sampling lifecycle 到 Streamable**
5. **接通高层工具闭环**
6. **重建测试矩阵与 host compatibility matrix**

---

## 11. 最终交付标准

单线实施完成时，应满足：

- 对外只有一条正式路径：Streamable MCP
- sampling 是主路径默认能力，而不是附加特性
- REST / stdio / legacy SSE bridge 不再作为正式系统组成部分存在
- 高层 sampling 工具全部可跑
- internal canonical execution layer 仍然稳定可测
- host compatibility 被真实记录，而不是靠 fallback 掩饰

---

*文档版本: 2026-03-16*
