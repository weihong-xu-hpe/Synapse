# Synapse Design Docs

本目录只保留**当前有效的活跃设计文档**。

已淘汰的文档（旧双线架构、REST 公开接口、HTTP/SSE 过渡方案等）已归档至 `archive/`。

---

## Active Docs

### 1. `streamable-mcp-single-path-architecture.md`

当前总设计入口。

回答：

- Synapse 以后是什么系统
- 为什么只保留 Streamable MCP 一条线
- internal canonical execution layer 在新架构中的位置

### 2. `streamable-mcp-implementation-plan.md`

当前实施拆分文档。

回答：

- 先删什么
- 再建什么
- 如何把唯一主路径做通
- 如何做安全、审计和兼容性验收

### 3. `TODO-local-llm-upgrade.md`

本地 LLM 升级执行计划（当前活跃）。

回答：

- 为什么把决策层从 sampling 回调切换为本地 LLM（`LocalLLMDecider`）
- sampling 基础设施如何简化为可选高级路径
- Dreamer 如何从手动触发改为自动巡逻
- OKF 格式约束如何落地
- 4 个子阶段的优先级与依赖关系

### 4. `TODO-dreamer-stats-thresholds.md`

Dreamer 观测指标与阈值配置化设计（待评审）。

回答：

- 为什么当前 0.75 / 0.8 / 0.9 类门槛只能作为保守默认值，不能长期写死
- 如何通过 `stats` 暴露 Dreamer/write path 的长期趋势和候选漏斗
- 哪些 Dreamer 门槛应该优先配置化
- 如何用实际运行数据微调 Mac 本地环境的阈值

### 5. `synapse-v2-memory-consolidation-and-handbook-architecture.md`

v2 记忆系统总设计（2026-10-31，待评审）。回答：

- 陈述性记忆（Synapse）与程序性记忆（`omp-skill-evolver` / Autolearn）的系统边界与责任矩阵
- Pillar 1 睡眠合并（Dreamer 2.0：语义合并 / 超大节点拆分 / 12h+事件触发）
- Pillar 2 使用信号保留与遗忘曲线（三级计数器 / 项目活动时钟 / ACT-R 激活 / hub 惩罚）
- Pillar 3 人类/AI 双重可读、可共享给 junior 的手册（物理目录分离 + 编译式 handbook）
- 四阶段落地路线图与验收标准

---

## Archive Policy

`archive/` 下的文档默认表示：

- 历史设计
- 已 superseded 的设计
- 过渡性技术说明
- 已完成但不再作为当前方向依据的实施材料

这些文档可以参考，但**不能再作为当前实现依据**。

目录约定：

- `archive/completed-implementation-history/` — 已完成的初始实施拆分（TODO-00 … TODO-14）
- `archive/completed-execution-history/` — 已完成或已被现行架构取代的执行型 TODO（sampling 收口、tool 收口、agent 验收跟进、lifecycle 简化草案）
- `archive/superseded-pre-streamable-reset/` — 假设双线 / REST / stdio / HTTP-SSE 为现行方向的旧设计

另外，根目录下的下列文件也只应视为历史注记，而不是当前架构说明：

- `docs/http-sampling-compatibility.md`
- `docs/http-mcp-full-test-playbook.md`

---

## Reading Order

如果你是第一次看当前设计，只需要按这个顺序读：

1. `streamable-mcp-single-path-architecture.md`
2. `streamable-mcp-implementation-plan.md`
3. `TODO-local-llm-upgrade.md`
4. `TODO-dreamer-stats-thresholds.md`
5. `synapse-v2-memory-consolidation-and-handbook-architecture.md`

读完这几份，再决定是否需要查看 `archive/` 里的历史材料。
