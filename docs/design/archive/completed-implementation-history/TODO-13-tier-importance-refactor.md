# TODO-13: Tier & Importance Refactor

> **状态**: 待审批  
> **背景**: 现有 `concept/decision/reference` 三档命名来自知识领域分类，与 tier 在代码中的唯一实际作用（控制 decay 速率和 janitor 阈值）语义不符。`importance` 字段的唯一实际用途（janitor reference 孤儿归档的附加条件）也不应由外部系统传入，访问次数本身就是最好的重要性代理。

---

## 变更范围

### 1. `NodeTier` 枚举：三档 → 两档

| 旧值 | 新值 | 含义 |
|---|---|---|
| `concept` | `note` | 速记、当前任务上下文，快速流转 |
| `decision` | （合并到 `memory`） | — |
| `reference` | `memory` | 重点记忆，长期保留 |

**文件**: `synapse/models/node.py`

---

### 2. `DecaySettings`：三组参数 → 两组

| 旧参数 | 新参数 | 默认值 |
|---|---|---|
| `concept_factor` | `note_factor` | `0.90`（~7 天半衰期） |
| `decision_factor` | ~~删除~~ | — |
| `reference_factor` | `memory_factor` | `0.992`（~90 天半衰期） |
| `concept_janitor_days` | `note_janitor_days` | `7` |
| `decision_janitor_days` | ~~删除~~ | — |
| `reference_janitor_days` | `memory_janitor_days` | `90` |
| `reference_janitor_max_importance` | ~~删除~~（整个条件移除） | — |

**文件**: `synapse/config.py`

---

### 3. `importance` 字段：从外部 API 完全移除

- `WriteNodeRequest` 删除 `importance` 字段
- `IntegrateKnowledgeRequest` 删除 `importance` 字段
- `NodeMetadata` 中 `importance` 保留但**不再接受外部传入**，固定写入默认值 `0.5`，由系统内部根据 `access_count` 自然反映
- Janitor 中 `reference_janitor_max_importance` 条件整体删除（改为只看 `last_accessed` + in-degree）

**文件**: `synapse/server/schemas.py`, `synapse/server/service.py`, `synapse/lifecycle/janitor.py`

---

### 4. 级联更新

| 文件 | 变更内容 |
|---|---|
| `synapse/models/node.py` | `NodeTier` 枚举值改为 `note / memory` |
| `synapse/config.py` | `DecaySettings` 删减参数，`get_factor` 映射更新 |
| `synapse/retrieval/pipeline.py` | `apply_decay` 中 tier 映射更新；context snippet 中 Tier 显示更新 |
| `synapse/lifecycle/janitor.py` | 三个 tier 批次 → 两个；移除 `max_importance` 参数 |
| `synapse/lifecycle/condensation.py` | `NodeTier.REFERENCE` → `NodeTier.MEMORY` |
| `synapse/storage/sqlite.py` | `find_orphan_candidates` 签名更新，移除 `max_importance` 参数 |
| `synapse/server/schemas.py` | 移除 `importance` 字段 |
| `synapse/server/service.py` | 移除 `importance` 参数传递 |
| `synapse/server/app.py` | 移除 `importance` 传参 |
| `config.toml` | `[decay]` 段更新 |
| `tests/` | 所有测试中 `tier="concept/decision/reference"` → `"note/memory"`；移除 `importance` 传参 |
| `docs/design/external-memory-skill-design.md` | tier 说明更新 |

---

## 不变的内容

- `NodeType`（`transient / persistent`）：保留，与 tier 正交，用途不同
- `SensitivityLevel`（`internal / private / public`）：保留
- `NodeStatus`（`active / superseded / disputed`）：保留
- decay 公式本身（`factor ^ elapsed_days`）：保留，只换参数名
- janitor 的 superseded 归档逻辑：保留（7 天归档）

---

## 预期测试结果

变更后 `python -m pytest tests/ -q` 应全部通过，无 tier / importance 相关的 fixture 残留。
