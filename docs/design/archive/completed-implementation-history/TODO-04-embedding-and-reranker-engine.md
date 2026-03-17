# TODO-04: Embedding & Reranker Engine

## Status: COMPLETED
## Priority: P0 (Core ML pipeline — retrieval depends on this)
## Design Doc Section: §4.2.1, §4.2.2

---

## Summary

实现 Synapse 的嵌入与重排序能力抽象层。Synapse **不负责启动推理进程**；本地模型通过 **Ollama** 提供，远端模型通过 **URL/API client** 提供。该 TODO 的重点是 provider abstraction、配置切换、降级行为，以及与检索层稳定解耦。

---

## Detailed Requirements

### 1. Embedding Model: `BAAI/bge-m3`

**主选模型**，核心优势：
- **中英跨语言检索**：基于海量中英平行语料训练，混合语言查询可靠检索
- **8,192 token 上下文**：覆盖最大 `reference` 节点（3,500 words ≈ 5,000 tokens），40%+ 安全余量
- **1024 维输出**：匹配 sqlite-vec 配置
- **Dense + Sparse + ColBERT 三重输出**（当前仅使用 Dense，Sparse 和 ColBERT 为未来升级路径）

**本地 provider（推荐）**：
- 通过 Ollama 调用本地 embedding 接口（默认 `http://127.0.0.1:11434`）
- Synapse 只作为 client，不启动 Ollama，不自动 `pull` 模型

**远端 provider（可选）**：
- 通过配置化 URL/API client 访问 embedding 服务
- 适用于 OpenAI-compatible endpoint、自建 internal gateway、或专用 embedding 服务

### 2. Reranker Model: `bge-reranker-v2-m3`

**同家族对齐**，确保 embedding 空间和 reranker 评估一致：
- ~560M 参数，8,192 max tokens，MIT license
- 可通过本地 provider（若支持）或远端 URL/API provider 暴露为 rerank 能力
- 输入：`(query, document)` pair → 输出：相关性分数

### 3. Abstraction Interface

实现 TODO-01 定义的抽象接口：

```python
class EmbeddingEngine(Protocol):
    def embed(self, text: str) -> list[float]:
        """Compute embedding vector for a single document."""
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple documents."""
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (e.g., 1024 for bge-m3)."""

class RerankerEngine(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Rerank documents against query.
        Returns list of (original_index, relevance_score) sorted by score desc."""
```

### 4. Model Registry & Hot-Swap

根据 `config.toml` 的 `[embedding]` 和 `[reranker]` 配置实例化对应模型：

并新增 **provider** 维度：

| Provider | Meaning |
|---|---|
| `ollama` | 调用已运行的本地 Ollama 实例 |
| `remote_api` | 调用远端 URL/API 服务 |
| `builtin` | 内建 deterministic fallback / test backend |

| Config Value | Model | Params | Dim | Max Tokens |
|---|---|---|---|---|
| `bge-m3` | `BAAI/bge-m3` | ~560M | 1024 | 8,192 |
| `jina-v3` | `jinaai/jina-embeddings-v3` | ~570M | 1024 | 8,192 |
| `gte-qwen2` | `Alibaba-NLP/gte-Qwen2-1.5B` | 1.5B | 1536 | 32,768 |
| `gemma-300m` | `google/embedding-gemma-300m` | 300M | 1024 | 8,192 |

Reranker:
| Config Value | Model |
|---|---|
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` |
| `jina-reranker-v2` | `jinaai/jina-reranker-v2-base-multilingual` |

**切换 embedding provider/model 需运行 `synapse rebuild-index`**（TODO-12 实现），因为不同模型或后端的 embedding 空间可能不兼容。

### 5. Provider Configuration

- `embedding.provider = "ollama"` 时：读取本地 Ollama `base_url` 与 endpoint
- `embedding.provider = "remote_api"` 时：读取远端 `base_url`、API key、headers、timeout
- `reranker.provider` 同理
- 所有 provider 行为应通过统一接口隐藏在 engine factory 后面

### 6. Graceful Degradation (§9.1)

- **provider 不可用 / endpoint 不可达**：记录 warning 日志，向量搜索返回空结果，FTS5 关键词搜索 + graph hop 继续工作
- **不阻断启动**：MCP server 即使没有 embedding 模型也能启动
- 提供 `is_available()` 方法供其他模块检查

### 7. Performance Targets

在标准笔记本 CPU 上的期望性能：
- **Embedding**（单文档，~1000 words）：< 500ms
- **Rerank**（9 candidates）：< 2s
- **Batch embedding**（10 documents）：< 4s
- 内存峰值：< 1.5 GB（bge-m3 INT8）

### 8. Local Runtime Responsibility

- Synapse **不负责** `ollama serve`
- Synapse **不负责** `ollama pull <model>`
- 如果使用 Ollama，本地模型的安装、升级、默认启动由用户或系统服务管理
- 如果使用远端 provider，endpoint 与 credential 由配置注入

---

## Dependencies
- **TODO-01**: Project structure, config, abstract interfaces

## Blocks
- TODO-05 (File watcher uses embedding engine)
- TODO-06 (Retrieval pipeline uses reranker)
- TODO-12 (Rebuild index re-embeds all nodes)

## Acceptance Criteria
- [x] `provider = "ollama"` 时可成功调用本地 Ollama embedding endpoint
- [x] `provider = "remote_api"` 时可成功调用远端 embedding / rerank endpoint
- [x] `embed()` 返回 1024 维向量
- [x] `rerank()` 返回按相关性排序的结果
- [x] 通过 `config.toml` 切换模型（至少 bge-m3 和 jina-v3）
- [x] Graceful degradation：模型不可用时不崩溃
- [x] 单文档 embedding < 500ms（本地 Ollama / 同等级本地 provider，CPU）
- [x] 中文、英文、混合语言文本均可正常 embed
- [x] 单元测试覆盖 embed、rerank、模型切换

## Implementation Notes

- Added provider-aware config models for embedding / reranker selection and provider blocks for `ollama` and `remote_api`
- Added provider-aware factories under `synapse/embedding/engines.py`: `create_embedding_engine(..., providers=...)` and `create_reranker_engine(..., providers=...)`
- Implemented a real Ollama HTTP embedding client with `/api/tags` availability probing, `bge-m3` / `bge-m3:latest` model alias handling, and response-shape normalization
- Implemented generic remote HTTP clients for embeddings and reranking with configurable base URL, endpoints, request timeout, custom headers, and optional `api_key_env` bearer/header interpolation
- Preserved deterministic built-in fallback engines for offline tests and graceful degradation, plus unavailable sentinels when fallback is disabled
- Implemented graceful degradation for unsupported Ollama rerank paths: empty or unavailable endpoints fall back to deterministic reranking without blocking startup
- Added automated coverage for config parsing, builtin fallback behavior, Ollama provider selection, remote API auth/header behavior, unavailable-provider degradation, and an opt-in live Ollama test guarded by `SYNAPSE_TEST_OLLAMA=1`
- Live local verification on 2026-03-07 against user Ollama + `bge-m3` succeeded: 1024-dim vectors returned and warm-call latency observed at ~89ms on the current machine
