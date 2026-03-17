# TODO-01: Project Scaffolding & Configuration System

## Status: COMPLETED
## Priority: P0 (Foundation — all other TODOs depend on this)
## Design Doc Section: §4.2.2, §7.1.1, §9.3

---

## Summary

搭建 Synapse 项目的基础骨架和配置系统。包括项目目录结构、依赖管理、`config.toml` 解析、CLI 入口点、以及核心抽象接口的定义。这是所有后续 TODO 的地基。

---

## Detailed Requirements

### 1. Project Structure
创建标准 Python 项目结构（推荐 `uv` 或 `poetry` 管理依赖）：

```
synapse/
├── pyproject.toml
├── config.toml                  # 默认配置文件
├── synapse/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry: `python -m synapse`
│   ├── cli.py                   # CLI commands (click/typer)
│   ├── config.py                # config.toml 解析 & 校验
│   ├── models/                  # 数据模型 (Node, Edge, etc.)
│   ├── storage/                 # SQLite, Markdown I/O
│   ├── embedding/               # Embedding & Reranker engine
│   ├── retrieval/               # Hybrid search pipeline
│   ├── sync/                    # File watcher & sync loop
│   ├── server/                  # MCP server & REST API
│   ├── lifecycle/               # Decay, Janitor, Condensation
│   ├── security/                # Sanitization pipeline
│   └── utils/                   # Logging, helpers
├── tests/
├── .synapse/                    # 运行时数据目录 (gitignored)
│   ├── active/                  # Active markdown nodes
│   ├── .archive/                # Archived nodes
│   ├── .audit/                  # Transmission audit logs
│   └── .logs/                   # System logs
└── README.md
```

### 2. Configuration System (`config.toml`)
完整解析设计文档中定义的 `config.toml` 结构，包括：

```toml
[server]
host = "0.0.0.0"
port = 8765
cors_allowed_origins = ["*"]
auth_token = ""                  # Optional static auth token (§7.1.1)

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "ollama"               # ollama | remote_api | builtin
model = "bge-m3"                 # bge-m3 | jina-v3 | gte-qwen2 | gemma-300m
dimension = 1024                 # auto-detected from model
timeout_seconds = 30

[reranker]
provider = "remote_api"          # remote_api | ollama | builtin
model = "bge-reranker-v2-m3"
max_candidates = 9               # post-hop candidate pool size
timeout_seconds = 30

[providers.ollama]
base_url = "http://127.0.0.1:11434"
embedding_endpoint = "/api/embed"
rerank_endpoint = ""
auto_start = false               # Synapse does not start Ollama itself
auto_pull = false                # Synapse does not pull models itself

[providers.remote_api]
base_url = "https://api.example.com"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = "SYNAPSE_MODEL_API_KEY"
headers = {}
request_timeout_seconds = 30

[retrieval]
engine = "sqlite"                # sqlite | lancedb
rrf_k = 60                       # RRF constant
similarity_threshold = 0.80      # Conflict detection threshold
top_k = 3                        # Final context assembly count

[decay]
concept_factor = 0.90
decision_factor = 0.977
reference_factor = 0.992
concept_janitor_days = 7
decision_janitor_days = 30
reference_janitor_days = 90

[sanitization]
custom_patterns = []             # Additional regex patterns for redaction

[logging]
retention_days = 7
max_file_size_mb = 50
log_dir = "./.synapse/.logs"
```

- 使用 `tomllib` (Python 3.11+) 或 `tomli` 解析
- 用 Pydantic model 做类型校验和默认值
- 支持 `SYNAPSE_CONFIG_PATH` 环境变量指定配置路径
- 注意：模型推理进程不由 Synapse 启动。`provider = "ollama"` 表示连接一个**已经运行**的本地 Ollama；`provider = "remote_api"` 表示连接一个远端 URL/API 服务。

### 3. CLI Entry Point
使用 `click` 或 `typer` 框架搭建命令行骨架：

```bash
synapse serve              # 启动 MCP server + file watcher
synapse rebuild-index      # 重建 SQLite 索引
synapse condense           # 手动触发 condensation
synapse install --service  # 安装 OS daemon
synapse status             # 查看服务状态
synapse version            # 版本信息
```

此阶段只需注册命令名和 placeholder 逻辑，具体实现由各自的 TODO 完成。

### 4. Logging Infrastructure
- 结构化日志输出到 `.synapse/.logs/`
- 按设计文档 §9.3 分文件：`mcp-daemon.log`, `file-watcher.log`, `janitor.log`, `audit.log`
- 日志轮转：7 天保留，每文件最大 50MB
- 使用 Python `logging` + `RotatingFileHandler`

### 5. Core Abstract Interfaces
定义核心抽象接口（ABC 或 Protocol），供后续 TODO 实现：
- `EmbeddingEngine` — 嵌入计算接口
- `RerankerEngine` — 重排序接口
- `SearchEngine` — 检索接口
- `NodeStore` — 节点 CRUD 接口

---

## Dependencies
- None (this is the foundation)

## Blocks
- All other TODOs depend on this

## Acceptance Criteria
- [ ] `synapse serve` 可启动（即使 server 返回空响应）
- [ ] `config.toml` 解析并用 Pydantic 校验通过
- [ ] `.synapse/active/`, `.synapse/.archive/`, `.synapse/.logs/`, `.synapse/.audit/` 目录自动创建
- [ ] CLI 所有子命令可执行（placeholder）
- [ ] 日志系统初始化成功，能按文件名写入
- [ ] `pytest` 基础框架可运行
- [ ] 抽象接口定义完成
