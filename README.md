# SmartDoc AI

智能文档问答系统：上传 PDF / TXT，自动解析、向量化入库，基于文档内容进行 AI 摘要与多轮问答。

作者：yangkunpeng1 · 2026-06-24

---

## 项目用途

SmartDoc AI 面向「**先读文档，再提问**」的场景，例如：

- 快速了解一份合同、报告、说明书的核心内容
- 针对文档细节进行多轮追问（RAG 检索增强生成）
- 保存历史会话，刷新页面后仍可恢复文档与对话

整体在本地运行：文档向量与对话历史保存在本机，大模型通过 **LM Studio** 本地推理，无需上传文档到公有云 API。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 文档上传 | 支持 PDF、TXT，单文件最大 20MB |
| 文档分析 | 自动分块、嵌入向量，写入 ChromaDB |
| AI 摘要 | 上传完成后生成文档摘要（依赖 LM Studio） |
| 智能问答 | 基于文档向量检索 + 本地大模型流式回答 |
| 历史会话 | SQLite 持久化，支持查看与删除 |
| 上传取消 | 离开页面或切换路由时中止后端处理 |
| 摘要重试 | LM Studio 恢复后可按文档重新生成摘要 |
| 存储对账 | 定时清理孤儿向量与过期临时文件 |

---

## 系统架构

```
浏览器 (Umi + React + Ant Design, :8001)
        │
        ▼
Node 网关 (Express, :3000)  ── 会话 / 上传 / 转发
        │
        ▼
Python AI (FastAPI, :8000)  ── 解析 / 向量 / RAG
        │
        ├── ChromaDB（向量库，data/chroma_db）
        └── LM Studio (:11435)  ── 摘要与问答
```

**数据分工**

- **SQLite**（`server-node/db/`）：会话元数据、消息记录
- **ChromaDB**（`data/chroma_db/`）：文档切块与向量
- **uploads/**：上传过程中的临时文件（处理完自动删除）

---

## 目录结构

```
smart-doc-ai/
├── web/              # 前端（Umi 4 + React + Ant Design）
├── server-node/      # Node 网关（Express + SQLite）
├── server-python/    # Python AI 服务（FastAPI + ChromaDB + BGE）
├── data/             # 运行时数据（Chroma，已在 .gitignore）
└── README.md
```

---

## 环境要求

- **Node.js** 18+（含原生 `fetch`）
- **Python** 3.10+
- **Yarn**
- **LM Studio**：开启 Local Server，加载模型（默认配置见 `server-python/main.py` 中的 `MODEL_NAME`）

---

## 快速开始

### 1. Python AI 服务

```bash
cd server-python
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Node 网关

```bash
cd server-node
cp .env.example .env
yarn install
yarn dev          # 开发模式，端口 3000
# 或 yarn build && yarn start
```

### 3. 前端

```bash
cd web
cp .env.example .env
yarn install
yarn dev          # 默认 http://localhost:8001
```

### 4. LM Studio

1. 加载对话模型（与 `main.py` 中 `MODEL_NAME` 一致）
2. 启动 Local Server，端口 **11435**

---

## 自动化测试

需先启动 Node（3000）与 Python（8000）服务：

```bash
cd server-node
yarn test         # 全流程 E2E（上传 → 问答 → 删会话，16 项）
yarn test:smoke   # 上传冒烟（12 项）
```

---

## 主要页面

| 路径 | 说明 |
|------|------|
| `/chat` | 上传文档、查看摘要、与文档对话 |
| `/history` | 历史会话列表与管理 |

---

## 配置说明

**server-node/.env**

```env
PORT=3000
PYTHON_AI_URL=http://localhost:8000
```

**web/.env**

```env
PORT=8001
```

开发环境下，前端 SSE 相关请求会直连 Node（`:3000`），避免 dev proxy 缓冲导致进度条不实时。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Umi 4、React、Ant Design |
| 网关 | Express、Multer、better-sqlite3 |
| AI 服务 | FastAPI、ChromaDB、sentence-transformers（BGE） |
| 大模型 | LM Studio（OpenAI 兼容 API） |

---

## 许可证

MIT
