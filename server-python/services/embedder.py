# 作者：yangkunpeng1
# 日期：2026-07-21
"""
BGE 文本嵌入。

- 文档块：原文编码
- 查询：加 BGE 检索指令前缀后再编码
- 默认 L2 归一化，便于余弦检索
"""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

# BGE 中文检索常用指令（只加在 query，不加在 document）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """懒加载，避免 import 时阻塞启动。"""
    global _model
    if _model is None:
        print(f"[EMBED] loading model {_MODEL_NAME} ...", flush=True)
        _model = SentenceTransformer(_MODEL_NAME)
        print("[EMBED] model ready", flush=True)
    return _model


def warmup_embedder() -> None:
    m = get_model()
    try:
        m.encode(["预热"], normalize_embeddings=True)
    except Exception:
        pass


def format_query_for_embed(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    if q.startswith(QUERY_INSTRUCTION):
        return q
    return QUERY_INSTRUCTION + q


def embed_texts(
    texts: Sequence[str],
    *,
    normalize: bool = True,
) -> list[list[float]]:
    """批量编码；空列表返回 []。"""
    items = list(texts)
    if not items:
        return []
    # 空串用占位，避免 encode 异常，并保持与 ids 对齐
    cleaned = [t if (t and str(t).strip()) else " " for t in items]
    emb = get_model().encode(cleaned, normalize_embeddings=normalize)
    return emb.tolist()


def embed_text(text: str, *, normalize: bool = True) -> list[float]:
    """单条文档/通用文本编码。"""
    return embed_texts([text], normalize=normalize)[0]


def embed_documents(
    texts: Sequence[str],
    *,
    normalize: bool = True,
) -> list[list[float]]:
    """入库用：文档块批量编码（不加查询指令）。"""
    return embed_texts(texts, normalize=normalize)


def embed_query(query: str, *, normalize: bool = True) -> list[float]:
    """检索用：查询加 BGE 指令后再编码。"""
    return embed_text(format_query_for_embed(query), normalize=normalize)
