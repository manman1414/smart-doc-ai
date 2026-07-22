# 作者：yangkunpeng1
# 日期：2026-07-23
"""
多轮对话溢出向量库：压缩出窗口的消息入库，供 search_chat 检索。
与文档 collection 分离，按 conversation_id 隔离。
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import chromadb

from .embedder import embed_documents, embed_query
from .vector_store import _CHROMA_PATH, _collection_space, _score_from_distance

_chat_client = chromadb.PersistentClient(path=_CHROMA_PATH)
chat_collection = _chat_client.get_or_create_collection(name="chat_turns")


def _chat_top_k() -> int:
    raw = os.environ.get("SMARTDOC_CHAT_TOP_K", "4")
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 4


def _stable_id(conversation_id: str, msg_index: int, role: str, content: str) -> str:
    digest = hashlib.sha1(
        f"{conversation_id}|{msg_index}|{role}|{content}".encode("utf-8")
    ).hexdigest()[:16]
    return f"chat:{conversation_id}:{msg_index}:{digest}"


def index_overflow_messages(
    conversation_id: str,
    messages: Sequence[dict],
    *,
    start_index: int = 0,
    doc_id: str = "",
) -> int:
    """
    将即将/已经压出窗口的消息写入对话向量库。
    start_index：这些消息在完整 history 中的起始下标（用于稳定 id）。
    返回成功写入条数。
    """
    cid = (conversation_id or "").strip()
    if not cid:
        return 0

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for i, m in enumerate(messages or []):
        role = str((m or {}).get("role") or "").strip()
        content = str((m or {}).get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        idx = int(start_index) + i
        label = "用户" if role == "user" else "助手"
        text = f"{label}：{content}"
        ids.append(_stable_id(cid, idx, role, content))
        docs.append(text)
        metas.append(
            {
                "conversation_id": cid,
                "doc_id": (doc_id or "").strip(),
                "role": role,
                "msg_index": idx,
                "kind": "chat",
            }
        )

    if not ids:
        return 0

    try:
        embeddings = embed_documents(docs)
        # upsert：同 id 覆盖，避免重复压缩时叠两份
        chat_collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )
        print(
            f"[CHAT_MEM] indexed n={len(ids)} conversation_id={cid[:12]}… "
            f"start_index={start_index}",
            flush=True,
        )
        return len(ids)
    except Exception as e:
        print(f"[CHAT_MEM] index failed: {e}", flush=True)
        return 0


def search_chat_turns(
    query: str,
    conversation_id: str,
    *,
    top_k: int | None = None,
) -> list[dict]:
    """按会话检索溢出对话。返回 [{text, score, role, msg_index, source}]。"""
    cid = (conversation_id or "").strip()
    q = (query or "").strip()
    if not cid or not q:
        return []

    k = _chat_top_k() if top_k is None else max(1, int(top_k))
    try:
        # 先看库里有多少，避免 n_results 过大报错
        existing = chat_collection.get(
            where={"conversation_id": {"$eq": cid}},
            include=[],
        )
        n = len(existing.get("ids") or [])
        if n <= 0:
            return []
        fetch_k = min(k, n)
        space = _collection_space()
        q_emb = embed_query(q)
        results = chat_collection.query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            where={"conversation_id": {"$eq": cid}},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[CHAT_MEM] search failed: {e}", flush=True)
        return []

    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    dists = (results.get("distances") or [[]])[0] or []

    out: list[dict] = []
    for i, doc in enumerate(docs):
        text = (doc or "").strip()
        if not text:
            continue
        dist = dists[i] if i < len(dists) else 0.0
        score = _score_from_distance(dist, space)
        meta = metas[i] if i < len(metas) else None
        meta = meta or {}
        try:
            msg_index = int(meta.get("msg_index") or 0)
        except (TypeError, ValueError):
            msg_index = 0
        out.append(
            {
                "text": text,
                "score": round(float(score), 4),
                "role": str(meta.get("role") or ""),
                "msg_index": msg_index,
                "source": "chat",
                "kind": "chat",
            }
        )
    out.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    return out


def delete_chat_vectors(conversation_id: str) -> None:
    cid = (conversation_id or "").strip()
    if not cid:
        return
    try:
        chat_collection.delete(where={"conversation_id": {"$eq": cid}})
        print(f"[CHAT_MEM] deleted conversation_id={cid[:12]}…", flush=True)
    except Exception as e:
        print(f"[CHAT_MEM] delete failed: {e}", flush=True)


def chat_turn_count(conversation_id: str) -> int:
    cid = (conversation_id or "").strip()
    if not cid:
        return 0
    try:
        result = chat_collection.get(
            where={"conversation_id": {"$eq": cid}},
            include=[],
        )
        return len(result.get("ids") or [])
    except Exception:
        return 0
