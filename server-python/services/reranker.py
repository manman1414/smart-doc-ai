# 作者：yangkunpeng1
# 日期：2026-07-21
"""
交叉编码器重排（BGE-Reranker）。

对「问题 + 候选片段」打分，比单向量相似度更准；懒加载模型。
"""

from __future__ import annotations

import math
import os
import threading
from typing import Sequence

_MODEL_NAME_DEFAULT = "BAAI/bge-reranker-base"
_model = None
_load_lock = threading.Lock()
_loading = False


def _rerank_pool_max() -> int:
    """送进 CrossEncoder 的候选上限；过大在 CPU 上极易拖到十几秒。"""
    raw = os.environ.get("SMARTDOC_RERANK_POOL", "8")
    try:
        return max(3, min(30, int(raw)))
    except ValueError:
        return 8


def _rerank_text_max_chars() -> int:
    raw = os.environ.get("SMARTDOC_RERANK_TEXT_CHARS", "480")
    try:
        return max(120, min(2000, int(raw)))
    except ValueError:
        return 480


def rerank_enabled() -> bool:
    raw = (os.environ.get("SMARTDOC_RERANK") or "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def is_reranker_ready() -> bool:
    return _model is not None


def _model_name() -> str:
    return (os.environ.get("SMARTDOC_RERANK_MODEL") or _MODEL_NAME_DEFAULT).strip()


def _rerank_min_score() -> float:
    """sigmoid 后的下限；默认 0（只排序不滤空）。可用 SMARTDOC_RERANK_MIN_SCORE 覆盖。"""
    raw = os.environ.get("SMARTDOC_RERANK_MIN_SCORE", "0")
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.0


def get_reranker():
    """懒加载 CrossEncoder（进程内只一次；文件在 HF 缓存）。"""
    global _model, _loading
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        _loading = True
        try:
            name = _model_name()
            print(f"[RERANK] loading CrossEncoder {name} ...", flush=True)
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(name)
            try:
                _model.predict([["预热", "预热文本"]])
            except Exception:
                pass
            print("[RERANK] model ready", flush=True)
        finally:
            _loading = False
    return _model


def warmup_reranker() -> bool:
    if not rerank_enabled():
        print("[RERANK] warmup skipped (disabled)", flush=True)
        return False
    get_reranker()
    return True


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rerank_pairs(
    query: str,
    texts: Sequence[str],
    *,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[tuple[int, float]]:
    """
    对候选文本重排。

    返回 [(原下标, sigmoid分数), ...]，按分数降序；已按 min_score / top_k 截断。
    """
    q = (query or "").strip()
    cap = _rerank_text_max_chars()
    docs = [(((t or "").strip() or " ")[:cap]) for t in texts]
    if not q or not docs:
        return []

    model = get_reranker()
    pairs = [[q, d] for d in docs]
    raw_scores = model.predict(pairs)
    try:
        raw_list = list(raw_scores)
    except TypeError:
        raw_list = [float(raw_scores)]

    scored: list[tuple[int, float]] = []
    for i, s in enumerate(raw_list):
        scored.append((i, _sigmoid(float(s))))

    scored.sort(key=lambda x: x[1], reverse=True)

    thresh = _rerank_min_score() if min_score is None else float(min_score)
    filtered = [(i, sc) for i, sc in scored if sc >= thresh]
    # 门槛滤光时仍按分数取 top，禁止空结果
    if not filtered and scored:
        filtered = list(scored)
    if top_k is not None:
        filtered = filtered[: max(1, int(top_k))]
    return filtered


def rerank_chunks(
    query: str,
    chunks: list[dict],
    *,
    top_k: int,
    min_score: float | None = None,
) -> list[dict]:
    """
    对带 text 的 chunk 列表重排；写回 score（重排分），保留 page/kind 等字段。
    """
    if not chunks:
        return []
    texts = [str(c.get("text") or "") for c in chunks]
    ranked = rerank_pairs(query, texts, top_k=top_k, min_score=min_score)
    out: list[dict] = []
    for idx, score in ranked:
        item = dict(chunks[idx])
        item["score"] = round(float(score), 4)
        item["dense_score"] = chunks[idx].get("score")
        out.append(item)
    return out
