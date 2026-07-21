# 作者：yangkunpeng1
# 日期：2026-07-21
"""
Chroma 向量库：检索 / 入库辅助。

检索：按文档块数自适应召回与最终条数（均有上下限）→ 阈值 → 去重 →（可选）重排。
"""

from __future__ import annotations

import os
import re

import chromadb

_CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "chroma_db",
)
client = chromadb.PersistentClient(path=_CHROMA_PATH)
collection = client.get_or_create_collection(name="documents")


def _top_k_min() -> int:
    raw = os.environ.get("SMARTDOC_TOP_K_MIN", "2")
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 2


def _top_k_max() -> int:
    """最终进 prompt 的上限（限制上下文长度）。"""
    raw = os.environ.get("SMARTDOC_TOP_K_MAX", "5")
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 5


def resolve_top_k(
    doc_id: str | None,
    top_k: int | None = None,
    candidate_n: int | None = None,
) -> int:
    """
    自适应最终条数（重排后保留几段），带上下限。

    - 显式 top_k：夹在 [MIN, MAX] 内
    - 否则按文档块数：小→少、大→多，但不超过 MAX
    - candidate_n：不超过实际候选数；候选不足 MIN 时允许更少
    """
    lo = _top_k_min()
    hi = max(lo, _top_k_max())

    if top_k is not None:
        target = int(top_k)
    else:
        n = doc_chunk_count(doc_id) if doc_id else 0
        if n <= 0:
            target = min(hi, max(lo, 3))
        elif n <= 12:
            target = lo
        elif n <= 60:
            target = min(hi, max(lo, 3))
        elif n <= 200:
            target = min(hi, max(lo, 4))
        else:
            target = hi

    target = max(lo, min(hi, target))

    if candidate_n is not None:
        if candidate_n <= 0:
            return 0
        # 候选不足下限时，有多少用多少
        return min(target, candidate_n)

    return target


def _recall_k_max() -> int:
    """自适应召回上限（原 SMARTDOC_RECALL_K，默认 50）。"""
    raw = os.environ.get("SMARTDOC_RECALL_K", "50")
    try:
        return max(1, min(100, int(raw)))
    except ValueError:
        return 50


def _recall_ratio() -> float:
    """按文档切块数比例召回。"""
    raw = os.environ.get("SMARTDOC_RECALL_RATIO", "0.12")
    try:
        return min(1.0, max(0.01, float(raw)))
    except ValueError:
        return 0.12


def resolve_recall_k(
    doc_id: str | None,
    final_k: int,
    recall_k: int | None = None,
) -> int:
    """
    按文档切块数自适应初召回条数。

    - 显式传入 recall_k：直接用（测试/调试）
    - 有 doc_id：min(上限, 块数, max(下限, top_k×5, 块数×比例))
    - 小文档：块数 ≤ 下限时直接召回全部
    - 无 doc_id：退化为 max(下限, top_k×5)，再夹上限
    """
    import math

    final_k = max(1, int(final_k))
    cap = _recall_k_max()
    floor = max(final_k * 3, 8)

    if recall_k is not None:
        return max(final_k, min(cap, int(recall_k)))

    if doc_id:
        n = doc_chunk_count(doc_id)
        if n <= 0:
            target = max(floor, final_k * 5)
            return min(cap, target)
        if n <= floor:
            return n
        by_ratio = int(math.ceil(n * _recall_ratio()))
        by_topk = final_k * 5
        target = max(floor, by_ratio, by_topk)
        return min(cap, n, target)

    return min(cap, max(floor, final_k * 5))


def _sim_threshold_default() -> float:
    """余弦相似度下限；默认放宽，避免库内有内容却问不到。"""
    raw = os.environ.get("SMARTDOC_SIM_THRESHOLD", "0.15")
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.15


def _small_doc_chunk_limit() -> int:
    """块数 ≤ 该值视为小文档，放宽检索阈值。"""
    raw = os.environ.get("SMARTDOC_SMALL_DOC_CHUNKS", "10")
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _sim_threshold_small() -> float:
    """小文档向量预过滤阈值（更松）。"""
    raw = os.environ.get("SMARTDOC_SIM_THRESHOLD_SMALL", "0.05")
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.05


def resolve_sim_threshold(
    doc_n: int,
    *,
    base: float | None = None,
    use_rerank: bool = False,
) -> float:
    """
    按文档规模选向量相似度阈值。
    小文档用更低阈值；有重排时预过滤也放宽。
    """
    thresh = _sim_threshold_default() if base is None else float(base)
    if use_rerank:
        thresh = min(thresh, 0.12)
    if doc_n > 0 and doc_n <= _small_doc_chunk_limit():
        thresh = min(thresh, _sim_threshold_small())
    return thresh


def _lexical_boost(query: str, text: str) -> float:
    """
    词面重合加分：题干/关键词若出现在切块里，大幅抬分。
    解决「短问 vs 整页试题」向量分偏低的问题。
    """
    from difflib import SequenceMatcher

    q = _norm_text(query)
    t = _norm_text(text)
    if not q or not t:
        return 0.0
    if q in t:
        return 0.55
    # 去掉问号等后再包含
    q2 = re.sub(r"[？?。！!，,、：:]+", "", q)
    if len(q2) >= 4 and q2 in t:
        return 0.5
    # 连续 4 字窗口命中
    hits = 0
    total = 0
    for i in range(0, max(0, len(q2) - 3)):
        total += 1
        if q2[i : i + 4] in t:
            hits += 1
    if total > 0 and hits / total >= 0.4:
        return 0.35
    ratio = SequenceMatcher(None, q2 or q, t).ratio()
    if ratio >= 0.25:
        return min(0.3, ratio * 0.5)
    return 0.0


def _collection_space() -> str:
    try:
        meta = getattr(collection, "metadata", None) or {}
        return str(meta.get("hnsw:space") or "l2").lower()
    except Exception:
        return "l2"


def _score_from_distance(distance: float, space: str) -> float:
    """
    把 Chroma 返回的 distance 转成越大越好的相似度分数（约 0~1）。
    - cosine：distance = 1 - cos_sim
    - l2：在单位向量上 cos_sim ≈ 1 - d²/2
    - ip：按 -distance 近似（归一化后接近 cos_sim）
    """
    d = float(distance)
    space = (space or "l2").lower()
    if space == "cosine":
        return max(0.0, min(1.0, 1.0 - d))
    if space == "ip":
        # Chroma 对 ip 常返回负内积作为 “distance”
        score = -d if d < 0 else d
        return max(0.0, min(1.0, float(score)))
    # l2（默认）
    d = max(0.0, d)
    return max(0.0, min(1.0, 1.0 - (d * d) / 2.0))


def _parse_page(meta: dict | None) -> int:
    meta = meta or {}
    try:
        return int(meta.get("page") or 0)
    except (TypeError, ValueError):
        return 0


def doc_chunk_count(doc_id: str) -> int:
    """某文档在库中的切块数（用于区分「没上传」和「不相关」）。"""
    if not doc_id:
        return 0
    try:
        result = collection.get(where={"doc_id": {"$eq": doc_id}}, include=[])
        return len(result.get("ids") or [])
    except Exception as e:
        print(f"[VECTOR] doc_chunk_count failed doc_id={doc_id}: {e}", flush=True)
        return 0


def _empty_query_result() -> dict:
    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def _query_with_fallback(kwargs: dict, *, doc_id: str | None, k: int) -> dict | None:
    """
    执行 Chroma query；若 n_results 大于命中数等原因失败，缩小 n_results 重试。
    """
    try:
        return collection.query(**kwargs)
    except Exception as e:
        print(f"[VECTOR] query failed doc_id={doc_id}: {e}", flush=True)
        if not doc_id:
            return None
        n = doc_chunk_count(doc_id)
        if n <= 0:
            return _empty_query_result()
        retry = dict(kwargs)
        retry["n_results"] = max(1, min(k, n))
        try:
            return collection.query(**retry)
        except Exception as e2:
            print(f"[VECTOR] query retry failed doc_id={doc_id}: {e2}", flush=True)
            return None


def search_similar(
    query: str,
    doc_id: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    recall_k: int | None = None,
) -> list[dict]:
    """
    向量检索 + 重排：

    1. 按文档块数自适应初召回（也可显式传 recall_k）
    2. 向量 score < 阈值丢弃
    3. 文本去重
    4. BGE CrossEncoder 重排（可用 SMARTDOC_RERANK=0 关闭）
    5. 自适应取最终 top_k（夹在 TOP_K_MIN~MAX）

    返回 [{"text", "page", "score", "kind", ...}, ...]；
    开启重排时 score 为重排分，dense_score 为向量分。
    """
    from .embedder import embed_query
    from .reranker import rerank_chunks, rerank_enabled

    q = (query or "").strip()
    if not q:
        return []

    # 先按文档规模估最终条数，再据此算召回；有候选后再按候选数收紧
    final_k = resolve_top_k(doc_id, top_k=top_k)
    fetch_k = resolve_recall_k(doc_id, max(final_k, 1), recall_k=recall_k)
    doc_n = doc_chunk_count(doc_id) if doc_id else 0

    base_thresh = _sim_threshold_default() if min_score is None else float(min_score)
    use_rerank = rerank_enabled()
    thresh = resolve_sim_threshold(doc_n, base=base_thresh, use_rerank=use_rerank)

    space = _collection_space()
    query_embedding = embed_query(q)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": fetch_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if doc_id:
        kwargs["where"] = {"doc_id": {"$eq": doc_id}}

    results = _query_with_fallback(kwargs, doc_id=doc_id, k=fetch_k)
    if results is None:
        return []

    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    dists = (results.get("distances") or [[]])[0] or []

    scored_all: list[dict] = []
    for i, doc in enumerate(docs):
        text = (doc or "").strip()
        if not text:
            continue
        dist = dists[i] if i < len(dists) else 0.0
        dense = _score_from_distance(dist, space)
        boost = _lexical_boost(q, text)
        score = min(1.0, dense + boost)
        meta = metas[i] if i < len(metas) else {}
        meta = meta or {}
        scored_all.append(
            {
                "text": text,
                "page": _parse_page(meta),
                "score": round(score, 4),
                "dense_score": round(dense, 4),
                "lexical_boost": round(boost, 4),
                "kind": str(meta.get("kind") or "text"),
                "source_ext": str(meta.get("source_ext") or ""),
            }
        )
    scored_all.sort(key=lambda x: x.get("score") or 0.0, reverse=True)

    if scored_all:
        top3 = [
            (
                c.get("page"),
                c.get("score"),
                c.get("dense_score"),
                c.get("lexical_boost"),
            )
            for c in scored_all[:3]
        ]
        print(f"[VECTOR] top scores (page,score,dense,boost)={top3}", flush=True)

    candidates = [c for c in scored_all if (c.get("score") or 0.0) >= thresh]
    # 只要 Chroma 有返回：阈值滤光时必须兜底，禁止对已入库文档返回空
    if not candidates and scored_all:
        keep_n = max(final_k, min(len(scored_all), fetch_k))
        candidates = scored_all[:keep_n]
        print(
            f"[VECTOR] thresh={thresh} wiped all; fallback keep top {len(candidates)}",
            flush=True,
        )
    # 指定 doc 且库非空，但 query 异常无返回时：拉全量再词面打分
    if not scored_all and doc_id and doc_n > 0:
        print("[VECTOR] chroma query empty; fallback get-all + lexical", flush=True)
        got = collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
        )
        gdocs = got.get("documents") or []
        gmetas = got.get("metadatas") or []
        for doc, meta in zip(gdocs, gmetas or []):
            text = (doc or "").strip()
            if not text:
                continue
            meta = meta or {}
            boost = _lexical_boost(q, text)
            scored_all.append(
                {
                    "text": text,
                    "page": _parse_page(meta),
                    "score": round(boost, 4),
                    "dense_score": 0.0,
                    "lexical_boost": round(boost, 4),
                    "kind": str(meta.get("kind") or "text"),
                    "source_ext": str(meta.get("source_ext") or ""),
                }
            )
        scored_all.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
        candidates = scored_all[: max(final_k, min(len(scored_all), fetch_k))]
    deduped = _dedupe_chunks(candidates)
    # 候选很少时不要硬要满额
    final_k = resolve_top_k(doc_id, top_k=top_k, candidate_n=len(deduped))
    if final_k <= 0:
        print(
            f"[VECTOR] search q={q[:40]!r} doc_id={doc_id or '-'} "
            f"doc_chunks={doc_n} recall_k={fetch_k} kept=0 (no candidates)",
            flush=True,
        )
        return []

    if use_rerank and deduped:
        try:
            from .reranker import is_reranker_ready

            if not is_reranker_ready():
                out = deduped[:final_k]
                stage = "dense_warmup"
                print("[VECTOR] reranker not ready yet, use dense this turn", flush=True)
            else:
                rerank_pool = deduped[: max(final_k, min(len(deduped), fetch_k))]
                # 重排只排序，默认不做硬门槛；避免滤空
                out = rerank_chunks(
                    q, rerank_pool, top_k=final_k, min_score=0.0
                )
                if not out and rerank_pool:
                    out = rerank_pool[:final_k]
                    stage = "rerank_fallback_dense"
                    print(
                        "[VECTOR] rerank empty; fallback to dense order",
                        flush=True,
                    )
                else:
                    stage = "rerank"
        except Exception as e:
            print(f"[VECTOR] rerank failed, fallback dense: {e}", flush=True)
            out = deduped[:final_k]
            stage = "dense_fallback"
    else:
        out = deduped[:final_k]
        stage = "dense"

    print(
        f"[VECTOR] search q={q[:40]!r} doc_id={doc_id or '-'} "
        f"doc_chunks={doc_n} recall_k={fetch_k} "
        f"recall={len(docs)} after_thresh={len(candidates)} "
        f"after_dedupe={len(deduped)} kept={len(out)} stage={stage} "
        f"top_k={final_k} thresh={thresh} space={space}",
        flush=True,
    )
    return out


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """把带页码的片段格式化成给大模型的上下文（先去重，避免模型复读）。

    TXT（source_ext=txt 或 page<=0）不加【第N页】。
    """
    unique = _dedupe_chunks(chunks)
    blocks: list[str] = []
    for c in unique:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        ext = str(c.get("source_ext") or "").lower().lstrip(".")
        page = c.get("page") or 0
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0
        if ext == "txt" or page <= 0:
            blocks.append(text)
        else:
            blocks.append(f"【第{page}页】\n{text}")
    return "\n\n".join(blocks)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _text_similarity(a: str, b: str) -> float:
    """0~1，含包含关系与 SequenceMatcher。"""
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 12 and shorter in longer:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _dedupe_chunks(chunks: list[dict], sim_thresh: float = 0.72) -> list[dict]:
    """
    去掉相同 / 包含 / 高度相似的片段（分块 overlap、相近检索易导致复读）。
    保序：按传入顺序（一般为相似度从高到低）。
    """
    kept: list[dict] = []
    for c in chunks or []:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        n = _norm_text(text)
        if not n:
            continue
        drop = False
        replace_at = -1
        for i, prev in enumerate(kept):
            pn = _norm_text(prev.get("text") or "")
            if not pn:
                continue
            sim = _text_similarity(n, pn)
            if sim >= sim_thresh:
                if len(n) > len(pn):
                    replace_at = i
                else:
                    drop = True
                break
        if drop:
            continue
        if replace_at >= 0:
            kept[replace_at] = c
        else:
            kept.append(c)
    return kept


def delete_doc_vectors(doc_id: str) -> None:
    """清理指定文档向量（取消/失败/同 id 重入前调用）。"""
    if not doc_id:
        return
    try:
        collection.delete(where={"doc_id": {"$eq": doc_id}})
    except Exception as e:
        print(f"[VECTOR] delete_doc_vectors failed doc_id={doc_id}: {e}", flush=True)


def get_doc_text(doc_id: str, max_chars: int = 8000) -> str:
    """按 chunk_index 拼接文档全文（供摘要重试）；尽量保留页码标记。"""
    result = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=["documents", "metadatas"],
    )
    if not result or not result.get("ids"):
        return ""

    metas = result.get("metadatas") or []
    docs = result.get("documents") or []
    pairs = []
    for meta, doc in zip(metas, docs):
        if not doc:
            continue
        meta = meta or {}
        idx = meta.get("chunk_index", 0)
        try:
            page = int(meta.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        pairs.append((idx, page, doc))
    pairs.sort(key=lambda x: x[0])

    # 任一 chunk 标明 txt，或全部 page<=0 → 不加页码标记
    is_txt = False
    for meta in metas:
        if meta and str(meta.get("source_ext") or "").lower().lstrip(".") == "txt":
            is_txt = True
            break

    parts: list[str] = []
    last_page = None
    for _, page, doc in pairs:
        if not is_txt and page and page != last_page:
            parts.append(f"【第{page}页】")
            last_page = page
        parts.append(doc)
    full = "\n".join(parts)
    return full[:max_chars] if max_chars else full
