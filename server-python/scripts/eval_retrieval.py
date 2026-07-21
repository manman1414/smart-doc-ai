# 作者：yangkunpeng1
# 日期：2026-07-21
"""
检索评测：召回率 / 精确率 / 重排对比（带逐步计算公式）

用法：
  python scripts/eval_retrieval.py scripts/eval_retrieval.example.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.vector_store import (  # noqa: E402
    doc_chunk_count,
    resolve_recall_k,
    resolve_top_k,
    search_similar,
)


def _hit_pages(chunks: list[dict], gold_pages: set[int]) -> set[int]:
    got = set()
    for c in chunks:
        try:
            p = int(c.get("page") or 0)
        except (TypeError, ValueError):
            p = 0
        if p in gold_pages:
            got.add(p)
    return got


def _hit_contains(chunks: list[dict], needles: list[str]) -> set[str]:
    found: set[str] = set()
    blob = "\n".join((c.get("text") or "") for c in chunks)
    for n in needles:
        n = (n or "").strip()
        if n and n in blob:
            found.add(n)
    return found


def _fmt_ratio(num: int, den: int) -> tuple[float, str]:
    if den <= 0:
        return 0.0, f"{num}/{den}=0 (分母为0)"
    val = num / den
    return val, f"{num}/{den}={val:.3f}"


def _as_bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "是"):
        return True
    if s in ("0", "false", "no", "n", "off", "否"):
        return False
    return default


def _set_rerank_env(enabled: bool) -> None:
    os.environ["SMARTDOC_RERANK"] = "1" if enabled else "0"


def _page_metrics(hits: list[dict], gold_pages: set[int]) -> dict:
    if not gold_pages:
        return {}
    hit = _hit_pages(hits, gold_pages)
    g = len(gold_pages)
    r, _ = _fmt_ratio(len(hit), g)
    p, _ = _fmt_ratio(len(hit), len(hits))
    return {
        "命中页": sorted(hit),
        "召回率": r,
        "精确率": p,
        "返回页码": [c.get("page") for c in hits],
        "返回分数": [c.get("score") for c in hits],
    }


def _search_pool_and_final(
    q: str, doc_id: str, top_k: int, recall_k: int
) -> tuple[list[dict], list[dict]]:
    final_hits = search_similar(q, doc_id=doc_id, top_k=top_k, recall_k=recall_k)
    old_min = os.environ.get("SMARTDOC_TOP_K_MIN")
    old_max = os.environ.get("SMARTDOC_TOP_K_MAX")
    try:
        os.environ["SMARTDOC_TOP_K_MIN"] = "1"
        os.environ["SMARTDOC_TOP_K_MAX"] = str(max(recall_k, 1))
        pool_hits = search_similar(
            q, doc_id=doc_id, top_k=recall_k, recall_k=recall_k
        )
    finally:
        if old_min is None:
            os.environ.pop("SMARTDOC_TOP_K_MIN", None)
        else:
            os.environ["SMARTDOC_TOP_K_MIN"] = old_min
        if old_max is None:
            os.environ.pop("SMARTDOC_TOP_K_MAX", None)
        else:
            os.environ["SMARTDOC_TOP_K_MAX"] = old_max
    return pool_hits, final_hits


def eval_case(
    case: dict,
    *,
    force_k: int | None,
    force_recall_k: int | None,
    default_rerank: bool,
    default_compare: bool,
) -> dict:
    q = (case.get("question") or case.get("问题") or "").strip()
    doc_id = (case.get("doc_id") or case.get("文档ID") or "").strip()
    note = (case.get("备注") or "").strip()
    gold_pages = {
        int(p)
        for p in (case.get("gold_pages") or case.get("金标页码") or [])
        if p is not None and str(p).strip() != ""
    }
    gold_contains = [
        str(x).strip()
        for x in (case.get("gold_contains") or case.get("金标原文") or [])
        if str(x).strip()
    ]
    use_rerank = _as_bool(
        case.get("启用重排") if "启用重排" in case else case.get("use_rerank"),
        default_rerank,
    )
    compare_rerank = _as_bool(
        case.get("对比重排") if "对比重排" in case else case.get("compare_rerank"),
        default_compare,
    )

    if not q or not doc_id:
        return {"ok": False, "error": "缺少 question/问题 或 doc_id/文档ID", "question": q}

    n = doc_chunk_count(doc_id)
    if n <= 0:
        return {"ok": False, "error": f"文档无向量 doc_id={doc_id}", "question": q}

    top_k = force_k if force_k is not None else resolve_top_k(doc_id)
    recall_k = (
        force_recall_k
        if force_recall_k is not None
        else resolve_recall_k(doc_id, max(top_k, 1))
    )

    calc_lines: list[str] = []
    metrics: dict = {}
    rerank_info: dict = {
        "启用": use_rerank,
        "对比": compare_rerank,
    }

    # ----- 主结果：按「启用重排」跑最终链路 -----
    _set_rerank_env(use_rerank)
    pool_hits, final_hits = _search_pool_and_final(q, doc_id, top_k, recall_k)

    if gold_pages:
        hit_final = _hit_pages(final_hits, gold_pages)
        hit_pool = _hit_pages(pool_hits, gold_pages)
        g = len(gold_pages)
        r_pool, r_pool_s = _fmt_ratio(len(hit_pool), g)
        r_final, r_final_s = _fmt_ratio(len(hit_final), g)
        p_final, p_final_s = _fmt_ratio(len(hit_final), len(final_hits))
        metrics["page_recall_at_pool"] = r_pool
        metrics["page_recall_at_final"] = r_final
        metrics["page_precision_at_final"] = p_final
        metrics["gold_pages"] = sorted(gold_pages)
        metrics["hit_pages_final"] = sorted(hit_final)
        metrics["hit_pages_pool"] = sorted(hit_pool)
        calc_lines.extend(
            [
                "【按页码计算】（当前设置：重排=" + ("开" if use_rerank else "关") + "）",
                f"  金标页码 = {sorted(gold_pages)}  （共 {g} 个）",
                f"  召回池命中页 = {sorted(hit_pool)}",
                f"  最终结果命中页 = {sorted(hit_final)}",
                f"  最终返回页码 = {[c.get('page') for c in final_hits]}  （共 {len(final_hits)} 段）",
                f"  召回率(召回池) = {r_pool_s}",
                f"  召回率(最终)   = {r_final_s}",
                f"  精确率(最终)   = {p_final_s}",
            ]
        )

    if gold_contains:
        hit_final_t = _hit_contains(final_hits, gold_contains)
        hit_pool_t = _hit_contains(pool_hits, gold_contains)
        g = len(gold_contains)
        r_pool, r_pool_s = _fmt_ratio(len(hit_pool_t), g)
        r_final, r_final_s = _fmt_ratio(len(hit_final_t), g)
        useful = sum(
            1
            for c in final_hits
            if any(n in (c.get("text") or "") for n in gold_contains)
        )
        p_final, p_final_s = _fmt_ratio(useful, len(final_hits))
        metrics["text_recall_at_pool"] = r_pool
        metrics["text_recall_at_final"] = r_final
        metrics["text_precision_at_final"] = p_final
        metrics["gold_contains"] = gold_contains
        metrics["hit_contains_final"] = sorted(hit_final_t)
        calc_lines.extend(
            [
                "【按原文关键词计算】",
                f"  金标原文 = {gold_contains}",
                f"  召回率(最终) = {r_final_s}",
                f"  精确率(最终) = {p_final_s}",
            ]
        )

    # ----- 重排对比：关 vs 开 -----
    if compare_rerank and gold_pages:
        _set_rerank_env(False)
        _, hits_off = _search_pool_and_final(q, doc_id, top_k, recall_k)
        _set_rerank_env(True)
        _, hits_on = _search_pool_and_final(q, doc_id, top_k, recall_k)
        # 恢复本题设置
        _set_rerank_env(use_rerank)

        m_off = _page_metrics(hits_off, gold_pages)
        m_on = _page_metrics(hits_on, gold_pages)
        rerank_info.update(
            {
                "未重排_页码": m_off.get("返回页码"),
                "重排后_页码": m_on.get("返回页码"),
                "未重排_分数": m_off.get("返回分数"),
                "重排后_分数": m_on.get("返回分数"),
                "未重排_命中页": m_off.get("命中页"),
                "重排后_命中页": m_on.get("命中页"),
                "未重排_召回率最终": m_off.get("召回率"),
                "重排后_召回率最终": m_on.get("召回率"),
                "未重排_精确率最终": m_off.get("精确率"),
                "重排后_精确率最终": m_on.get("精确率"),
            }
        )
        r0 = float(m_off.get("召回率") or 0)
        r1 = float(m_on.get("召回率") or 0)
        p0 = float(m_off.get("精确率") or 0)
        p1 = float(m_on.get("精确率") or 0)
        if r1 > r0 or (r1 == r0 and p1 > p0):
            improve = True
        elif r1 < r0 or (r1 == r0 and p1 < p0):
            improve = False
        else:
            improve = "持平"
        rerank_info["是否改善"] = improve

        calc_lines.extend(
            [
                "【重排对比】（不重排 vs 重排，同一题同一金标）",
                f"  未重排: 页码={m_off.get('返回页码')} 命中={m_off.get('命中页')} "
                f"召回率={m_off.get('召回率'):.3f} 精确率={m_off.get('精确率'):.3f}",
                f"  重排后: 页码={m_on.get('返回页码')} 命中={m_on.get('命中页')} "
                f"召回率={m_on.get('召回率'):.3f} 精确率={m_on.get('精确率'):.3f}",
                f"  是否改善 = {improve}  （先比召回率，再比精确率）",
            ]
        )
    elif compare_rerank and not gold_pages:
        calc_lines.append("【重排对比】跳过：需要金标页码才能对比")

    if not gold_pages and not gold_contains:
        return {
            "ok": False,
            "error": "需要 gold_pages/金标页码 或 gold_contains/金标原文",
            "question": q,
        }

    return {
        "ok": True,
        "备注": note,
        "question": q,
        "doc_id": doc_id,
        "doc_chunks": n,
        "top_k": top_k,
        "recall_k": recall_k,
        "final_n": len(final_hits),
        "pool_n": len(pool_hits),
        "final_pages": [c.get("page") for c in final_hits],
        "final_scores": [c.get("score") for c in final_hits],
        "启用重排": use_rerank,
        "对比重排": compare_rerank,
        "重排": rerank_info,
        "metrics": metrics,
        "计算过程": calc_lines,
    }


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索召回率/精确率/重排对比评测")
    parser.add_argument("cases", help="题集 JSON 路径")
    parser.add_argument("--k", type=int, default=None, help="强制最终条数")
    parser.add_argument("--recall-k", type=int, default=None, help="强制召回条数")
    args = parser.parse_args()

    path = Path(args.cases)
    if not path.is_file():
        print(f"找不到题集: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data.get("cases") or []
    if not cases:
        print("题集为空")
        return 1

    switches = data.get("评测开关") or {}
    default_rerank = _as_bool(switches.get("启用重排"), True)
    default_compare = _as_bool(switches.get("对比重排"), True)

    report: list[str] = [
        "=" * 60,
        "检索评测报告（召回率 / 精确率 / 重排对比）",
        f"题集文件: {path}",
        f"题目数量: {len(cases)}",
        f"默认启用重排: {default_rerank} | 默认对比重排: {default_compare}",
        "=" * 60,
    ]
    print("\n".join(report))

    rows = []
    for i, case in enumerate(cases, 1):
        block = [f"\n-------- 第 {i}/{len(cases)} 题 --------"]
        row = eval_case(
            case,
            force_k=args.k,
            force_recall_k=args.recall_k,
            default_rerank=default_rerank,
            default_compare=default_compare,
        )
        rows.append(row)
        if not row.get("ok"):
            block.append(f"跳过/错误: {row.get('error')} | {row.get('question')}")
            print("\n".join(block))
            report.extend(block)
            continue

        block.append(f"备注: {row.get('备注') or '（无）'}")
        block.append(f"问题: {row['question']}")
        block.append(
            f"启用重排={row.get('启用重排')} | 对比重排={row.get('对比重排')} | "
            f"文档块数={row['doc_chunks']} | recall_k={row['recall_k']} | top_k={row['top_k']}"
        )
        block.append(
            f"实际返回: {row['final_n']} 段, 页码={row['final_pages']}, 分数={row['final_scores']}"
        )
        block.extend(row.get("计算过程") or [])
        print("\n".join(block))
        report.extend(block)

    ok_rows = [r for r in rows if r.get("ok")]
    summary = [
        "\n" + "=" * 60,
        "汇总平均分",
        f"有效题数: {len(ok_rows)}/{len(rows)}",
    ]

    def collect(key: str) -> list[float]:
        return [
            float(r["metrics"][key])
            for r in ok_rows
            if key in r.get("metrics", {})
        ]

    name_map = {
        "page_recall_at_pool": "按页-召回率(召回池)",
        "page_recall_at_final": "按页-召回率(最终)",
        "page_precision_at_final": "按页-精确率(最终)",
        "text_recall_at_pool": "按原文-召回率(召回池)",
        "text_recall_at_final": "按原文-召回率(最终)",
        "text_precision_at_final": "按原文-精确率(最终)",
    }
    for key, cname in name_map.items():
        vals = collect(key)
        if vals:
            summary.append(f"  {cname} = {_avg(vals):.3f}  （{len(vals)} 题平均）")

    improve_stats = {"改善": 0, "变差": 0, "持平": 0}
    for r in ok_rows:
        v = (r.get("重排") or {}).get("是否改善")
        if v is True:
            improve_stats["改善"] += 1
        elif v is False:
            improve_stats["变差"] += 1
        elif v == "持平":
            improve_stats["持平"] += 1
    if any(improve_stats.values()):
        summary.append(
            f"  重排对比: 改善={improve_stats['改善']} 变差={improve_stats['变差']} "
            f"持平={improve_stats['持平']}"
        )

    summary.extend(
        [
            "",
            "怎么看：",
            "  召回率≈1 → 该找的都找到了",
            "  精确率≈1 → 返回段大多有用",
            "  重排是否改善=true → 加重排后最终结果更好",
            "=" * 60,
        ]
    )
    print("\n".join(summary))
    report.extend(summary)

    out_json = path.with_suffix(".results.json")
    out_report = path.with_name(path.stem + ".report.txt")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report.write_text("\n".join(report), encoding="utf-8")
    print(f"\n明细 JSON: {out_json}")
    print(f"中文报告: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
