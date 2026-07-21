# 作者：yangkunpeng1
# 日期：2026-07-21
"""临时诊断：打印某 doc_id 在 Chroma 中的切块。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.vector_store import collection, doc_chunk_count, get_doc_text

DOC_ID = sys.argv[1] if len(sys.argv) > 1 else "2c0f0e8d-cc1f-4849-8e04-055b0383dbe5"
OUT = Path(__file__).with_name("diag_chunks_out.txt")

lines: list[str] = []
lines.append(f"doc_id={DOC_ID}")
lines.append(f"count={doc_chunk_count(DOC_ID)}")
r = collection.get(
    where={"doc_id": {"$eq": DOC_ID}},
    include=["documents", "metadatas"],
)
ids = r.get("ids") or []
docs = r.get("documents") or []
metas = r.get("metadatas") or []
lines.append(f"ids={len(ids)}")
for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas)):
    meta = meta or {}
    text = doc or ""
    lines.append(f"--- chunk {i} id={cid}")
    lines.append(f"meta={meta}")
    lines.append(f"chars={len(text)}")
    lines.append(f"preview={text[:300]!r}")
    if "365" in text or "公转" in text or "地球" in text:
        lines.append("FLAG: contains 365/公转/地球")
lines.append("==== FULL get_doc_text ====")
lines.append(get_doc_text(DOC_ID, 0))
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
