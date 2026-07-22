import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import uuid
import sys
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from services.chunker import chunk_pages
from services.vector_store import (
    collection,
    search_similar,
    delete_doc_vectors,
    get_doc_text,
    format_chunks_for_prompt,
    doc_chunk_count,
)
from services.embedder import embed_documents, warmup_embedder
from services.parser import read_file_content, is_read_error, iter_parse_events
from services.summarizer import summarize_document
from services.conversation_memory import (
    merge_memory_summary,
    merge_memory_facts,
    rewrite_question,
    build_history_prompt_messages,
    history_keep_turns,
    history_keep_messages,
    normalize_history,
    trim_joined_blocks,
    final_doc_max_chars,
    final_chat_max_chars,
    retrieve_relevant_facts,
    retrieve_relevant_summary,
    estimate_messages_chars,
)
from services.react_agent import run_react
from services.chat_memory_store import (
    index_overflow_messages,
    search_chat_turns,
    delete_chat_vectors,
)


def _warmup_models_background() -> None:
    """启动后后台预热，避免第一次提问卡在加载/下载模型。"""
    flag = (os.environ.get("SMARTDOC_WARMUP") or "1").strip().lower()
    if flag in ("0", "false", "off", "no"):
        print("[WARMUP] skipped by SMARTDOC_WARMUP", flush=True)
        return
    try:
        print("[WARMUP] embedder ...", flush=True)
        warmup_embedder()
        from services.reranker import warmup_reranker

        print("[WARMUP] reranker ...", flush=True)
        warmup_reranker()
        print("[WARMUP] done", flush=True)
    except Exception as e:
        print(f"[WARMUP] failed: {e}", flush=True)


def _yunqi_extract_once_background() -> None:
    """仅当 SMARTDOC_YUNQI_EXTRACT=1 时：OCR 抽取云启 PDF（运维旁路，默认关闭）。"""
    from pathlib import Path

    flag = (os.environ.get("SMARTDOC_YUNQI_EXTRACT") or "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return

    scripts = Path(__file__).resolve().parent / "scripts"
    done = scripts / "_yunqi_extract_done.flag"
    if done.exists():
        print("[YUNQI] extract skipped (done flag exists)", flush=True)
        return
    try:
        print("[YUNQI] starting PDF extract/OCR ...", flush=True)
        import runpy

        runpy.run_path(str(scripts / "_extract_yunqi_pdf.py"), run_name="__main__")
        print("[YUNQI] extract finished", flush=True)
    except Exception as e:
        print(f"[YUNQI] extract failed: {e}", flush=True)
        try:
            (scripts / "_yunqi_extract_done.flag").write_text(
                f"FAILED: {e}\n", encoding="utf-8"
            )
        except Exception:
            pass


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(
        target=_warmup_models_background, daemon=True, name="model-warmup"
    ).start()
    # 默认不跑；需显式 SMARTDOC_YUNQI_EXTRACT=1
    if (os.environ.get("SMARTDOC_YUNQI_EXTRACT") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        threading.Thread(
            target=_yunqi_extract_once_background, daemon=True, name="yunqi-extract"
        ).start()
    yield


app = FastAPI(title="SmartDoc AI Core", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="not-needed",
    timeout=120.0,
)

MODEL_NAME = "123@q4_k_m"

class SummarizeRequest(BaseModel):
   filePath: str = ""
   originalName: str = ""
class SummarizeDocRequest(BaseModel):
    doc_id: str = ""
class ProcessRequest(BaseModel):
    filePath: str = ""
    originalName: str = ""
    doc_id: str = None
class AskRequest(BaseModel):
    question: str
    doc_id: str
    history: list = []
    memory_summary: str = ""
    # 已并入 memory_summary 的 history 条数（避免每轮重复压缩）
    memory_covered: int = 0
    # 硬事实清单（只追加不改写，与叙述摘要分离）
    memory_facts: str = ""
    # 会话 ID：溢出对话向量按此隔离
    conversation_id: str = ""

# ==================== 文档处理（SSE 流式 + 可取消） ====================

@app.post("/ai/process-document")
async def process_document(req: ProcessRequest, request: Request):
    """流式处理：立即返回 SSE headers，边处理边推送进度"""

    async def generate():
        print(f"[MAIN] process_document START file={req.originalName}", flush=True)

        cancelled = threading.Event()

        async def monitor():
            while not cancelled.is_set():
                if await request.is_disconnected():
                    cancelled.set()
                    print(f"[MAIN] process_document CANCELLED (client disconnect) file={req.originalName}", flush=True)
                    break
                await asyncio.sleep(0.1)

        monitor_task = asyncio.create_task(monitor())
        loop = asyncio.get_event_loop()
        doc_id = req.doc_id or str(uuid.uuid4())
        wrote_vectors = False

        try:
            # 解析阶段：子线程按页推送 progress，主协程转 SSE
            parse_q: queue.Queue = queue.Queue()

            def run_parse():
                try:
                    for event in iter_parse_events(
                        req.filePath,
                        req.originalName,
                        cancel_check=cancelled.is_set,
                    ):
                        parse_q.put(event)
                        if event.get("type") in ("error", "result"):
                            break
                except Exception as e:
                    parse_q.put({"type": "error", "message": f"[PDF解析错误: {e}]"})
                finally:
                    parse_q.put(None)

            parse_future = loop.run_in_executor(None, run_parse)
            full_text = ""
            pages: list = []
            while True:
                item = await loop.run_in_executor(None, parse_q.get)
                if item is None:
                    break
                et = item.get("type")
                if et == "progress":
                    yield f"data: {json.dumps({k: item[k] for k in ('stage', 'progress', 'page', 'total', 'mode', 'message', 'image_coverage', 'ocr', 'strategy') if k in item})}\n\n"
                elif et == "error":
                    msg = item.get("message") or "解析失败"
                    if cancelled.is_set() or "取消" in str(msg):
                        yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'stage': 'error', 'message': f'文件读取失败：{msg}'})}\n\n"
                    await parse_future
                    return
                elif et == "result":
                    full_text = item.get("text") or ""
                    pages = item.get("pages") or []

            await parse_future

            if cancelled.is_set():
                yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                return
            if is_read_error(full_text):
                yield f"data: {json.dumps({'stage': 'error', 'message': f'文件读取失败：{full_text}'})}\n\n"
                return

            # TXT / 纯文字页 → recursive；有表图的页 → structure（按页选型）
            src_ext = ""
            if "." in (req.originalName or ""):
                src_ext = req.originalName.rsplit(".", 1)[-1].lower()

            if not pages and full_text.strip():
                # TXT 无页码；PDF 等缺 pages 时兜底为第 1 页
                pages = [
                    {
                        "page": 0 if src_ext == "txt" else 1,
                        "text": full_text,
                    }
                ]

            yield f"data: {json.dumps({'stage': 'chunking', 'progress': 10})}\n\n"
            chunk_items = chunk_pages(pages, source_ext=src_ext)
            total = len(chunk_items)
            kind_stat: dict[str, int] = {}
            strategy_stat: dict[str, int] = {}
            for c in chunk_items:
                k = str(c.get("kind") or "text")
                kind_stat[k] = kind_stat.get(k, 0) + 1
                s = str(c.get("strategy") or "")
                if s:
                    strategy_stat[s] = strategy_stat.get(s, 0) + 1
            print(
                f"[MAIN] chunking done chars={len(full_text)} pages={len(pages)} "
                f"chunks={total} strategies={strategy_stat} kinds={kind_stat}",
                flush=True,
            )
            if not chunk_items:
                yield f"data: {json.dumps({'stage': 'error', 'message': '文本为空'})}\n\n"
                return

            # 同 doc_id 先清再建，避免重复上传 / 重试残留
            await loop.run_in_executor(None, delete_doc_vectors, doc_id)

            BATCH_SIZE = 20
            try:
                for start in range(0, total, BATCH_SIZE):
                    if cancelled.is_set():
                        if wrote_vectors:
                            await loop.run_in_executor(None, delete_doc_vectors, doc_id)
                        yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                        return

                    batch_items = chunk_items[start:start + BATCH_SIZE]
                    batch = [c["text"] for c in batch_items]
                    batch_ids = [f"{doc_id}_chunk_{i}" for i in range(start, start + len(batch))]
                    batch_metadatas = [
                        {
                            "doc_id": doc_id,
                            "chunk_index": start + j,
                            # TXT：强制 page=0，不在提示里标「第 N 页」
                            "page": (
                                0
                                if src_ext == "txt"
                                else int(batch_items[j].get("page") or 1)
                            ),
                            "kind": str(batch_items[j].get("kind") or "text"),
                            "source_ext": src_ext or "",
                            # 同图多块关联；无则空串（Chroma metadata 不宜 None）
                            "figure_id": str(batch_items[j].get("figure_id") or ""),
                        }
                        for j in range(len(batch_items))
                    ]

                    embeddings = await loop.run_in_executor(None, embed_documents, batch)

                    def _add_batch(
                        emb=embeddings,
                        docs=batch,
                        ids=batch_ids,
                        metas=batch_metadatas,
                    ):
                        collection.add(
                            embeddings=emb,
                            documents=docs,
                            ids=ids,
                            metadatas=metas,
                        )

                    await loop.run_in_executor(None, _add_batch)
                    wrote_vectors = True

                    done = min(start + len(batch), total)
                    pct = 10 + int(85 * done / total)
                    yield f"data: {json.dumps({'stage': 'embedding', 'progress': pct, 'done': done, 'total': total})}\n\n"
                    await asyncio.sleep(0)
            except Exception as e:
                print(f"[MAIN] embed/store FAILED doc_id={doc_id}: {e}", flush=True)
                if wrote_vectors:
                    await loop.run_in_executor(None, delete_doc_vectors, doc_id)
                yield f"data: {json.dumps({'stage': 'error', 'message': f'向量化失败：{e}'})}\n\n"
                return

            if cancelled.is_set():
                if wrote_vectors:
                    await loop.run_in_executor(None, delete_doc_vectors, doc_id)
                yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                return

            print(f"[MAIN] process_document DONE doc_id={doc_id} chunks={total}", flush=True)
            yield f"data: {json.dumps({'stage': 'done', 'doc_id': doc_id, 'chunk_count': total})}\n\n"

        finally:
            monitor_task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ==================== RAG 问答 ====================

def _llm_chat_text(messages: list, max_tokens: int = 300) -> str:
    """同步短调用（摘要合并 / 事实抽取 / 提问改写）。"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def _memory_sse_fields(memory_summary: str, memory_covered: int, memory_facts: str) -> dict:
    return {
        "memory_summary": memory_summary,
        "memory_covered": memory_covered,
        "memory_facts": memory_facts,
    }


def sse_ask(req: AskRequest, cancel_check=None):
    """同步生成 SSE 帧；cancel_check() 为 True 时尽早停止（客户端断开）。"""

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    t0 = time.perf_counter()

    def _lap(label: str) -> None:
        print(
            f"[MAIN] sse_ask timing {label} +{(time.perf_counter() - t0) * 1000:.0f}ms",
            flush=True,
        )

    print(
        f"[MAIN] sse_ask START question={req.question[:50]}... doc_id={req.doc_id} "
        f"history={len(req.history or [])} keep_turns={history_keep_turns()}",
        flush=True,
    )
    if _cancelled():
        yield f"data: {json.dumps({'error': '请求已取消'})}\n\n"
        return

    hist = normalize_history(req.history)
    keep = history_keep_messages()
    memory_summary = (req.memory_summary or "").strip()
    memory_facts = (req.memory_facts or "").strip()
    try:
        covered = max(0, int(req.memory_covered or 0))
    except (TypeError, ValueError):
        covered = 0

    if len(hist) <= keep:
        recent = hist
        memory_covered = min(covered, 0)
    else:
        recent = hist[-keep:]
        need_cover = len(hist) - keep
        covered = min(covered, need_cover)
        new_older = hist[covered:need_cover]
        if new_older:
            if _cancelled():
                yield f"data: {json.dumps({'error': '请求已取消', **_memory_sse_fields(memory_summary, covered, memory_facts)})}\n\n"
                return
            yield f"data: {json.dumps({'stage': 'memory', 'message': '压缩记忆中…'}, ensure_ascii=False)}\n\n"
            conv_id = (req.conversation_id or "").strip()

            def _do_index():
                if not conv_id:
                    return 0
                return index_overflow_messages(
                    conv_id,
                    new_older,
                    start_index=covered,
                    doc_id=req.doc_id or "",
                )

            # 事实抽取 / 摘要压缩 / 旧对话入库 并行，缩短答前等待
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_idx = pool.submit(_do_index)
                fut_facts = pool.submit(
                    merge_memory_facts, memory_facts, new_older, _llm_chat_text
                )
                fut_sum = pool.submit(
                    merge_memory_summary, memory_summary, new_older, _llm_chat_text
                )
                try:
                    fut_idx.result()
                except Exception as e:
                    print(f"[MAIN] index overflow chat failed: {e}", flush=True)
                memory_facts = fut_facts.result()
                memory_summary = fut_sum.result()
            _lap("memory_compress")
            print(
                f"[MAIN] sse_ask memory +{len(new_older)} msgs "
                f"summary_chars={len(memory_summary)} facts_chars={len(memory_facts)}",
                flush=True,
            )
        memory_covered = need_cover

    mem = _memory_sse_fields(memory_summary, memory_covered, memory_facts)

    if _cancelled():
        yield f"data: {json.dumps({'error': '请求已取消', **mem})}\n\n"
        return

    def _rewrite(q: str) -> str:
        return rewrite_question(
            q,
            memory_summary=memory_summary,
            memory_facts=memory_facts,
            recent_messages=recent,
            chat_fn=_llm_chat_text,
        )

    def _search(query: str) -> list:
        return search_similar(query, doc_id=req.doc_id) or []

    conv_id = (req.conversation_id or "").strip()

    def _search_chat(query: str) -> list:
        if not conv_id:
            return []
        return search_chat_turns(query, conv_id) or []

    yield f"data: {json.dumps({'stage': 'react', 'message': '检索中…'}, ensure_ascii=False)}\n\n"
    react = run_react(
        req.question,
        memory_summary=memory_summary,
        memory_facts=memory_facts,
        recent_messages=recent,
        chat_fn=_llm_chat_text,
        search_fn=_search,
        search_chat_fn=_search_chat if conv_id else None,
        rewrite_fn=_rewrite,
        cancel_check=_cancelled,
    )
    _lap(f"react_{react.finish_reason}")
    for t in react.trace:
        yield f"data: {json.dumps({'step': t.get('step'), 'action': t.get('action'), 'thought': (t.get('thought') or '')[:200]}, ensure_ascii=False)}\n\n"

    if _cancelled() or react.finish_reason == "cancelled":
        yield f"data: {json.dumps({'error': '请求已取消', **mem})}\n\n"
        return

    relevant_chunks = react.chunks
    chat_chunks = react.chat_chunks
    if react.search_queries:
        print(f"[MAIN] sse_ask react doc_queries={react.search_queries}", flush=True)
    if react.chat_queries:
        print(f"[MAIN] sse_ask react chat_queries={react.chat_queries}", flush=True)

    if not relevant_chunks and not react.used_memory and not chat_chunks:
        if req.doc_id and doc_chunk_count(req.doc_id) > 0:
            msg = "未找到与问题足够相关的内容，请换种问法或换个角度描述。"
        else:
            msg = "未找到相关文档内容，请先上传文档。"
        yield f"data: {json.dumps({'error': msg, **mem})}\n\n"
        return

    if not relevant_chunks:
        # 纯对话回顾 / 旧对话检索
        context = ""
        context_numbered = ""
        pages: list[int] = []
        print(
            f"[MAIN] sse_ask react memory/chat-only reason={react.finish_reason} "
            f"chat_hits={len(chat_chunks)}",
            flush=True,
        )
    else:
        context = format_chunks_for_prompt(relevant_chunks, dedupe=True)
        try:
            previews = []
            for i, c in enumerate(relevant_chunks):
                t = (c.get("text") or "").strip().replace("\n", " ")
                previews.append(f"#{i} p={c.get('page')} len={len(t)} {t[:40]!r}")
            print(f"[MAIN] sse_ask context_chunks={previews}", flush=True)
        except Exception:
            pass
        pages = sorted(
            {
                int(c.get("page") or 0)
                for c in relevant_chunks
                if int(c.get("page") or 0) > 0
            }
        )
        scores = [c.get("score") for c in relevant_chunks]
        print(
            f"[MAIN] sse_ask retrieved chunks={len(relevant_chunks)} pages={pages} "
            f"scores={scores} context_chars={len(context)} react={react.finish_reason}",
            flush=True,
        )
        ctx_blocks: list[str] = []
        for i, c in enumerate(relevant_chunks, 1):
            t = (c.get("text") or "").strip()
            if not t:
                continue
            ctx_blocks.append(f"【片段{i}】\n{t}")
        context_numbered = "\n\n".join(ctx_blocks) if ctx_blocks else context

    chat_block = ""
    if chat_chunks:
        lines = []
        for i, c in enumerate(chat_chunks, 1):
            t = (c.get("text") or "").strip()
            if t:
                lines.append(f"【旧对话{i}】\n{t}")
        if lines:
            chat_block = "\n\n".join(lines)

    # 最终回答：按主流规模裁剪（存储侧摘要/事实仍可更长）
    with_doc = bool(relevant_chunks)
    if with_doc and context_numbered:
        context_numbered = trim_joined_blocks(context_numbered, final_doc_max_chars())
    if chat_block:
        chat_block = trim_joined_blocks(
            chat_block, final_chat_max_chars(with_doc=with_doc)
        )

    # 最终回答前：按问题检索相关事实/摘要（非整表粘贴）
    retrieved_facts = retrieve_relevant_facts(req.question, memory_facts)
    retrieved_summary = retrieve_relevant_summary(req.question, memory_summary)
    _lap("memory_retrieve")
    print(
        f"[MAIN] sse_ask memory_retrieve "
        f"facts_chars={len(retrieved_facts)}/{len(memory_facts or '')} "
        f"summary_chars={len(retrieved_summary)}/{len(memory_summary or '')}",
        flush=True,
    )

    if with_doc:
        system_content = (
            "你是文档问答助手。依据【文档片段】作答，可参考对话记忆。"
            "规则：完整覆盖要点；禁止复读；同事实合并；勿标页码；"
            "与文档冲突时以文档为准。"
        )
        extra_chat = f"\n\n【相关历史对话】\n{chat_block}\n" if chat_block else ""
        user_content = (
            f"【文档片段】\n{context_numbered}"
            f"{extra_chat}\n"
            f"【用户问题】\n{req.question}\n\n"
            "请给出完整答案（每点只写一次，勿标页码）："
        )
        messages = [{"role": "system", "content": system_content}]
        messages.extend(
            build_history_prompt_messages(
                retrieved_summary,
                recent,
                memory_facts=retrieved_facts,
                for_final=True,
                question="",  # 已检索过，避免二次检索
            )
        )
        messages.append({"role": "user", "content": user_content})
    else:
        system_content = (
            "你是对话助手。依据记忆与旧对话回答；勿编造；不确定就说明。"
        )
        mem_parts = []
        if retrieved_facts:
            mem_parts.append(f"【相关硬事实】\n{retrieved_facts}")
        if retrieved_summary:
            mem_parts.append(f"【相关摘要】\n{retrieved_summary}")
        mem_part = "\n\n".join(mem_parts) if mem_parts else "（无相关摘要/事实）"
        chat_part = chat_block or "（无旧对话命中）"
        user_content = (
            f"{mem_part}\n\n"
            f"【相关历史对话】\n{chat_part}\n\n"
            f"【用户问题】\n{req.question}\n\n"
            "请根据上述材料回答："
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    prompt_chars = estimate_messages_chars(messages)
    print(
        f"[MAIN] sse_ask final_prompt_chars={prompt_chars} "
        f"with_doc={with_doc} doc_cap={final_doc_max_chars()} "
        f"chat_cap={final_chat_max_chars(with_doc=with_doc)}",
        flush=True,
    )
    yield f"data: {json.dumps({'stage': 'answer', 'message': '生成回答中…'}, ensure_ascii=False)}\n\n"
    _lap("before_stream")
    try:
        if _cancelled():
            yield f"data: {json.dumps({'error': '请求已取消', **mem})}\n\n"
            return
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
                frequency_penalty=0.45,
                presence_penalty=0.15,
                stream=True,
            )
        except Exception as e_pen:
            print(f"[MAIN] sse_ask penalty unsupported, fallback: {e_pen}", flush=True)
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
                stream=True,
            )
        # 真·SSE：模型出 token 立刻下发（不做收齐后再推）
        for chunk in stream:
            if _cancelled():
                print("[MAIN] sse_ask cancelled mid-stream", flush=True)
                break
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'token': delta.content})}\n\n"
        if _cancelled():
            yield f"data: {json.dumps({'error': '请求已取消', **mem})}\n\n"
            return
        sources = [{"page": p} for p in pages]
        _lap("stream_done")
        yield f"data: {json.dumps({'done': True, 'sources': sources, **mem})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'AI 回答生成失败：{str(e)}', **mem})}\n\n"

@app.post("/ai/ask")
async def ask(req: AskRequest, request: Request):
    """把同步 sse_ask 放到后台线程，避免阻塞事件循环；客户端断开可取消。"""
    if await request.is_disconnected():
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': '请求已取消'})}\n\n"]),
            media_type="text/event-stream",
        )

    async def generate():
        loop = asyncio.get_event_loop()
        out_q: queue.Queue = queue.Queue(maxsize=256)
        cancel = threading.Event()

        def worker():
            try:
                for frame in sse_ask(req, cancel_check=cancel.is_set):
                    if cancel.is_set():
                        break
                    out_q.put(frame)
            except Exception as e:
                out_q.put(
                    f"data: {json.dumps({'error': f'AI 回答生成失败：{str(e)}'})}\n\n"
                )
            finally:
                out_q.put(None)

        threading.Thread(target=worker, daemon=True, name="sse-ask").start()

        async def watch_disconnect():
            while not cancel.is_set():
                if await request.is_disconnected():
                    cancel.set()
                    print("[MAIN] ask CANCELLED (client disconnect)", flush=True)
                    return
                await asyncio.sleep(0.15)

        watch_task = asyncio.create_task(watch_disconnect())
        try:
            while True:
                item = await loop.run_in_executor(None, out_q.get)
                if item is None:
                    break
                yield item
                if cancel.is_set():
                    break
        finally:
            cancel.set()
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ==================== AI 摘要 ====================

def _format_llm_error(exc: Exception) -> str:
    """将 LM Studio / OpenAI SDK 异常转为用户可读文案，避免暴露 Error code: 502 等"""
    msg = str(exc).lower()
    if '502' in msg or 'bad gateway' in msg:
        return 'AI 摘要生成失败：LM Studio 服务不可用，请启动 Local Server 并加载模型'
    if '503' in msg or 'service unavailable' in msg:
        return 'AI 摘要生成失败：LM Studio 服务繁忙或未就绪，请稍后重试'
    if '404' in msg or 'not found' in msg:
        return 'AI 摘要生成失败：模型不存在，请检查 LM Studio 中是否已加载对应模型'
    if 'connection' in msg or 'refused' in msg or 'connect' in msg:
        return 'AI 摘要生成失败：无法连接 LM Studio，请确认 Local Server 已启动'
    if 'timeout' in msg or 'timed out' in msg:
        return 'AI 摘要生成超时'
    if 'error code' in msg:
        return 'AI 摘要生成失败：LM Studio 调用异常，请检查服务与模型是否就绪'
    return 'AI 摘要生成失败：请检查 LM Studio 是否已启动并加载模型'

@app.post("/ai/summarize")
async def summarize(req: SummarizeRequest, request: Request):
    if await request.is_disconnected():
        return {"summary": "请求已取消"}

    loop = asyncio.get_event_loop()
    full_text = await loop.run_in_executor(None, read_file_content, req.filePath, req.originalName)
    if is_read_error(full_text):
        return {"summary": f"文件内容提取失败：{full_text}"}

    return await _generate_summary_from_text(full_text, request)


@app.post("/ai/summarize-doc")
async def summarize_doc(req: SummarizeDocRequest, request: Request):
    """按 doc_id 从 Chroma 取全文做分段摘要（上传完成后 / 重试摘要）。"""
    if await request.is_disconnected():
        return {"summary": "请求已取消"}
    if not req.doc_id:
        return {"summary": "文件内容提取失败：缺少 doc_id"}

    loop = asyncio.get_event_loop()
    # max_chars=0：取全文，由 summarizer 分段控制调用次数
    full_text = await loop.run_in_executor(None, get_doc_text, req.doc_id, 0)
    if not full_text:
        return {"summary": "文件内容提取失败：文档向量不存在或为空"}

    return await _generate_summary_from_text(full_text, request)


async def _generate_summary_from_text(prompt_text: str, request: Request) -> dict:
    stop = threading.Event()

    async def watch_disconnect():
        while not stop.is_set():
            if await request.is_disconnected():
                stop.set()
                print("[MAIN] summarize CANCELLED (client disconnect)", flush=True)
                return
            await asyncio.sleep(0.1)

    def llm_complete(prompt: str) -> str:
        if stop.is_set():
            return ""
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            if stop.is_set():
                break
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
        return "".join(parts).strip()

    def run_summary() -> str:
        return summarize_document(
            prompt_text,
            llm_complete=llm_complete,
            cancel_check=stop.is_set,
        )

    loop = asyncio.get_event_loop()
    watch_task = asyncio.create_task(watch_disconnect())
    llm_future = loop.run_in_executor(None, run_summary)

    try:
        done, _ = await asyncio.wait(
            {watch_task, llm_future},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop.is_set() or await request.is_disconnected():
            print("[MAIN] summarize STOPPED (llm stream)", flush=True)
            return {"summary": "请求已取消"}

        if llm_future in done:
            try:
                summary = llm_future.result()
            except Exception as e:
                return {"summary": _format_llm_error(e)}
            if not summary:
                return {"summary": "请求已取消"}
            return {"summary": summary}

        return {"summary": "请求已取消"}
    except asyncio.CancelledError:
        stop.set()
        print("[MAIN] summarize CANCELLED (handler cancelled)", flush=True)
        return {"summary": "请求已取消"}
    except Exception as e:
        return {"summary": _format_llm_error(e)}
    finally:
        stop.set()
        watch_task.cancel()


@app.delete("/ai/doc/{doc_id}")
async def delete_doc(doc_id: str):
    """删除指定 doc_id 的全部向量（对账 / 删会话时调用）"""
    delete_doc_vectors(doc_id)
    return {"ok": True, "doc_id": doc_id}


@app.delete("/ai/chat-memory/{conversation_id}")
async def delete_chat_memory(conversation_id: str):
    """删除指定会话的溢出对话向量"""
    delete_chat_vectors(conversation_id)
    return {"ok": True, "conversation_id": conversation_id}


@app.get("/ai/doc-ids")
async def list_doc_ids():
    """列出 Chroma 中所有 doc_id（供 Node 对账）"""
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas") or []
    doc_ids: set[str] = set()
    for meta in metas:
        if meta and meta.get("doc_id"):
            doc_ids.add(str(meta["doc_id"]))
    return {"doc_ids": sorted(doc_ids)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)