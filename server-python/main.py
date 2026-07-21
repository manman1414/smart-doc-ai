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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(
        target=_warmup_models_background, daemon=True, name="model-warmup"
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
                        }
                        for j in range(len(batch_items))
                    ]

                    embeddings = await loop.run_in_executor(None, embed_documents, batch)

                    collection.add(
                        embeddings=embeddings,
                        documents=batch,
                        ids=batch_ids,
                        metadatas=batch_metadatas,
                    )
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

def sse_ask(req: AskRequest):
    print(f"[MAIN] sse_ask START question={req.question[:50]}... doc_id={req.doc_id}", flush=True)
    relevant_chunks = search_similar(req.question, doc_id=req.doc_id)
    if not relevant_chunks:
        if req.doc_id and doc_chunk_count(req.doc_id) > 0:
            msg = "未找到与问题足够相关的内容，请换种问法或换个角度描述。"
        else:
            msg = "未找到相关文档内容，请先上传文档。"
        yield f"data: {json.dumps({'error': msg})}\n\n"
        return

    context = format_chunks_for_prompt(relevant_chunks)
    pages = sorted(
        {
            int(c.get("page") or 0)
            for c in relevant_chunks
            if int(c.get("page") or 0) > 0
        }
    )
    is_txt = any(
        str(c.get("source_ext") or "").lower() == "txt" for c in relevant_chunks
    )
    has_pages = bool(pages) and not is_txt
    scores = [c.get("score") for c in relevant_chunks]
    print(
        f"[MAIN] sse_ask retrieved chunks={len(relevant_chunks)} pages={pages} "
        f"has_pages={has_pages} scores={scores} context_chars={len(context)}",
        flush=True,
    )

    if has_pages:
        system_content = (
            "你是一个文档问答助手。请根据提供的文档片段回答用户问题。"
            "每个片段前标有【第N页】。"
            "要求：1）每条信息只写一次，列表项禁止重复；"
            "2）若依据某页内容，在相关处用（第N页）标注来源，不要编造页码；"
            "3）回答简洁，直接给结论与必要说明。"
        )
        user_tail = "请基于以上文档简洁回答（勿重复），并标注页码来源："
    else:
        system_content = (
            "你是一个文档问答助手。请根据提供的文档片段回答用户问题。"
            "要求：1）每条信息只写一次，列表项禁止重复；"
            "2）不要标注页码（本文档无页码概念）；"
            "3）回答简洁，直接给结论与必要说明。"
        )
        user_tail = "请基于以上文档简洁回答（勿重复），不要标注页码："

    messages = [{"role": "system", "content": system_content}]
    for msg in req.history[-10:]:
        messages.append(
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"文档内容：\n{context}\n\n"
                f"用户问题：{req.question}\n\n"
                f"{user_tail}"
            ),
        }
    )

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=0.3, max_tokens=500, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'token': delta.content})}\n\n"
        sources = [{"page": p} for p in pages]
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'AI 回答生成失败：{str(e)}'})}\n\n"

@app.post("/ai/ask")
async def ask(req: AskRequest, request: Request):
    if await request.is_disconnected():
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': '请求已取消'})}\n\n"]),
            media_type="text/event-stream"
        )
    return StreamingResponse(
        sse_ask(req), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
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