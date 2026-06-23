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
from services.chunker import chunk_text
from services.vector_store import collection, search_similar, delete_doc_vectors, get_doc_text
from services.embedder import embed_text

app = FastAPI(title="SmartDoc AI Core", version="0.1.0")

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

def read_file_content(file_path: str, original_name: str) -> str:
    if not file_path:
        return "[错误：文件路径为空]"
    ext = original_name.split('.')[-1].lower() if '.' in original_name else ''
    if ext == 'pdf':
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if not text.strip():
                return "[PDF解析错误：文档中没有可提取的文字]"
            return text
        except Exception as e:
            return f"[PDF解析错误: {str(e)}]"
    elif ext == 'txt':
        try:
            if not os.path.exists(file_path):
                return f"[TXT读取错误：文件不存在 {file_path}]"
            for encoding in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    if content.strip():
                        return content
                except UnicodeDecodeError:
                    continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if not content.strip():
                return "[TXT读取错误：文件内容为空]"
            return content
        except Exception as e:
            return f"[TXT读取错误: {str(e)}]"
    else:
        return f"[不支持的文件类型: .{ext}]"

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
            yield f"data: {json.dumps({'stage': 'reading', 'progress': 5})}\n\n"
            full_text = await loop.run_in_executor(None, read_file_content, req.filePath, req.originalName)

            if cancelled.is_set():
                yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                return
            if not full_text or full_text.startswith("["):
                yield f"data: {json.dumps({'stage': 'error', 'message': f'文件读取失败：{full_text}'})}\n\n"
                return

            yield f"data: {json.dumps({'stage': 'chunking', 'progress': 10})}\n\n"
            chunks = chunk_text(full_text)
            total = len(chunks)
            if not chunks:
                yield f"data: {json.dumps({'stage': 'error', 'message': '文本为空'})}\n\n"
                return

            BATCH_SIZE = 20
            for start in range(0, total, BATCH_SIZE):
                if cancelled.is_set():
                    if wrote_vectors:
                        await loop.run_in_executor(None, delete_doc_vectors, doc_id)
                    yield f"data: {json.dumps({'stage': 'error', 'message': '请求已取消'})}\n\n"
                    return

                batch = chunks[start:start + BATCH_SIZE]
                batch_ids = [f"{doc_id}_chunk_{i}" for i in range(start, start + len(batch))]
                batch_metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(start, start + len(batch))]

                embeddings = await loop.run_in_executor(None, embed_text, batch)

                collection.add(
                    embeddings=embeddings,
                    documents=batch,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                wrote_vectors = True

                done = min(start + len(batch), total)
                pct = 10 + int(85 * done / total)
                yield f"data: {json.dumps({'stage': 'embedding', 'progress': pct, 'done': done, 'total': total})}\n\n"

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
    relevant_chunks = search_similar(req.question, doc_id=req.doc_id, top_k=3)
    if not relevant_chunks:
        yield f"data: {json.dumps({'error': '未找到相关文档内容，请先上传文档。'})}\n\n"
        return

    context = "\n\n".join(relevant_chunks)
    messages = [{"role": "system", "content": "你是一个文档问答助手。请根据提供的文档内容回答用户问题。"}]
    for msg in req.history[-10:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": f"文档内容：\n{context}\n\n用户问题：{req.question}\n\n请基于以上文档内容回答："})

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=0.3, max_tokens=500, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'token': delta.content})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
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
    if not full_text or full_text.startswith("["):
        return {"summary": f"文件内容提取失败：{full_text}"}

    return await _generate_summary_from_text(full_text[:3000], request)


@app.post("/ai/summarize-doc")
async def summarize_doc(req: SummarizeDocRequest, request: Request):
    """LM 恢复后按 doc_id 从 Chroma 重试摘要（上传临时文件已删除）"""
    if await request.is_disconnected():
        return {"summary": "请求已取消"}
    if not req.doc_id:
        return {"summary": "文件内容提取失败：缺少 doc_id"}

    loop = asyncio.get_event_loop()
    full_text = await loop.run_in_executor(None, get_doc_text, req.doc_id, 8000)
    if not full_text:
        return {"summary": "文件内容提取失败：文档向量不存在或为空"}

    return await _generate_summary_from_text(full_text[:3000], request)


async def _generate_summary_from_text(prompt_text: str, request: Request) -> dict:
    prompt = f"请用中文简要总结以下文档的核心内容（200字以内）：\n---\n{prompt_text}\n---\n摘要："

    stop = threading.Event()

    async def watch_disconnect():
        while not stop.is_set():
            if await request.is_disconnected():
                stop.set()
                print("[MAIN] summarize CANCELLED (client disconnect)", flush=True)
                return
            await asyncio.sleep(0.1)

    def run_llm_stream():
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

    loop = asyncio.get_event_loop()
    watch_task = asyncio.create_task(watch_disconnect())
    llm_future = loop.run_in_executor(None, run_llm_stream)

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