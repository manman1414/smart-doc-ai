import { Router, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import { registerDoc, createConversation, linkDocToConversation } from '../db';
import { sendJsonError, toClientError } from '../utils/clientError';

const router = Router();

const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 20 * 1024 * 1024 + 1},
  fileFilter: (req, file, cb) => {
    const allowed = ['application/pdf', 'text/plain'];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error('仅支持 PDF 和 TXT 文件'));
  }
});

const taskStore = new Map<string, { filePath: string; originalName: string; fileSize: number; }>();
/** 上传成功即注册，确保 SSE 连接前也能 cancel */
const activeTaskAcs = new Map<string, AbortController>();

/** 对账清理：进行中的 upload 临时文件路径（勿删） */
export function getActiveUploadPaths(): Set<string> {
  return new Set(Array.from(taskStore.values()).map((t) => path.resolve(t.filePath)));
}

function abortTask(taskId: string, reason: string): boolean {
  console.log(`[PROCESS] cancel 请求 taskId=${taskId.slice(0, 8)}… reason=${reason}`);
  const ac = activeTaskAcs.get(taskId);
  if (!ac) {
    console.log(`[PROCESS] cancel 未命中进行中任务 (可能已完成或 taskId 无效)`);
    return false;
  }
  if (!ac.signal.aborted) ac.abort();
  console.log(`[PROCESS] 任务 ${taskId.slice(0, 8)}… 已中止 (${reason})`);
  return true;
}

function isSummaryError(summary: string): boolean {
  if (!summary?.trim()) return true;
  const prefixes = ['AI 摘要生成失败', 'AI 摘要生成超时', '文件内容提取失败', '请求已取消'];
  return prefixes.some((p) => summary.startsWith(p));
}

function watchClientDisconnect(req: Request, res: Response, onDisconnect: () => void) {
  const markGone = (reason: string) => {
    if (res.writableFinished || res.writableEnded) return;
    onDisconnect();
    console.log(`[PROCESS] SSE 客户端断开 (${reason})`);
  };
  req.on('aborted', () => markGone('aborted'));
  res.on('close', () => markGone('close'));
}

router.post('/', upload.single('file'), async (req: Request, res: Response) => {
  try {
    const file = req.file;
    if (!file) return sendJsonError(res, 400, '请上传文件');
    const raw = file.originalname;
    const decoded = Buffer.from(raw, 'latin1').toString('utf8');
    const originalName = /[\u4e00-\u9fff]/.test(decoded) ? decoded : raw;
    const taskId = require('crypto').randomUUID();
    taskStore.set(taskId, { filePath: path.resolve(file.path), originalName, fileSize: file.size });
    activeTaskAcs.set(taskId, new AbortController());
    console.log(`[PROCESS] 上传完成 taskId=${taskId.slice(0, 8)}… file=${originalName}`);
    res.json({ taskId, fileName: originalName, fileSize: file.size });
  } catch (error: unknown) {
    console.error('上传处理失败:', error);
    sendJsonError(res, 500, toClientError(error, '处理文件失败'));
  }
});

const handleCancel = (req: Request, res: Response) => {
  abortTask(req.params.taskId, 'cancel-api');
  res.json({ ok: true });
};

router.post('/cancel/:taskId', handleCancel);
router.get('/cancel/:taskId', handleCancel);

router.get('/process/:taskId', async (req: Request, res: Response) => {
  const { taskId } = req.params;
  const task = taskStore.get(taskId);
  if (!task) { sendJsonError(res, 404, '任务不存在'); return; }

  let ac = activeTaskAcs.get(taskId);
  if (!ac) {
    ac = new AbortController();
    activeTaskAcs.set(taskId, ac);
  }

  if (ac.signal.aborted) {
    console.log(`[PROCESS] 任务 ${taskId.slice(0, 8)}… 已被取消，跳过 AI 处理`);
    activeTaskAcs.delete(taskId);
    taskStore.delete(taskId);
    try { require('fs').unlinkSync(task.filePath); } catch { /* ignore */ }
    sendJsonError(res, 499, '任务已取消');
    return;
  }

  console.log(`[PROCESS] SSE 连接 taskId=${taskId.slice(0, 8)}…`);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = (data: any) => {
    if (res.writableEnded || res.writableFinished) return;
    res.write(`data: ${JSON.stringify(data)}\n\n`);
    if (typeof (res as any).flush === 'function') (res as any).flush();
  };

  let clientGone = false;
  const isGone = () => clientGone || Boolean(req.aborted) || ac!.signal.aborted;

  watchClientDisconnect(req, res, () => {
    clientGone = true;
    ac!.abort();
  });

  const heartbeat = setInterval(() => {
    if (isGone() || res.writableEnded) return;
    res.write(': ping\n\n');
    if (typeof (res as any).flush === 'function') (res as any).flush();
  }, 10000);

  const pythonUrl = process.env.PYTHON_AI_URL || 'http://localhost:8000';

  /** Python embedding progress(10~95) → 整体进度 35~90 */
  const mapEmbedProgress = (embedPct: number, done?: number, total?: number) => {
    const pct = embedPct ?? (total ? 10 + Math.round(85 * (done ?? 0) / total) : 10);
    return Math.min(90, Math.max(35, 35 + Math.round(((pct - 10) / 85) * 55)));
  };

  try {
    // Phase 1：先分析入库，再 AI 摘要
    send({ stage: 'vectorize', progress: 28, message: '正在解析文档…' });
    if (isGone()) return;

    const processResp = await fetch(`${pythonUrl}/ai/process-document`, {
      signal: ac.signal, method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filePath: task.filePath, originalName: task.originalName }),
    });
    if (isGone()) return;
    if (!processResp.ok) throw new Error('文档处理服务异常');

    let docId = '';
    const reader = processResp.body!.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    try {
      while (true) {
        if (isGone()) break;
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || line.startsWith(':') || !line.startsWith('data:')) continue;
          const payload = line.slice(line.startsWith('data: ') ? 6 : 5).trim();
          if (!payload) continue;
          try {
            const evt = JSON.parse(payload);
            if (evt.stage === 'reading') {
              send({ stage: 'vectorize', progress: 30, message: '正在读取文档…' });
            } else if (evt.stage === 'chunking') {
              send({ stage: 'vectorize', progress: 35, message: '正在分块…' });
            } else if (evt.stage === 'embedding') {
              send({
                stage: 'vectorize',
                progress: mapEmbedProgress(evt.progress, evt.done, evt.total),
                message: `分析文档 ${evt.done}/${evt.total}`,
              });
            } else if (evt.stage === 'done') {
              docId = evt.doc_id || '';
            } else if (evt.stage === 'error') {
              throw new Error(toClientError(evt.message, '文档处理失败'));
            }
          } catch (e) {
            if (e instanceof Error && !(e instanceof SyntaxError)) throw e;
          }
        }
      }
    } finally {
      try { await reader.cancel(); } catch { /* ignore */ }
    }

    if (isGone()) return;
    if (!docId) throw new Error('文档处理未完成');

    registerDoc(docId);

    send({ stage: 'vectorize', progress: 90, message: '文档已准备好，正在生成 AI 摘要…' });
    if (isGone()) return;

    let sumProg = 90;
    const summarizeTimer = setInterval(() => {
      if (isGone()) return;
      sumProg = Math.min(sumProg + 1, 98);
      send({ stage: 'summarize', progress: sumProg, message: '正在生成 AI 摘要…' });
    }, 1200);

    let summaryResp: globalThis.Response;
    try {
      summaryResp = await fetch(`${pythonUrl}/ai/summarize`, {
        signal: ac.signal, method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath: task.filePath, originalName: task.originalName }),
      });
    } finally {
      clearInterval(summarizeTimer);
    }
    if (isGone()) return;
    if (!summaryResp.ok) throw new Error('摘要服务异常');
    const summaryData = await summaryResp.json();
    const summaryOk = !isSummaryError(summaryData.summary || '');

    send({
      stage: 'summarize',
      progress: 98,
      message: summaryOk ? 'AI 摘要已生成' : 'AI 摘要失败，您仍可直接提问',
    });
    if (isGone()) return;

    const conversationId = require('crypto').randomUUID();
    const createdAt = new Date().toISOString();

    createConversation({
      id: conversationId,
      document_name: task.originalName,
      document_size: task.fileSize,
      summary: summaryData.summary || '',
      doc_id: docId,
      messages: '[]',
      created_at: createdAt,
    });
    linkDocToConversation(docId, conversationId);

    send({
      stage: 'done', progress: 100,
      result: {
        summary: summaryData.summary || '', summaryOk, docId, conversationId,
        fileName: task.originalName, fileSize: task.fileSize,
      },
    });
  } catch (error: unknown) {
    if (isGone() || (error instanceof Error && error.name === 'AbortError')) {
      console.log('[PROCESS] 用户离开，已中止后端 AI 任务');
      return;
    }
    console.error('[PROCESS] 失败:', error);
    send({ stage: 'error', message: toClientError(error, 'AI 处理失败') });
  } finally {
    clearInterval(heartbeat);
    activeTaskAcs.delete(taskId);
    taskStore.delete(taskId);
    try { require('fs').unlinkSync(task.filePath); } catch { /* ignore */ }
    if (!res.writableEnded) res.end();
  }
});

export default router;
