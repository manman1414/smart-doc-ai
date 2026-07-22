import { Router, Request, Response } from 'express';
import { sendJsonError, toClientError } from '../utils/clientError';

const router = Router();

function watchClientDisconnect(req: Request, res: Response, onDisconnect: () => void) {
  let gone = false;
  const markGone = (reason: string) => {
    if (gone || res.writableFinished || res.writableEnded) return;
    gone = true;
    onDisconnect();
    console.log(`[ASK] SSE 客户端断开 (${reason})`);
  };
  req.on('aborted', () => markGone('aborted'));
  res.on('close', () => markGone('close'));
}

/**
 * POST /api/chat/ask
 * 转发请求到 Python AI 服务，并直接 pipe SSE 流到前端
 * 客户端停止/关页时 abort upstream，避免 LM 空转
 */
router.post('/ask', async (req: Request, res: Response) => {
  const abortController = new AbortController();
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  let clientGone = false;

  const onClientGone = () => {
    if (clientGone) return;
    clientGone = true;
    abortController.abort();
    reader?.cancel().catch(() => {});
  };
  watchClientDisconnect(req, res, onClientGone);

  // 覆盖记忆压缩 + 改写 + 检索 + 整段流式；首包超时不宜过短
  const timeoutId = setTimeout(() => abortController.abort(), 300_000);

  try {
    const { question, doc_id, history, memory_summary, memory_covered, memory_facts, conversation_id } = req.body;
    if (!question || !doc_id) {
      clearTimeout(timeoutId);
      return sendJsonError(res, 400, '请提供问题和文档信息');
    }
    const pythonUrl = process.env.PYTHON_AI_URL || 'http://localhost:8000';
    const response = await fetch(`${pythonUrl}/ai/ask`, {
      signal: abortController.signal,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        doc_id,
        history: history || [],
        memory_summary: memory_summary || '',
        memory_covered: memory_covered || 0,
        memory_facts: memory_facts || '',
        conversation_id: conversation_id || '',
      }),
    });

    if (!response.ok || !response.body) {
      clearTimeout(timeoutId);
      throw new Error('AI 服务异常');
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    reader = response.body.getReader();
    try {
      while (!clientGone) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!res.writableEnded) {
          res.write(value);
        }
      }
    } finally {
      clearTimeout(timeoutId);
      if (!res.writableEnded) {
        res.end();
      }
    }
  } catch (error: unknown) {
    clearTimeout(timeoutId);
    if (clientGone || (error instanceof Error && error.name === 'AbortError')) {
      if (!res.headersSent) {
        return sendJsonError(res, 499, '请求已取消');
      }
      if (!res.writableEnded) res.end();
      return;
    }
    console.error('问答请求失败:', error);
    if (!res.headersSent) {
      sendJsonError(res, 500, toClientError(error, '问答服务异常'));
    } else if (!res.writableEnded) {
      res.end();
    }
  }
});

/** POST /api/chat/summarize — LM 恢复后按 doc_id 重试摘要 */
router.post('/summarize', async (req: Request, res: Response) => {
  try {
    const { doc_id } = req.body;
    if (!doc_id) {
      return sendJsonError(res, 400, '缺少文档信息');
    }

    const pythonUrl = process.env.PYTHON_AI_URL || 'http://localhost:8000';
    const response = await fetch(`${pythonUrl}/ai/summarize-doc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id }),
    });

    if (!response.ok) {
      throw new Error('摘要服务异常');
    }

    const data = await response.json();
    res.json(data);
  } catch (error: unknown) {
    console.error('摘要重试失败:', error);
    sendJsonError(res, 500, toClientError(error, '摘要服务异常'));
  }
});

export default router;
