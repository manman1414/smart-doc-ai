import { Router, Request, Response } from 'express';
import { sendJsonError, toClientError } from '../utils/clientError';

const router = Router();

/**
 * POST /api/chat/ask
 * 转发请求到 Python AI 服务，并直接 pipe SSE 流到前端
 */
router.post('/ask', async (req: Request, res: Response) => {
  try {
    const { question, doc_id, history } = req.body;
    if (!question || !doc_id) {
      return sendJsonError(res, 400, '请提供问题和文档信息');
    }
    const pythonUrl= process.env.PYTHON_AI_URL || 'http://localhost:8000';
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), 120_000);
    const response = await fetch(`${pythonUrl}/ai/ask`, {
      signal: abortController.signal,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, doc_id, history }),
    });
    clearTimeout(timeoutId);

    if (!response.ok || !response.body) {
      throw new Error('AI 服务异常');
    }

    // 直接 pipe Python 的 SSE 流到前端
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    // Node 18+ ReadableStream → pipe to Express response
    const reader = response.body.getReader();
    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          res.end();
          break;
        }
        res.write(value);
      }
    };
    pump().catch(() => res.end());

  } catch (error: unknown) {
    console.error('问答请求失败:', error);
    sendJsonError(res, 500, toClientError(error, '问答服务异常'));
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
