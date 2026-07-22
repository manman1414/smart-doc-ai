/**
 * 会话历史接口
 *
 * GET  /api/conversations          — 获取所有会话列表
 * GET  /api/conversations/:id      — 获取单个会话详情
 * DELETE /api/conversations/:id    — 删除会话
 */
import { Router, Request, Response } from 'express';
import {
  listConversations,
  getConversation,
  deleteConversationById,
  createConversation,
  updateConversationMessages,
  linkDocToConversation,
  deleteDocRegistry,
  updateConversationSummary,
  updateConversationMemorySummary,
  updateConversationMemoryCovered,
} from '../db';
import { deleteChromaDoc } from '../services/pythonAi';
import { sendJsonError } from '../utils/clientError';
const router = Router();
declare module 'express' {
  interface Request {
    params: Record<string, string>;
  }
}
/** 获取会话列表 */
router.get('/', (_req: Request, res: Response) => {
  try {
    const rows = listConversations();
    // 将数据库行转换为前端期望的格式
    const conversations = rows.map((row) => ({
      id: row.id,
      documentName: row.document_name,
      documentSize: row.document_size,
      summary: row.summary,
      docId: row.doc_id,
      messages: safeParseJson(row.messages, []),
      createdAt: row.created_at,
      memorySummary: row.memory_summary || '',
      memoryCovered: row.memory_covered || 0,
    }));
    res.json(conversations);
  } catch (error: any) {
    console.error('获取会话列表失败:', error);
    sendJsonError(res, 500, '获取会话列表失败');
  }
});
/** 更新会话消息（有 docId 时可创建仅含文档信息的草稿会话，供刷新恢复） */
router.put('/:id/messages', (req: Request, res: Response) => {
  try {
    const { messages, documentName, documentSize, summary, docId, createdAt, memorySummary, memoryCovered } = req.body;
    if (!Array.isArray(messages)) {
      return sendJsonError(res, 400, '消息格式不正确');
    }

    const hasDoc = Boolean(docId && documentName);

    // 无对话且无文档元信息 → 跳过
    if (messages.length === 0 && !hasDoc) {
      return res.json({ success: true, skipped: true });
    }

    const existing = getConversation(req.params.id);
    if (!existing) {
      createConversation({
        id: req.params.id,
        document_name: documentName || '',
        document_size: documentSize || 0,
        summary: summary || '',
        doc_id: docId || '',
        messages: '[]',
        created_at: createdAt || new Date().toISOString(),
        memory_summary: typeof memorySummary === 'string' ? memorySummary : '',
        memory_covered: typeof memoryCovered === 'number' ? memoryCovered : 0,
      });
    }

    if (docId) {
      linkDocToConversation(docId, req.params.id);
    }

    if (existing && typeof summary === 'string') {
      updateConversationSummary(req.params.id, summary);
    }

    if (typeof memorySummary === 'string') {
      updateConversationMemorySummary(req.params.id, memorySummary);
    }

    if (typeof memoryCovered === 'number') {
      updateConversationMemoryCovered(req.params.id, memoryCovered);
    }

    if (messages.length > 0) {
      updateConversationMessages(req.params.id, JSON.stringify(messages));
    }

    res.json({ success: true });
  } catch (error: any) {
    console.error('更新消息失败:', error);
    sendJsonError(res, 500, '更新消息失败');
  }
});
/** 获取单个会话详情 */
router.get('/:id', (req: Request, res: Response) => {
  try {
    const row = getConversation(req.params.id);

    if (!row) {
      return sendJsonError(res, 404, '会话不存在');
    }
    res.json({
      id: row.id,
      documentName: row.document_name,
      documentSize: row.document_size,
      summary: row.summary,
      docId: row.doc_id,
      messages: safeParseJson(row.messages, []),
      createdAt: row.created_at,
      memorySummary: row.memory_summary || '',
      memoryCovered: row.memory_covered || 0,
    });
  } catch (error: any) {
    console.error('获取会话详情失败:', error);
    sendJsonError(res, 500, '获取会话详情失败');
  }
});
/** 删除会话（同步删除 Chroma 向量与 doc_registry） */
router.delete('/:id', async (req: Request, res: Response) => {
  try {
    const row = getConversation(req.params.id);
    if (!row) {
      return sendJsonError(res, 404, '会话不存在');
    }

    const docId = row.doc_id;
    if (docId) {
      await deleteChromaDoc(docId);
      deleteDocRegistry(docId);
    }

    deleteConversationById(req.params.id);
    res.json({ success: true });
  } catch (error: any) {
    console.error('删除会话失败:', error);
    sendJsonError(res, 500, '删除会话失败');
  }
});

/** 安全解析 JSON，失败时返回默认值 */
function safeParseJson(raw: string, fallback: any): any {
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export default router;
