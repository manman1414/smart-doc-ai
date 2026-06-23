/**
 * 面向客户端的错误文案（纯中文，不含 HTTP 状态码与英文）
 * 作者: Cursor Agent / 2026-06-24
 */
import type { Response } from 'express';

const DEFAULT_FALLBACK = '服务异常，请稍后重试';

/** 将内部异常转为可展示的中文文案 */
export function toClientError(raw: unknown, fallback = DEFAULT_FALLBACK): string {
  if (raw == null) return fallback;

  let msg =
    typeof raw === 'string'
      ? raw
      : raw instanceof Error
        ? raw.message
        : String(raw);
  msg = msg.trim();
  if (!msg) return fallback;

  if (/abort/i.test(msg) || msg === 'The operation was aborted') {
    return '请求已取消';
  }
  if (/fetch failed|ECONNREFUSED|ENOTFOUND|network/i.test(msg)) {
    return '无法连接 AI 服务，请确认服务已启动';
  }

  // 含 HTTP 状态码或 error code 的技术性文案
  if (
    /\berror code:\s*\d+/i.test(msg) ||
    /\b(返回|failed|status)\s*[:：]?\s*\d{3}\b/i.test(msg) ||
    /\(\s*\d{3}\s*\)/.test(msg) ||
    /\b\d{3}\b/.test(msg) && /返回|failed|status|code|gateway/i.test(msg)
  ) {
    if (/摘要/.test(msg)) return '摘要服务异常';
    if (/文档|处理|vector|embedding/i.test(msg)) return '文档处理服务异常';
    if (/问答|ask/i.test(msg)) return '问答服务异常';
    return fallback;
  }

  // 纯英文或无中文
  if (!/[\u4e00-\u9fff]/.test(msg)) return fallback;

  // 去掉常见英文字段名
  msg = msg
    .replace(/\b(doc_id|docId|taskId|question|messages|AbortError)\b/gi, '')
    .replace(/error code:\s*\d+/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

  if (!msg || !/[\u4e00-\u9fff]/.test(msg)) return fallback;
  return msg;
}

export function sendJsonError(res: Response, status: number, message: string): void {
  res.status(status).json({ error: toClientError(message, message) });
}
