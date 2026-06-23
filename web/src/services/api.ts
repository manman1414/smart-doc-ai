/**
 * SmartDoc AI 前端 API 服务层
 *
 * 所有接口通过 UmiJS proxy 转发到 Node.js 网关 (Express, 端口 3000)
 * Node 网关再转发到 Python AI 服务 (FastAPI, 端口 8000)
 *
 * 上传流程：
 *   1. POST /api/upload       →  Node 接收文件后立即返回 taskId
 *   2. GET  /api/upload/process/:taskId  →  SSE 推送 AI 处理进度
 *
 * 问答流程：
 *   POST /api/chat/ask  →  Node:3000 ──→ Python:8000 /ai/ask ──→ LM Studio
 */
import type { Conversation } from '@/types';

/** 开发环境直连 Node:3000，避免 Umi/utoopack dev proxy 缓冲 SSE 导致进度条不实时 */
export function getApiBase(): string {
  if (typeof window === 'undefined') return '/api';
  const { port } = window.location;
  if (process.env.NODE_ENV === 'development' && (port === '8001' || port === '8000')) {
    return 'http://127.0.0.1:3000/api';
  }
  return '/api';
}

/** 解析 SSE 单行 data（兼容 `\r\n` 行尾，避免 JSON.parse 静默失败） */
function parseSseDataLine(rawLine: string): ProcessEvent | null {
  const line = rawLine.trim();
  if (!line || line.startsWith(':')) return null;
  if (!line.startsWith('data:')) return null;
  const payload = line.slice(line.startsWith('data: ') ? 6 : 5).trim();
  if (!payload) return null;
  try {
    return JSON.parse(payload) as ProcessEvent;
  } catch {
    return null;
  }
}

// ==================== 文件上传 ====================

/** 页面离开时用多种方式通知 Node 中止（Umi 代理下单纯 abort fetch 无法传到 Node） */
export function cancelUploadTask(taskId: string): void {
  if (!taskId) return;
  const url = `${getApiBase()}/upload/cancel/${taskId}`;

  // 1. 同步 XHR — 页面卸载时最可靠
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, false);
    xhr.send();
  } catch { /* ignore */ }

  // 2. sendBeacon + keepalive fetch 兜底
  navigator.sendBeacon?.(url, new Blob([], { type: 'text/plain' }));
  fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
}

/**
 * 上传文档文件 — 仅上传，不等待 AI 处理
 *
 * @param file        - 用户选择的文件 (PDF 或 TXT)
 * @param onProgress  - 上传进度回调，接收 0~100 的百分比（实时）
 * @param xhrRef      - 可选的 XHR 引用，用于页面卸载时取消上传
 * @returns taskId + 文件元信息，后续调用 processUpload 获取 AI 结果
 */
export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void,
  xhrRef?: { current: XMLHttpRequest | null },
): Promise<{ taskId: string; fileName: string; fileSize: number }> {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    if (xhrRef) xhrRef.current = xhr;

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress?.(pct);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhrRef) xhrRef.current = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve({
            taskId: data.taskId || '',
            fileName: data.fileName || file.name,
            fileSize: data.fileSize || file.size,
          });
        } catch {
          reject(new Error('解析响应失败'));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.error || '上传失败'));
        } catch {
          reject(new Error('上传失败'));
        }
      }
    });

    xhr.addEventListener('error', () => { if (xhrRef) xhrRef.current = null; reject(new Error('网络错误，请检查后端服务是否启动')); });
    xhr.addEventListener('abort', () => { if (xhrRef) xhrRef.current = null; reject(new Error('上传已取消')); });

    xhr.open('POST', `${getApiBase()}/upload`);
    xhr.send(formData);
  });
}

/** AI 处理进度事件类型 */
export interface ProcessEvent {
  stage: 'summarize' | 'vectorize' | 'done' | 'error';
  progress?: number;
  message?: string;
  result?: {
    summary: string;
    /** 摘要是否成功（LM Studio 失败时为 false，但向量化可能已完成） */
    summaryOk?: boolean;
    docId: string;
    conversationId: string;
    fileName: string;
    fileSize: number;
  };
}

/**
 * 连接 SSE 获取 AI 处理进度
 *
 * @param taskId     - 上传后返回的任务 ID
 * @param onEvent    - 进度事件回调
 * @param onError    - 错误回调
 * @param signal     - AbortSignal 用于取消
 */
export async function processUpload(
  taskId: string,
  onEvent: (event: ProcessEvent) => void,
  onError: (error: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

  const abortReader = () => {
    reader?.cancel().catch(() => {});
  };
  signal?.addEventListener('abort', abortReader, { once: true });

  try {
    const response = await fetch(`${getApiBase()}/upload/process/${taskId}`, { signal });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      onError((err as { error?: string }).error || '连接失败');
      return;
    }

    reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      if (signal?.aborted) return;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const data = parseSseDataLine(line);
        if (data) onEvent(data);
      }
    }
  } catch (err: unknown) {
    if (signal?.aborted) return;
    const msg = err instanceof Error ? err.message : '连接中断';
    onError(msg);
  } finally {
    signal?.removeEventListener('abort', abortReader);
    abortReader();
  }
}

// ==================== 聊天问答 ====================

export async function* streamChat(
  question: string,
  docId: string,
  history: { role: string; content: string }[] = [],
  signal?: AbortSignal,
): AsyncGenerator<{ text: string; done: boolean }> {
  const response = await fetch(`${getApiBase()}/chat/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, doc_id: docId, history }),
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error((err as any).error || '请求失败');
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';

  while (true) {
    if (signal?.aborted) {
      yield { text: fullText, done: true };
      return;
    }

    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) {
        const raw = trimmed.slice(trimmed.startsWith('data: ') ? 6 : 5).trim();
        try {
          const data = JSON.parse(raw);
          if (data.error) throw new Error(data.error);
          if (data.done) {
            yield { text: fullText, done: true };
            return;
          }
          if (data.token) {
            for (const ch of data.token) {
              fullText += ch;
              yield { text: fullText, done: false };
              await new Promise((r) => setTimeout(r, 10));
            }
          }
        } catch (e: any) {
          throw e;
        }
      }
    }
  }

  yield { text: fullText, done: true };
}

/** LM 恢复后按 doc_id 重试生成摘要 */
export async function retrySummaryByDocId(docId: string): Promise<{ summary: string }> {
  const response = await fetch(`${getApiBase()}/chat/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: docId }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || '摘要重试失败');
  }
  return response.json();
}

// ==================== 会话历史 ====================

export async function getConversations(): Promise<Conversation[]> {
  try {
    const response = await fetch(`${getApiBase()}/conversations`);
    if (!response.ok) {
      return [];
    }
    const data = await response.json();
    return data;
  } catch {
    return [];
  }
}

export async function getConversation(id: string): Promise<Conversation | null> {
  const response = await fetch(`${getApiBase()}/conversations/${id}`);
  if (!response.ok) return null;
  return response.json();
}

export async function saveConversation(conv: Conversation): Promise<void> {
  const response = await fetch(`${getApiBase()}/conversations/${conv.id}/messages`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: conv.messages,
      documentName: conv.documentName,
      documentSize: conv.documentSize,
      summary: conv.summary,
      docId: conv.docId,
      createdAt: conv.createdAt,
    }),
  });
  if (!response.ok) {
    throw new Error('保存会话失败');
  }
}

/** 上传完成后持久化文档会话（无对话消息，刷新可恢复） */
export async function saveUploadSession(conv: Omit<Conversation, 'messages'>): Promise<void> {
  await saveConversation({ ...conv, messages: [] });
}

export async function deleteConversation(id: string): Promise<boolean> {
  const response = await fetch(`${getApiBase()}/conversations/${id}`, { method: 'DELETE' });
  return response.ok;
}

// ==================== 工具函数 ====================

export function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function getSuggestedQuestions(): string[] {
  return [
    '这份文档的核心结论是什么？',
    '用三句话总结文档内容。',
    '文档中提到的关键数据有哪些？',
  ];
}

/** 判断摘要文本是否为失败信息（Python 失败时仍返回 200 + 错误文案） */
export function isSummaryError(summary: string): boolean {
  if (!summary?.trim()) return true;
  const prefixes = ['AI 摘要生成失败', 'AI 摘要生成超时', '文件内容提取失败', '请求已取消'];
  return prefixes.some((p) => summary.startsWith(p));
}

/** 将摘要失败信息转为用户可读文案（过滤 Error code: 502 等技术细节） */
export function formatSummaryErrorMessage(summary: string): string {
  if (!isSummaryError(summary)) return summary;

  if (summary.startsWith('AI 摘要生成超时')) {
    return '摘要生成超时，请稍后重试';
  }
  if (summary.startsWith('请求已取消')) {
    return '请求已取消';
  }
  if (summary.startsWith('文件内容提取失败')) {
    return summary.replace(/^文件内容提取失败：?/, '文档读取失败：');
  }

  // 已是友好文案（Python 新版返回）则去掉前缀直接展示
  const body = summary.replace(/^AI 摘要生成失败：?/, '').trim();
  if (body && !/^error code:/i.test(body) && !/internal server error/i.test(body)) {
    return body;
  }

  const raw = body.toLowerCase();
  if (raw.includes('502') || raw.includes('bad gateway')) {
    return 'LM Studio 服务不可用，请启动 Local Server 并加载模型';
  }
  if (raw.includes('503')) {
    return 'LM Studio 服务繁忙或未就绪，请稍后重试';
  }
  if (raw.includes('404') || raw.includes('not found')) {
    return '模型不存在，请检查 LM Studio 中是否已加载对应模型';
  }
  if (raw.includes('connection') || raw.includes('refused') || raw.includes('connect')) {
    return '无法连接 LM Studio，请确认 Local Server 已启动';
  }
  if (/error code:\s*\d+/i.test(body)) {
    return 'LM Studio 调用异常，请检查服务与模型是否就绪';
  }

  return '摘要生成失败，请检查 LM Studio 是否已启动并加载模型';
}