/**
 * 全局上传任务会话 — 跨路由/Layout 与 Chat 页共享
 * 作者: Cursor Agent / 2026-06-24
 *
 * Chat 页注册 taskId，Layout 在离开 /chat 时 cancel（Chat 卸载后 history.listen 无效）
 */
import { cancelUploadTask } from './api';

let activeTaskId = '';
let abortLocal: (() => void) | null = null;

/** 上传/解析开始（拿到 taskId 后调用） */
export function registerUploadTask(taskId: string, localAbort?: () => void): void {
  activeTaskId = taskId;
  abortLocal = localAbort ?? null;
}

/** 上传/解析正常结束 */
export function clearUploadTask(): void {
  activeTaskId = '';
  abortLocal = null;
}

/** 是否有进行中的上传任务 */
export function hasActiveUploadTask(): boolean {
  return !!activeTaskId;
}

/** 离开 chat 或刷新时中止（通知 Node + 本地 abort） */
export function cancelActiveUpload(reason: string): void {
  const taskId = activeTaskId;
  if (!taskId) return;

  try {
    abortLocal?.();
  } catch { /* ignore */ }

  cancelUploadTask(taskId);
  activeTaskId = '';
  abortLocal = null;

  if (process.env.NODE_ENV === 'development') {
    console.info(`[uploadSession] cancel (${reason}) taskId=${taskId.slice(0, 8)}…`);
  }
}
