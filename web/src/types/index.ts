/** SmartDoc AI 核心数据类型 */

/** 消息角色：用户提问 / AI 回答 / 系统通知 */
export type MessageRole = 'user' | 'assistant' | 'system';

/** 对话消息 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  /** 是否正在流式输出中 */
  isStreaming?: boolean;
}

/** 会话（后端 SQLite 持久化） */
export interface Conversation {
  id: string;
  documentName: string;
  documentSize?: number;
  summary: string;
  /** ChromaDB 文档 ID，恢复到聊天页时用于继续提问 */
  docId?: string;
  createdAt: string;
  messages: Message[];
  /** 更早多轮对话的滚动摘要（最近几轮仍用 messages 原文） */
  memorySummary?: string;
  /** 已并入滚动摘要的消息条数 */
  memoryCovered?: number;
}

/** 已上传文档信息 */
export interface DocInfo {
  name: string;
  size: number;
  /**
   * 文档在 ChromaDB 中的唯一标识
   * 由 Node.js 网关上传成功后返回，后续聊天接口必须携带
   * 数据流：上传页面存储 → 聊天请求时传入 → Python 用 docId 做向量检索
   */
  docId?: string;
}

/** 上传状态机 */
export type UploadStatus = 'idle' | 'uploading' | 'parsing' | 'done' | 'error';

/** 通用 API 响应包装（预留，当前未使用） */
export interface ApiResponse<T = unknown> {
  code: number;
  data: T;
  message: string;
}

/** 分页查询参数 */
export interface PageQuery {
  page: number;
  pageSize: number;
}

/** 分页查询结果 */
export interface PageResult<T> {
  list: T[];
  total: number;
  page: number;
  pageSize: number;
}
