/**
 * SQLite 数据库初始化与操作
 * 使用 better-sqlite3 同步 API，存储会话记录和消息
 */
import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

const DB_DIR = path.join(__dirname, '..', '..', 'db');
const DB_PATH = path.join(DB_DIR, 'smartdoc.db');

/** 全局单例数据库实例 */
let db: Database.Database | null = null;

/** 获取数据库实例（懒初始化） */
export function getDb(): Database.Database {
  if (!db) {
    // 确保 db 目录存在（better-sqlite3 不会自动创建父目录）
    if (!fs.existsSync(DB_DIR)) {
      fs.mkdirSync(DB_DIR, { recursive: true });
    }
    db = new Database(DB_PATH);
    db.pragma('journal_mode = WAL');
    initTables(db);
  }
  return db;
}

/** 创建表结构 */
function initTables(database: Database.Database): void {
  database.exec(`
    CREATE TABLE IF NOT EXISTS conversations (
      id            TEXT PRIMARY KEY,
      document_name TEXT    NOT NULL,
      document_size INTEGER DEFAULT 0,
      summary       TEXT    DEFAULT '',
      doc_id        TEXT    DEFAULT '',
      messages      TEXT    DEFAULT '[]',
      created_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS doc_registry (
      doc_id          TEXT PRIMARY KEY,
      created_at      TEXT NOT NULL,
      conversation_id TEXT DEFAULT NULL
    );
  `);
}

// ==================== 文档向量注册（对账宽限期） ====================

/** 上传 AI 处理成功后登记 doc_id（用于 12h 宽限期） */
export function registerDoc(docId: string): void {
  if (!docId) return;
  const d = getDb();
  d.prepare(`
    INSERT OR IGNORE INTO doc_registry (doc_id, created_at)
    VALUES (?, ?)
  `).run(docId, new Date().toISOString());
}

/** 首次保存会话时关联 conversation_id */
export function linkDocToConversation(docId: string, conversationId: string): void {
  if (!docId || !conversationId) return;
  registerDoc(docId);
  const d = getDb();
  d.prepare(`
    UPDATE doc_registry SET conversation_id = ? WHERE doc_id = ?
  `).run(conversationId, docId);
}

/** SQLite 会话中仍被引用的 doc_id */
export function getReferencedDocIds(): string[] {
  const d = getDb();
  const rows = d.prepare(`
    SELECT DISTINCT doc_id AS doc_id FROM conversations WHERE doc_id != ''
  `).all() as { doc_id: string }[];
  return rows.map((r) => r.doc_id);
}

/** 宽限期内登记的 doc_id（上传后尚未产生对话） */
export function getDocIdsWithinGracePeriod(hours: number): string[] {
  const d = getDb();
  const rows = d.prepare(`
    SELECT doc_id FROM doc_registry
    WHERE datetime(created_at) > datetime('now', ? || ' hours')
  `).all(`-${hours}`) as { doc_id: string }[];
  return rows.map((r) => r.doc_id);
}

/** 更新会话摘要（摘要重试成功后同步 SQLite） */
export function updateConversationSummary(id: string, summary: string): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET summary = ? WHERE id = ?').run(summary, id);
}

export function deleteDocRegistry(docId: string): void {
  if (!docId) return;
  const d = getDb();
  d.prepare('DELETE FROM doc_registry WHERE doc_id = ?').run(docId);
}

// ==================== 会话操作 ====================

export interface ConversationRow {
  id: string;
  document_name: string;
  document_size: number;
  summary: string;
  doc_id: string;
  messages: string;  // JSON 字符串
  created_at: string;
}

/** 获取所有会话列表（按创建时间倒序） */
export function listConversations(): ConversationRow[] {
  const d = getDb();
  return d.prepare('SELECT * FROM conversations ORDER BY created_at DESC').all() as ConversationRow[];
}

/** 获取单个会话 */
export function getConversation(id: string): ConversationRow | undefined {
  const d = getDb();
  return d.prepare('SELECT * FROM conversations WHERE id = ?').get(id) as ConversationRow | undefined;
}

/** 创建会话 */
export function createConversation(row: ConversationRow): void {
  const d = getDb();
  d.prepare(`
    INSERT INTO conversations (id, document_name, document_size, summary, doc_id, messages, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(row.id, row.document_name, row.document_size, row.summary, row.doc_id, row.messages, row.created_at);
}

/** 更新会话消息 */
export function updateConversationMessages(id: string, messages: string): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET messages = ? WHERE id = ?').run(messages, id);
}

/** 删除会话 */
export function deleteConversationById(id: string): boolean {
  const d = getDb();
  const result = d.prepare('DELETE FROM conversations WHERE id = ?').run(id);
  return result.changes > 0;
}
