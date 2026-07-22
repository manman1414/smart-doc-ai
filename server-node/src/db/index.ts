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
      created_at    TEXT    NOT NULL,
      memory_summary TEXT   DEFAULT '',
      memory_covered INTEGER DEFAULT 0,
      memory_facts  TEXT    DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS doc_registry (
      doc_id          TEXT PRIMARY KEY,
      created_at      TEXT NOT NULL,
      conversation_id TEXT DEFAULT NULL
    );
  `);

  // 兼容旧库：补齐 memory_summary / memory_covered / memory_facts 列
  const cols = database.prepare(`PRAGMA table_info(conversations)`).all() as { name: string }[];
  if (!cols.some((c) => c.name === 'memory_summary')) {
    database.exec(`ALTER TABLE conversations ADD COLUMN memory_summary TEXT DEFAULT ''`);
  }
  if (!cols.some((c) => c.name === 'memory_covered')) {
    database.exec(`ALTER TABLE conversations ADD COLUMN memory_covered INTEGER DEFAULT 0`);
  }
  if (!cols.some((c) => c.name === 'memory_facts')) {
    database.exec(`ALTER TABLE conversations ADD COLUMN memory_facts TEXT DEFAULT ''`);
  }
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
  memory_summary?: string;
  memory_covered?: number;
  memory_facts?: string;
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
    INSERT INTO conversations (id, document_name, document_size, summary, doc_id, messages, created_at, memory_summary, memory_covered, memory_facts)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.id,
    row.document_name,
    row.document_size,
    row.summary,
    row.doc_id,
    row.messages,
    row.created_at,
    row.memory_summary || '',
    row.memory_covered || 0,
    row.memory_facts || '',
  );
}

/** 更新会话消息 */
export function updateConversationMessages(id: string, messages: string): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET messages = ? WHERE id = ?').run(messages, id);
}

/** 更新多轮对话滚动摘要（与文档 summary 不同） */
export function updateConversationMemorySummary(id: string, memorySummary: string): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET memory_summary = ? WHERE id = ?').run(memorySummary || '', id);
}

/** 更新已压缩进摘要的消息条数 */
export function updateConversationMemoryCovered(id: string, covered: number): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET memory_covered = ? WHERE id = ?').run(covered || 0, id);
}

/** 更新硬事实清单（与叙述摘要分离，只追加不改写） */
export function updateConversationMemoryFacts(id: string, memoryFacts: string): void {
  const d = getDb();
  d.prepare('UPDATE conversations SET memory_facts = ? WHERE id = ?').run(memoryFacts || '', id);
}

/** 删除会话 */
export function deleteConversationById(id: string): boolean {
  const d = getDb();
  const result = d.prepare('DELETE FROM conversations WHERE id = ?').run(id);
  return result.changes > 0;
}
