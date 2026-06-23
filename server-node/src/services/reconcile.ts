/**
 * SQLite ↔ Chroma 对账清理
 * - 宽限期 12h：上传后尚未产生对话的 doc_id 暂不删
 * - 对账间隔 24h：删除无 SQLite 引用且已过宽限的 Chroma 向量
 * - 同步清理 uploads/ 超过 24h 且无活跃任务的临时文件
 *
 * 作者: Cursor Agent / 2026-06-24
 */
import fs from 'fs';
import path from 'path';
import {
  getReferencedDocIds,
  getDocIdsWithinGracePeriod,
  deleteDocRegistry,
} from '../db';
import { listChromaDocIds, deleteChromaDoc } from './pythonAi';
import { getActiveUploadPaths } from '../routes/upload';

/** 宽限期（小时）：上传后无对话仍保留向量 */
export const GRACE_PERIOD_HOURS = 12;
/** 对账间隔（毫秒） */
export const RECONCILE_INTERVAL_MS = 24 * 60 * 60 * 1000;
/** uploads/ 临时文件 TTL（毫秒） */
export const UPLOAD_FILE_TTL_MS = 24 * 60 * 60 * 1000;

const UPLOADS_DIR = path.join(__dirname, '..', '..', 'uploads');

/** 构建应保留的 doc_id 集合 */
export function buildKeepDocIds(): Set<string> {
  const keep = new Set<string>();
  for (const id of getReferencedDocIds()) keep.add(id);
  for (const id of getDocIdsWithinGracePeriod(GRACE_PERIOD_HOURS)) keep.add(id);
  return keep;
}

/** 清理 uploads/ 下过期且非进行中的临时文件 */
export function cleanStaleUploadFiles(ttlMs: number = UPLOAD_FILE_TTL_MS): number {
  if (!fs.existsSync(UPLOADS_DIR)) return 0;

  const activePaths = getActiveUploadPaths();
  const now = Date.now();
  let removed = 0;

  for (const name of fs.readdirSync(UPLOADS_DIR)) {
    const full = path.resolve(UPLOADS_DIR, name);
    if (activePaths.has(full)) continue;

    try {
      const stat = fs.statSync(full);
      if (now - stat.mtimeMs > ttlMs) {
        fs.unlinkSync(full);
        removed += 1;
        console.log(`[CLEANUP] 删除过期临时文件 ${name}`);
      }
    } catch (e) {
      console.warn(`[CLEANUP] 跳过 uploads/${name}:`, e);
    }
  }

  return removed;
}

/** 执行一次 SQLite ↔ Chroma 对账 */
export async function runReconciliation(): Promise<void> {
  console.log('[CLEANUP] 开始对账…');
  const keep = buildKeepDocIds();

  let chromaDocIds: string[] = [];
  try {
    chromaDocIds = await listChromaDocIds();
  } catch (e) {
    console.error('[CLEANUP] 无法获取 Chroma doc_id 列表，跳过向量清理:', e);
  }

  let deletedVectors = 0;
  for (const docId of chromaDocIds) {
    if (keep.has(docId)) continue;
    const ok = await deleteChromaDoc(docId);
    if (ok) {
      deleteDocRegistry(docId);
      deletedVectors += 1;
      console.log(`[CLEANUP] 删除 orphan 向量 doc_id=${docId.slice(0, 8)}…`);
    }
  }

  const removedFiles = cleanStaleUploadFiles();

  console.log(
    `[CLEANUP] 对账完成 keep=${keep.size} chroma=${chromaDocIds.length} ` +
    `deleted_vectors=${deletedVectors} removed_uploads=${removedFiles}`,
  );
}

/** 启动定时对账（启动 1 分钟后首次执行，之后每 24h） */
export function startReconciliationScheduler(): void {
  const run = () => {
    runReconciliation().catch((e) => console.error('[CLEANUP] 对账失败:', e));
  };

  console.log(
    `[CLEANUP] 已启动定时对账：宽限 ${GRACE_PERIOD_HOURS}h，间隔 ${RECONCILE_INTERVAL_MS / 3600000}h`,
  );

  setTimeout(run, 60_000);
  setInterval(run, RECONCILE_INTERVAL_MS);
}
