/**
 * Node → Python AI 服务 HTTP 客户端（向量清理对账）
 * 作者: Cursor Agent / 2026-06-24
 */

const PYTHON_URL = process.env.PYTHON_AI_URL || 'http://localhost:8000';

/** 列出 Chroma 中所有 doc_id */
export async function listChromaDocIds(): Promise<string[]> {
  const resp = await fetch(`${PYTHON_URL}/ai/doc-ids`);
  if (!resp.ok) {
    throw new Error(`listChromaDocIds failed: ${resp.status}`);
  }
  const data = (await resp.json()) as { doc_ids?: string[] };
  return data.doc_ids ?? [];
}

/** 删除 Chroma 中指定 doc_id 的全部向量 */
export async function deleteChromaDoc(docId: string): Promise<boolean> {
  if (!docId) return false;
  const resp = await fetch(`${PYTHON_URL}/ai/doc/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  });
  if (!resp.ok) {
    console.error(`[CLEANUP] deleteChromaDoc ${docId.slice(0, 8)}… failed: ${resp.status}`);
    return false;
  }
  return true;
}
