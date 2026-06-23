/**
 * 上传流程冒烟测试 — 验证 Phase 1：先 vectorize 再 summarize
 * 作者: Cursor Agent / 2026-06-24
 *
 * 用法: node scripts/smoke-upload-phase1.mjs
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.API_BASE || 'http://127.0.0.1:3000/api';
const TIMEOUT_MS = 10 * 60 * 1000;

const results = [];
const pass = (name, detail = '') => results.push({ name, ok: true, detail });
const fail = (name, detail = '') => results.push({ name, ok: false, detail });

async function checkHealth() {
  const [node, py] = await Promise.all([
    fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false),
    fetch('http://127.0.0.1:8000/health').then((r) => r.ok).catch(() => false),
  ]);
  if (!node) fail('Node 健康检查', BASE);
  else pass('Node 健康检查');
  if (!py) fail('Python 健康检查', '8000/health');
  else pass('Python 健康检查');
  return node && py;
}

async function uploadFile(filePath) {
  const buf = fs.readFileSync(filePath);
  const blob = new Blob([buf], { type: 'text/plain' });
  const form = new FormData();
  form.append('file', blob, path.basename(filePath));

  const resp = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!resp.ok) throw new Error(`upload HTTP ${resp.status}`);
  const data = await resp.json();
  if (!data.taskId) throw new Error('upload 无 taskId');
  return data;
}

async function collectSseEvents(taskId) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const resp = await fetch(`${BASE}/upload/process/${taskId}`, {
    signal: controller.signal,
  });
  if (!resp.ok) throw new Error(`process HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const events = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(line.startsWith('data: ') ? 6 : 5).trim();
        if (!payload) continue;
        try {
          events.push(JSON.parse(payload));
        } catch { /* skip */ }
      }
    }
  } finally {
    clearTimeout(timer);
  }
  return events;
}

function assertPhase1Order(events) {
  const stages = events.map((e) => e.stage).filter(Boolean);
  const firstVectorize = stages.indexOf('vectorize');
  const firstSummarize = stages.indexOf('summarize');

  if (firstVectorize === -1) {
    fail('SSE 含 vectorize 阶段');
    return;
  }
  pass('SSE 含 vectorize 阶段');

  if (firstSummarize === -1) {
    fail('SSE 含 summarize 阶段');
    return;
  }
  pass('SSE 含 summarize 阶段');

  if (firstVectorize < firstSummarize) {
    pass('Phase1 顺序', `vectorize@${firstVectorize} 先于 summarize@${firstSummarize}`);
  } else {
    fail('Phase1 顺序', `vectorize@${firstVectorize} summarize@${firstSummarize}`);
  }

  const vectorizeProgress = events
    .filter((e) => e.stage === 'vectorize' && e.progress != null)
    .map((e) => e.progress);
  const summarizeProgress = events
    .filter((e) => e.stage === 'summarize' && e.progress != null)
    .map((e) => e.progress);

  if (vectorizeProgress.length && summarizeProgress.length) {
    const maxV = Math.max(...vectorizeProgress);
    const minS = Math.min(...summarizeProgress);
    if (maxV >= 28 && minS >= 90) {
      pass('进度刻度', `vectorize max=${maxV}, summarize min=${minS}`);
    } else {
      fail('进度刻度', `vectorize max=${maxV}, summarize min=${minS}`);
    }
  }

  const done = events.find((e) => e.stage === 'done');
  if (!done?.result?.docId) {
    fail('done 含 docId');
    return;
  }
  pass('done 含 docId', done.result.docId.slice(0, 8) + '…');

  if (done.result.summary !== undefined) {
    pass('done 含 summary 字段');
  } else {
    fail('done 含 summary 字段');
  }

  return done.result;
}

async function assertUploadSessionPersisted(conversationId, docId) {
  const resp = await fetch(`${BASE}/conversations/${conversationId}`);
  if (!resp.ok) {
    fail('上传后会话持久化', `HTTP ${resp.status}`);
    return;
  }
  const conv = await resp.json();
  if (conv.docId === docId && conv.documentName) {
    pass('上传后会话持久化', `conversation=${conversationId.slice(0, 8)}…`);
  } else {
    fail('上传后会话持久化', 'docId 或 documentName 缺失');
  }

  const chatMsgs = (conv.messages || []).filter(
    (m) => m.role === 'user' || m.role === 'assistant',
  );
  if (chatMsgs.length === 0) {
    pass('无对话时 messages 为空');
  } else {
    fail('无对话时 messages 应为空', `实际 ${chatMsgs.length} 条`);
  }
}

async function main() {
  console.log('=== SmartDoc 上传冒烟测试 (Phase 1) ===\n');

  if (!(await checkHealth())) {
    printReport();
    process.exit(1);
  }

  const fixtureDir = path.join(__dirname, 'fixtures');
  fs.mkdirSync(fixtureDir, { recursive: true });
  const testFile = path.join(fixtureDir, 'smoke-test.txt');
  fs.writeFileSync(
    testFile,
    `SmartDoc Phase1 冒烟测试文档\n生成时间: ${new Date().toISOString()}\n`.repeat(20),
    'utf8',
  );

  let taskId;
  try {
    const up = await uploadFile(testFile);
    taskId = up.taskId;
    pass('文件上传', `taskId=${taskId.slice(0, 8)}…`);
  } catch (e) {
    fail('文件上传', e.message);
    printReport();
    process.exit(1);
  }

  let events;
  try {
    console.log('等待 AI 处理（可能数分钟，含 embedding）…');
    events = await collectSseEvents(taskId);
    pass('SSE 连接完成', `共 ${events.length} 个事件`);
  } catch (e) {
    fail('SSE 连接', e.message);
    printReport();
    process.exit(1);
  }

  const doneResult = assertPhase1Order(events);

  if (doneResult?.conversationId) {
    await assertUploadSessionPersisted(doneResult.conversationId, doneResult.docId);
  }

  printReport();
  const failed = results.filter((r) => !r.ok);
  process.exit(failed.length ? 1 : 0);
}

function printReport() {
  console.log('\n--- 结果 ---');
  for (const r of results) {
    console.log(`${r.ok ? '✅' : '❌'} ${r.name}${r.detail ? ` — ${r.detail}` : ''}`);
  }
  const ok = results.filter((r) => r.ok).length;
  console.log(`\n${ok}/${results.length} 通过`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
