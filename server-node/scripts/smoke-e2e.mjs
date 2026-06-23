/**
 * SmartDoc 全流程 E2E 自动化测试
 * 上传 → 会话持久化 → 问答 → 摘要重试 → 删会话 → 错误文案检查
 * 作者: Cursor Agent / 2026-06-24
 *
 * 用法: node scripts/smoke-e2e.mjs
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

function hasChinese(s) {
  return /[\u4e00-\u9fff]/.test(s);
}

function hasBadErrorText(s) {
  if (!s) return true;
  if (/\(\s*\d{3}\s*\)/.test(s)) return true;
  if (/\berror code:\s*\d+/i.test(s)) return true;
  if (/\b(doc_id|taskId|AbortError|failed|status)\b/i.test(s)) return true;
  if (!hasChinese(s)) return true;
  return false;
}

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

async function collectSseEvents(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const resp = await fetch(url, { ...options, signal: controller.signal });
  if (!resp.ok) throw new Error(`SSE HTTP ${resp.status}`);

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

function assertUploadFlow(events) {
  const stages = events.map((e) => e.stage).filter(Boolean);
  const firstVectorize = stages.indexOf('vectorize');
  const firstSummarize = stages.indexOf('summarize');

  if (firstVectorize === -1) fail('SSE 含 vectorize 阶段');
  else pass('SSE 含 vectorize 阶段');

  if (firstSummarize === -1) fail('SSE 含 summarize 阶段');
  else pass('SSE 含 summarize 阶段');

  if (firstVectorize !== -1 && firstSummarize !== -1 && firstVectorize < firstSummarize) {
    pass('Phase1 顺序', `vectorize@${firstVectorize} 先于 summarize@${firstSummarize}`);
  } else {
    fail('Phase1 顺序');
  }

  const done = events.find((e) => e.stage === 'done');
  if (!done?.result?.docId) {
    fail('done 含 docId');
    return null;
  }
  pass('done 含 docId', done.result.docId.slice(0, 8) + '…');
  if (done.result.summary !== undefined) pass('done 含 summary 字段');
  else fail('done 含 summary 字段');

  return done.result;
}

async function assertSession(conversationId, docId) {
  const resp = await fetch(`${BASE}/conversations/${conversationId}`);
  if (!resp.ok) {
    fail('上传后会话持久化', `HTTP ${resp.status}`);
    return;
  }
  const conv = await resp.json();
  if (conv.docId === docId && conv.documentName) {
    pass('上传后会话持久化', `conversation=${conversationId.slice(0, 8)}…`);
  } else {
    fail('上传后会话持久化');
  }
}

async function testChatAsk(docId) {
  const events = await collectSseEvents(`${BASE}/chat/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: '这份文档是关于什么的？用一句话回答。',
      doc_id: docId,
      history: [],
    }),
  });

  const errEvt = events.find((e) => e.error);
  if (errEvt) {
    fail('问答 SSE', errEvt.error?.slice(0, 60));
    return false;
  }

  const hasToken = events.some((e) => e.token);
  const hasDone = events.some((e) => e.done);
  if (hasToken || hasDone) {
    pass('问答 SSE', hasToken ? '收到 token 流' : '收到 done');
    return true;
  }
  fail('问答 SSE', '无 token 或 done');
  return false;
}

async function testSummarizeRetry(docId) {
  const resp = await fetch(`${BASE}/chat/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: docId }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    fail('摘要重试', err.error || `HTTP ${resp.status}`);
    return;
  }
  const data = await resp.json();
  if (typeof data.summary === 'string' && data.summary.length > 0) {
    pass('摘要重试', `长度 ${data.summary.length}`);
  } else {
    fail('摘要重试', 'summary 为空');
  }
}

async function testSaveMessages(conversationId, docId, documentName) {
  const messages = [
    { id: 'u1', role: 'user', content: '测试问题', timestamp: Date.now() },
    { id: 'a1', role: 'assistant', content: '测试回答', timestamp: Date.now() },
  ];
  const resp = await fetch(`${BASE}/conversations/${conversationId}/messages`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      documentName,
      documentSize: 100,
      summary: '测试摘要',
      docId,
      createdAt: new Date().toISOString(),
    }),
  });
  if (!resp.ok) {
    fail('保存对话消息', `HTTP ${resp.status}`);
    return;
  }
  const convResp = await fetch(`${BASE}/conversations/${conversationId}`);
  const conv = await convResp.json();
  const chatCount = (conv.messages || []).filter(
    (m) => m.role === 'user' || m.role === 'assistant',
  ).length;
  if (chatCount >= 2) pass('保存对话消息', `${chatCount} 条`);
  else fail('保存对话消息', `实际 ${chatCount} 条`);
}

async function testDeleteConversation(conversationId) {
  const resp = await fetch(`${BASE}/conversations/${conversationId}`, { method: 'DELETE' });
  if (!resp.ok) {
    fail('删除会话', `HTTP ${resp.status}`);
    return;
  }
  const getResp = await fetch(`${BASE}/conversations/${conversationId}`);
  if (getResp.status === 404) pass('删除会话');
  else fail('删除会话', '删除后仍可访问');
}

async function testClientErrorFormat() {
  const resp = await fetch(`${BASE}/chat/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  if (resp.status !== 400) {
    fail('错误响应状态码', `期望 400 实际 ${resp.status}`);
    return;
  }
  const body = await resp.json().catch(() => ({}));
  const msg = body.error || '';
  if (hasBadErrorText(msg)) {
    fail('错误文案纯中文', msg || '(空)');
  } else {
    pass('错误文案纯中文', msg);
  }
}

async function testConversationList() {
  const resp = await fetch(`${BASE}/conversations`);
  if (!resp.ok) {
    fail('会话列表', `HTTP ${resp.status}`);
    return;
  }
  const list = await resp.json();
  if (Array.isArray(list)) pass('会话列表', `${list.length} 条`);
  else fail('会话列表', '非数组');
}

async function main() {
  console.log('=== SmartDoc 全流程 E2E 自动化测试 ===\n');

  if (!(await checkHealth())) {
    printReport();
    process.exit(1);
  }

  await testClientErrorFormat();
  await testConversationList();

  const fixtureDir = path.join(__dirname, 'fixtures');
  fs.mkdirSync(fixtureDir, { recursive: true });
  const testFile = path.join(fixtureDir, 'e2e-test.txt');
  fs.writeFileSync(
    testFile,
    `SmartDoc E2E 测试文档\n主题：自动化测试与文档问答。\n生成时间: ${new Date().toISOString()}\n`.repeat(30),
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
    console.log('等待 AI 处理（上传 → 向量化 → 摘要）…');
    events = await collectSseEvents(`${BASE}/upload/process/${taskId}`);
    pass('上传 SSE 完成', `共 ${events.length} 个事件`);
  } catch (e) {
    fail('上传 SSE', e.message);
    printReport();
    process.exit(1);
  }

  const doneResult = assertUploadFlow(events);
  if (!doneResult) {
    printReport();
    process.exit(1);
  }

  const { docId, conversationId, fileName } = doneResult;
  await assertSession(conversationId, docId);

  console.log('测试问答…');
  await testChatAsk(docId);

  console.log('测试摘要重试…');
  await testSummarizeRetry(docId);

  console.log('测试消息保存…');
  await testSaveMessages(conversationId, docId, fileName);

  console.log('测试删除会话…');
  await testDeleteConversation(conversationId);

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
