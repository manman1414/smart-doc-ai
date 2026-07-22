/**
 * 智能问答主页面 — 全部走后端 API，无本地缓存/状态机
 *
 * 页面结构：
 *   左侧 Sider（320px）—— 文件上传区 / 文档信息卡 / AI 摘要卡 / 推荐问题
 *   右侧 Content       —— 聊天消息列表 + 底部输入框
 *
 * 数据流：
 *   上传 → POST /api/upload → { taskId } → SSE /api/upload/process/:taskId → { summary, docId, conversationId }
 *   提问 → POST /api/chat/ask { question, doc_id } → 流式更新消息
 *   消息同步 → PUT /api/conversations/:id/messages
 *   历史恢复 → URL ?conversation=xxx → GET /api/conversations/:id → 全量恢复
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { flushSync } from 'react-dom';
import { Layout, Progress, Typography, Empty, Divider, Button, message } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { useSearchParams } from 'umi';
import type { Message, DocInfo, UploadStatus } from '@/types';
import FileUpload from '@/components/FileUpload';
import DocInfoCard from '@/components/DocInfoCard';
import AISummaryCard from '@/components/AISummaryCard';
import ChatBubble from '@/components/ChatBubble';
import ChatInput from '@/components/ChatInput';
import {
  uploadDocument,
  processUpload,
  streamChat,
  saveConversation as saveConvApi,
  saveUploadSession,
  getConversation as fetchConversation,
  retrySummaryByDocId,
  uid,
  getSuggestedQuestions,
  isSummaryError,
} from '@/services/api';
import {
  registerUploadTask,
  clearUploadTask,
  cancelActiveUpload,
} from '@/services/uploadSession';
import styles from './index.less';

/** 上传阶段占整体进度的比例（0~25%） */
const UPLOAD_PROGRESS_WEIGHT = 25;
const { Sider, Content } = Layout;
const { Text } = Typography;

/** 初始欢迎消息 */
const WELCOME_MSG: Message = {
  id: 'welcome',
  role: 'system',
  content: '👋 欢迎使用 SmartDoc AI\n请先上传一份 PDF 或 TXT 文档，即可开始智能问答。',
  timestamp: Date.now(),
};

const ChatPage: React.FC = () => {
  // ======================== 状态 ========================

  /** 当前对话消息列表（含系统通知、用户提问、AI 回答） */
  const [messages, setMessages] = useState<Message[]>([WELCOME_MSG]);
  /** 已上传文档的元信息（名称、大小、docId） */
  const [doc, setDoc] = useState<DocInfo | null>(null);
  /**
   * ChromaDB 文档 ID
   * 上传成功后由 Node.js 网关返回，后续每次 POST /api/chat/ask 必须附带
   */
  const [docId, setDocId] = useState('');
  /** AI 摘要文本（上传后同步获取，展示在左侧卡片） */
  const [summary, setSummary] = useState('');
  /** 上传状态机：idle → uploading → done / error */
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  /** 上传进度百分比 0~100（上传 + AI 处理统一刻度） */
  const [uploadPercent, setUploadPercent] = useState(0);
  /** AI 处理阶段: idle | summarize | vectorize */
  const [processingStage, setProcessingStage] = useState<string>('idle');
  /** 进度条下方说明文字 */
  const [progressMessage, setProgressMessage] = useState('');
  /** 摘要是否成功（失败时文档可能已向量化，仍可问答） */
  const [summaryOk, setSummaryOk] = useState(true);
  /** 左侧摘要卡片：LM 恢复后重试生成中 */
  const [summaryStreaming, setSummaryStreaming] = useState(false);
  /** 是否正在接收 AI 流式回答 */
  const [streaming, setStreaming] = useState(false);
  /** 推荐问题列表（摘要生成后展示） */
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  /**
   * 后端 SQLite 中的会话 ID
   * 上传时由 Node.js 创建记录并返回，消息同步时用于 UPDATE
   */
  const [conversationId, setConversationId] = useState('');
  /** 更早多轮的滚动摘要（与文档 summary 不同） */
  const [memorySummary, setMemorySummary] = useState('');
  const memorySummaryRef = useRef('');
  useEffect(() => { memorySummaryRef.current = memorySummary; }, [memorySummary]);
  const [memoryCovered, setMemoryCovered] = useState(0);
  const memoryCoveredRef = useRef(0);
  useEffect(() => { memoryCoveredRef.current = memoryCovered; }, [memoryCovered]);
  /** 会话创建时间（上传完成时捕获，写入后端后不再变化） */
  const [createdAt, setCreatedAt] = useState('');

  /** 消息列表滚动容器 DOM 引用 */
  const listRef = useRef<HTMLDivElement>(null);
  /** 流式输出中止标志（handleStop 时置 true） */
  const abortRef = useRef(false);
  /** AbortController 用于中断 fetch 请求 */
  const abortCtrlRef = useRef<AbortController | null>(null);
  /** 上传 XHR 引用，页面刷新/卸载时用于取消上传 */
  const uploadXhrRef = useRef<XMLHttpRequest | null>(null);
  /** SSE 处理进度 AbortController */
  const processCtrlRef = useRef<AbortController | null>(null);
  /** 用于区分 StrictMode 伪卸载 vs 真正离开页面 */
  const mountGenRef = useRef(0);
  const uploadStatusRef = useRef<UploadStatus>(uploadStatus);
  /** 当前上传 taskId，离开页面时通知 Node cancel */
  const currentTaskIdRef = useRef('');
  /** 单调递增进度，避免 SSE 乱序或重复导致条往回跳 */
  const uploadPercentRef = useRef(0);

  const applyUploadProgress = useCallback((pct: number, message?: string) => {
    const next = Math.max(uploadPercentRef.current, Math.min(100, pct));
    uploadPercentRef.current = next;
    flushSync(() => {
      setUploadPercent(next);
      if (message) setProgressMessage(message);
    });
  }, []);

  /** 中止上传 / AI 解析 / 流式问答 */
  const abortActiveWork = useCallback((reason = 'user') => {
    abortRef.current = true;
    abortCtrlRef.current?.abort();
    uploadXhrRef.current?.abort();
    processCtrlRef.current?.abort();
    cancelActiveUpload(reason);
  }, []);

  /** 以下 refs 供组件卸载清理使用 */
  const messagesRef = useRef(messages);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const convIdRef = useRef(conversationId);
  useEffect(() => { convIdRef.current = conversationId; }, [conversationId]);
  const docRef = useRef(doc);
  useEffect(() => { docRef.current = doc; }, [doc]);
  const summaryRef = useRef(summary);
  useEffect(() => { summaryRef.current = summary; }, [summary]);
  const docIdRef = useRef(docId);
  useEffect(() => { docIdRef.current = docId; }, [docId]);
  const summaryOkRef = useRef(summaryOk);
  useEffect(() => { summaryOkRef.current = summaryOk; }, [summaryOk]);
  const summaryRetryInFlightRef = useRef(false);
  useEffect(() => { uploadStatusRef.current = uploadStatus; }, [uploadStatus]);

  /** 摘要曾失败且 LM 已恢复时，在用户提问时后台重试摘要 */
  const tryRetrySummary = useCallback(() => {
    if (summaryOkRef.current || !docIdRef.current || summaryRetryInFlightRef.current) return;

    summaryRetryInFlightRef.current = true;
    setSummaryStreaming(true);

    void retrySummaryByDocId(docIdRef.current)
      .then(({ summary: s }) => {
        const ok = !isSummaryError(s);
        setSummary(s);
        setSummaryOk(ok);
        const convId = convIdRef.current;
        const d = docRef.current;
        if (convId && d) {
          void saveUploadSession({
            id: convId,
            documentName: d.name,
            documentSize: d.size,
            summary: s,
            docId: docIdRef.current,
            createdAt: createdAt || new Date().toISOString(),
          });
        }
      })
      .catch(() => { /* LM 仍不可用，保持当前失败态 */ })
      .finally(() => {
        summaryRetryInFlightRef.current = false;
        setSummaryStreaming(false);
      });
  }, [createdAt]);

  // ======================== URL 参数恢复 ========================

  /** 读写地址栏 query string（用于历史记录→聊天页的 ?conversation=xxx 传参） */
  const [searchParams, setSearchParams] = useSearchParams();

  /**
   * 会话恢复 — 仅 mount 时执行一次
   * 流程：读取 ?conversation=xxx → GET /api/conversations/:id → 恢复全部状态
   */
  useEffect(() => {
    const convId = searchParams.get('conversation');
    if (!convId) return;

    (async () => {
      // 调后端获取会话详情
      const conv = await fetchConversation(convId);
      if (!conv) {
        message.warning('会话不存在或已被删除');
        return;
      }

      // 恢复文档相关状态
      setDoc({ name: conv.documentName, size: conv.documentSize || 0, docId: conv.docId });
      setDocId(conv.docId || '');
      setConversationId(conv.id);
      setCreatedAt(conv.createdAt || '');
      setSummary(conv.summary);
      setMemorySummary(conv.memorySummary || '');
      setMemoryCovered(conv.memoryCovered || 0);
      setSummaryOk(!isSummaryError(conv.summary));
      setUploadStatus('done');
      setSuggestedQuestions(getSuggestedQuestions());

      // 恢复消息列表：去流式标记 + 移除残留的"AI 生成中"系统通知
      const cleanMessages = conv.messages
        .filter(m => !(m.role === 'system' && m.content.includes('AI 生成中')))
        .map(m => ({ ...m, isStreaming: false }));

      setMessages([
        {
          id: 'welcome-restored', role: 'system',
          content: `📂 已恢复会话「${conv.documentName}」，您可以继续提问。`,
          timestamp: Date.now()
        },
        ...cleanMessages,
      ]);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ======================== 副作用 ========================

  /**
   * 组件卸载：StrictMode 伪卸载跳过；真卸载时 cancel + 持久化对话
   * （刷新/关页由 Layout 的 pagehide 处理；跳转历史由 Layout menu-nav 处理）
   */
  useEffect(() => {
    const gen = ++mountGenRef.current;
    return () => {
      const unmountGen = gen;
      queueMicrotask(() => {
        if (mountGenRef.current !== unmountGen) return;

        if (currentTaskIdRef.current || uploadStatusRef.current === 'uploading') {
          abortActiveWork('unmount');
        }

        abortRef.current = true;
        abortCtrlRef.current?.abort();
        if (uploadStatusRef.current === 'uploading') return;

        const msgs = messagesRef.current;
        const convId = convIdRef.current;
        const d = docRef.current;
        if (!convId || !d) return;

        const chatMsgs = msgs
          .filter(m => m.role === 'user' || m.role === 'assistant')
          .map(m => m.isStreaming
            ? { ...m, isStreaming: false, content: (m.content || '') + ' (已停止)' }
            : m
          );

        fetch(`/api/conversations/${convId}/messages`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: chatMsgs,
            documentName: d.name,
            documentSize: d.size,
            summary: summaryRef.current,
            docId: docIdRef.current,
            createdAt: new Date().toISOString(),
            memorySummary: memorySummaryRef.current || '',
            memoryCovered: memoryCoveredRef.current || 0,
          }),
        }).catch(() => {});
      });
    };
  }, [abortActiveWork]);

  /** 每次消息更新后自动滚动到底部 */
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  /**
   * 消息同步到后端 SQLite
   * 每次 messages 变化时触发，过滤掉系统消息，仅保存 user/assistant 对话
   * 调用 PUT /api/conversations/:id/messages
   */
  useEffect(() => {
    if (!conversationId || !doc) return;
    const chatMessages = messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ ...m, isStreaming: false }));     // ★ 永不对后端持久化 isStreaming，避免刷新后残留
    if (chatMessages.length === 0) return;
    const convId = searchParams.get('conversation');
    //如果地址栏没id并且实际有对话，则地址栏增加id
    if (!convId && conversationId) {
      const newParams = new URLSearchParams(searchParams); // 复制一份
      newParams.set('conversation', conversationId);
      setSearchParams(newParams); // 传入新实例

    }

    saveConvApi({
      id: conversationId,
      documentName: doc.name,
      documentSize: doc.size,
      summary,
      docId,                                          // ★ 存入 ChromaDB ID，恢复会话后能继续提问
      createdAt: createdAt || new Date().toISOString(),// 首次用上传时间，兜底用当前时间
      messages: chatMessages,
      memorySummary: memorySummary || '',
      memoryCovered: memoryCovered || 0,
    });
  }, [messages, summary, doc, conversationId, createdAt, memorySummary, memoryCovered]);

  // ======================== 事件处理 ========================

  /**
   * 处理文件上传
   */
  const handleUpload = useCallback(async (file: File) => {
    setUploadStatus('uploading');
    uploadPercentRef.current = 0;
    setUploadPercent(0);
    setProgressMessage('准备上传…');
    setSummaryOk(true);
    setProcessingStage('idle');
    processCtrlRef.current?.abort();
    abortRef.current = false;
    currentTaskIdRef.current = '';

    try {
      // ★ 阶段 1: 上传文件（实时进度 0~25%），Node 立即返回 taskId
      const { taskId } = await uploadDocument(
        file,
        (pct) => {
          const mapped = Math.round((pct / 100) * UPLOAD_PROGRESS_WEIGHT);
          applyUploadProgress(mapped, `上传中 ${pct}%`);
        },
        uploadXhrRef,
      );

      currentTaskIdRef.current = taskId;
      registerUploadTask(taskId, () => {
        uploadXhrRef.current?.abort();
        processCtrlRef.current?.abort();
      });

      setProcessingStage('vectorize');
      setProgressMessage('连接 AI 处理服务…');
      const ac = new AbortController();
      processCtrlRef.current = ac;

      await processUpload(
        taskId,
        (event) => {
          if (event.progress != null || event.message) {
            applyUploadProgress(
              event.progress ?? uploadPercentRef.current,
              event.message,
            );
          }

          if (event.stage === 'summarize') {
            setProcessingStage('summarize');
          } else if (event.stage === 'vectorize') {
            setProcessingStage('vectorize');
          } else if (event.stage === 'done' && event.result) {
            const ok = event.result.summaryOk ?? !isSummaryError(event.result.summary);
            const created = new Date().toISOString();
            setProcessingStage('idle');
            uploadPercentRef.current = 100;
            setUploadPercent(100);
            setProgressMessage('');
            setUploadStatus('done');
            setDoc({ name: event.result.fileName, size: event.result.fileSize, docId: event.result.docId });
            setDocId(event.result.docId);
            setConversationId(event.result.conversationId);
            setCreatedAt(created);
            setMemorySummary('');
            setMemoryCovered(0);
            setSummary(event.result.summary);
            setSummaryOk(ok);
            setSuggestedQuestions(getSuggestedQuestions());
            setStreaming(false);
            setMessages([
              {
                id: uid(), role: 'system',
                content: ok
                  ? `📄 文档「${event.result.fileName}」已上传，AI 摘要已生成，您可以开始提问。`
                  : `📄 文档「${event.result.fileName}」已上传完成，但 AI 摘要未生成（请检查 LM Studio），您可以直接提问。`,
                timestamp: Date.now(),
              },
            ]);
            // ★ 上传完成即写入 SQLite + URL，刷新后可恢复文档
            void saveUploadSession({
              id: event.result.conversationId,
              documentName: event.result.fileName,
              documentSize: event.result.fileSize,
              summary: event.result.summary,
              docId: event.result.docId,
              createdAt: created,
            }).then(() => {
              setSearchParams({ conversation: event.result!.conversationId });
            }).catch(() => {
              message.warning('会话保存失败，刷新后可能无法恢复文档');
            });
            processCtrlRef.current = null;
            currentTaskIdRef.current = '';
            clearUploadTask();
          } else if (event.stage === 'error') {
            throw new Error(event.message || 'AI 处理失败');
          }
        },
        // onError
        (error) => { throw new Error(error); },
        ac.signal,
      );
    } catch (err: any) {
      // 用户刷新/离开页面触发的 abort，静默结束
      if (processCtrlRef.current?.signal.aborted || err?.name === 'AbortError') {
        processCtrlRef.current = null;
        currentTaskIdRef.current = '';
        clearUploadTask();
        return;
      }
      processCtrlRef.current = null;
      currentTaskIdRef.current = '';
      clearUploadTask();
      setProcessingStage('idle');
      setProgressMessage('');
      setUploadStatus('error');
      uploadPercentRef.current = 0;
      setUploadPercent(0);
      setMessages(prev => [...prev,
        { id: uid(), role: 'system', content: `❌ 上传失败：${err.message || '未知错误'}`, timestamp: Date.now() },
      ]);
    }
  }, [applyUploadProgress, setSearchParams]);

  /**
   * 发送聊天消息
   *
   * @param text - 用户输入的问题文本
   *
   * 流程：
   *   1. 创建 userMsg + 占位 aiMsg 加入消息列表
   *   2. 调用 streamChat(text, docId) → POST /api/chat/ask → 逐字符流式渲染
   *   3. 用户可随时通过 handleStop 中断（abortRef.current = true）
   */
  const handleSend = useCallback(async (text: string) => {
    if (streaming) return;                              // 防止重复发送
    if (!docId) return;                                 // 未上传文档，不响应

    tryRetrySummary();

    // 用户消息
    const userMsg: Message = { id: uid(), role: 'user', content: text, timestamp: Date.now() };
    // AI 占位消息（后续流式填充内容）
    const aiMsgId = uid();
    const aiMsg: Message = { id: aiMsgId, role: 'assistant', content: '', timestamp: Date.now(), isStreaming: true };
    // 系统通知：AI 生成中
    const genMsgId = uid();
    const genMsg: Message = { id: genMsgId, role: 'system', content: '🤖 AI 生成中，请稍等…', timestamp: Date.now() };

    setMessages(prev => [
      // 移除未上传文档提示
      ...prev.filter(m => m.id !== 'welcome' && m.id !== 'welcome-restored'),
      userMsg, genMsg, aiMsg,
    ]);
    setStreaming(true);
    abortRef.current = false;
    // 每次请求创建新的 AbortController
    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;

    try {
      // ★ 传入 docId + signal 做 RAG；附带滚动摘要与完整 history（后端再拆分）
      const history = messagesRef.current
        .filter(m => m.role !== 'system' && m.id !== 'welcome' && m.id !== 'welcome-restored')
        .map(m => ({ role: m.role, content: m.content }));
      const gen = streamChat(
        text,
        docId,
        history,
        ctrl.signal,
        memorySummaryRef.current || '',
        memoryCoveredRef.current || 0,
      );
      for await (const { text: chunk, done, memorySummary: nextMem, memoryCovered: nextCovered } of gen) {
        if (abortRef.current) break;                    // 用户点了停止
        if (done) {
          if (typeof nextMem === 'string') setMemorySummary(nextMem);
          if (typeof nextCovered === 'number') setMemoryCovered(nextCovered);
        }
        setMessages(prev => prev.map(m =>
          m.id === aiMsgId ? { ...m, content: chunk, isStreaming: !done } : m
        ));
      }
    } catch (err: any) {
      setStreaming(false);
      const errorMsg = err?.message || '请求失败';
      setMessages(prev => [
        ...prev
          .filter(m => m.id !== genMsgId)
          .map(m => m.id === aiMsgId ? { ...m, isStreaming: false, content: m.content || errorMsg } : m),
        { id: uid(), role: 'system', content: '❌ ' + errorMsg, timestamp: Date.now() },
      ]);
    } finally {
      setStreaming(false);
      if (abortRef.current) {
        setMessages(prev => prev.map(m =>
          m.id === aiMsgId ? {
            ...m, isStreaming: false,
            content: m.content?.endsWith(' (已停止)') ? m.content : (m.content || '') + ' (已停止)',
          } : m
        ));
        setMessages(prev => [
          ...prev.filter(m => m.id !== genMsgId),
          { id: uid(), role: 'system', content: '⏹️ AI 生成已停止。', timestamp: Date.now() },
        ]);
      } else {
        setMessages(prev => [
          ...prev
            .filter(m => m.id !== genMsgId)
            .map(m => m.id === aiMsgId ? { ...m, isStreaming: false } : m),
          { id: uid(), role: 'system', content: '✅ AI 生成完毕，欢迎继续提问。', timestamp: Date.now() },
        ]);
      }
    }
  }, [streaming, docId, tryRetrySummary]);

  /** 停止当前 AI 流式回答 */
  const handleStop = useCallback(() => {
    abortRef.current = true;
    abortCtrlRef.current?.abort();
    setStreaming(false);
    setMessages(prev => prev.map(m =>
      m.isStreaming ? { ...m, isStreaming: false, content: m.content + ' (已停止)' } : m
    ));
  }, []);

  /**
   * 清除当前文档 — 重置所有状态到初始值
   * 清理：文档信息 / AI 摘要 / 聊天记录 / 上传状态 / URL 参数
   */
  /**
   * 清除当前文档 — 重置本地视图状态
   *
   * 注意：不删除后端历史记录！
   * 历史记录只在历史页手动删除时清除，这里只做前端 UI 重置。
   */
  const handleClearDocument = useCallback(() => {
    if (streaming) return;
    setDoc(null);
    setDocId('');
    setConversationId('');
    setMemorySummary('');
    setMemoryCovered(0);
    setCreatedAt('');
    setSummary('');
    setSummaryOk(true);
    setSummaryStreaming(false);
    setUploadStatus('idle');
    uploadPercentRef.current = 0;
    setUploadPercent(0);
    setProgressMessage('');
    setProcessingStage('idle');
    processCtrlRef.current?.abort();
    setSuggestedQuestions([]);
    setMessages([WELCOME_MSG]);
    abortRef.current = false;
    setSearchParams({});
  }, [streaming, setSearchParams]);

  /** 点击推荐问题 → 直接发起提问 */
  const handleAskQuestion = useCallback((q: string) => handleSend(q), [handleSend]);

  // ======================== 渲染 ========================
  return (
    <Layout hasSider className={styles.page}>
      {/* ★ 左侧栏：文件上传 + 文档信息 + AI 摘要 + 推荐问题 */}
      <Sider width={320} className={styles.sider} theme="light">
        {/* 文件上传区 */}
        <div className={styles.siderSection}>
          <Text strong type="secondary" style={{ fontSize: 13 }}>📁 文件上传</Text>
          {uploadStatus === 'done' ? null : (             // 上传完成后隐藏上传区
            <div style={{ marginTop: 8 }}>
              <FileUpload onUpload={handleUpload} disabled={uploadStatus === 'uploading'} />
              {uploadStatus === 'uploading' && (
                <div style={{ marginTop: 12 }}>
                  <Progress
                    percent={Math.round(uploadPercent)}
                    size="small"
                    status="active"
                    showInfo
                    strokeLinecap="round"
                  />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {progressMessage || (processingStage === 'summarize'
                      ? '🤖 AI 摘要生成中…'
                      : processingStage === 'vectorize'
                        ? '📊 正在分析文档…'
                        : '处理中…')}
                  </Text>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 文档信息卡片（上传成功后显示） */}
        {doc && <>
          <Divider style={{ margin: 0 }} />
          <div className={styles.siderSection}>
            <DocInfoCard
              doc={doc}
              processing={uploadStatus === 'uploading'}
              summaryOk={summaryOk}
            />
          </div>
          <div style={{ padding: '0 16px 12px' }}>
            <Button icon={<DeleteOutlined />} danger size="small" block
              onClick={handleClearDocument} disabled={streaming}>
              清除当前文档
            </Button>
          </div>
        </>}

        {/* AI 摘要 + 推荐问题（有文档或摘要时显示） */}
        {(doc || summary) && <>
          <Divider style={{ margin: 0 }} />
          <div className={styles.siderSection}>
            <AISummaryCard
              summary={summary}
              streaming={summaryStreaming}
              suggestedQuestions={summaryOk ? suggestedQuestions : []}
              onAskQuestion={handleAskQuestion}
            />
          </div>
        </>}
      </Sider>

      {/* ★ 右侧主区：聊天消息列表 + 底部输入框 */}
      <Content className={styles.chatArea}>
        <div className={styles.messageList} ref={listRef}>
          {messages.map(msg => <ChatBubble key={msg.id} message={msg} />)}
        </div>
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          streaming={streaming}
          disabled={uploadStatus === 'uploading'}        // 上传中禁用输入
        />
      </Content>
    </Layout>
  );
};

export default ChatPage;
