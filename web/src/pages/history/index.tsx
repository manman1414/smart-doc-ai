/**
 * 历史记录页面 — 直接调后端 API，无本地缓存
 *
 * 数据流：
 *   GET /api/conversations      → 获取全部会话列表
 *   DELETE /api/conversations/:id → 删除单条会话
 *   查看会话 → navigate('/chat?conversation=xxx') → 聊天页读取恢复
 */
import React, { useState, useCallback, useEffect } from 'react';
import { Row, Col, Empty, Pagination, Spin, Typography, message, Flex } from 'antd';
import { useNavigate, useSearchParams } from 'umi';
import HistoryCard from '@/components/HistoryCard';
import { getConversations, deleteConversation } from '@/services/api';
import type { Conversation } from '@/types';

const { Title, Text } = Typography;
/** 每页显示的会话卡片数 */
const PAGE_SIZE = 6;

const HistoryPage: React.FC = () => {
  /** 读写地址栏 query string（用于历史记录→聊天页的 ?conversation=xxx 传参） */
  const [searchParams, setSearchParams] = useSearchParams();
  /** 会话列表（从后端 API 获取） */
  const [conversations, setConversations] = useState<Conversation[]>([]);
  /** 加载中状态 */
  const [loading, setLoading] = useState(true);
  /** 当前分页页码 */
  const [page, setPage] = useState(1);
  /** UmiJS 路由跳转 */
  const navigate = useNavigate();

  /**
   * 从后端 API 刷新会话列表
   * 每次进入页面时调用
   */
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getConversations();            // GET /api/conversations
      setConversations(data);
    } catch {
      message.error('加载历史记录失败');
    } finally {
      setLoading(false);
    }
  }, []);

  /** 挂载时 + navigate 跳转时自动刷新 */
  useEffect(() => { refresh(); }, [refresh]);

  /**
   * 查看会话 — 通过 URL 参数跳转到聊天页
   * @param id - 会话 ID
   * 聊天页 mount 时从 searchParams 读取并调 GET /api/conversations/:id 恢复
   */
  const handleView = useCallback((id: string) => {
    navigate({ pathname: '/chat', search: `?conversation=${id}` });
  }, [navigate]);

  /**
   * 删除会话 — 调后端 DELETE API + 前端乐观移除
   *   如果当前页删空了，自动跳到前一页
   * @param id - 会话 ID
   */
  const handleDelete = useCallback(async (id: string) => {
    await deleteConversation(id);                       // DELETE /api/conversations/:id
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id);
      // 当前页删空时自动跳到前一页
      const maxPage = Math.ceil(next.length / PAGE_SIZE) || 1;
      if (page > maxPage) setPage(maxPage);
      return next;
    });
    // 1. 复制一份当前参数（防止直接修改原对象）
    const newParams = new URLSearchParams(searchParams);
    // 2. 删除 conversation 参数
    newParams.delete('conversation');
    // 3. 设置新参数（地址栏会更新，组件会重新渲染）
    setSearchParams(newParams);
    message.success('删除成功');
  }, [page, searchParams, setSearchParams]);

  /** 分页切换 */
  const handlePageChange = useCallback((p: number) => setPage(p), []);

  // 前端分页切片
  const total = conversations.length;
  const paginated = conversations.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <Flex vertical style={{ padding: 24, height: '100%', overflowY: 'auto' }}>
      <Flex vertical gap={4} style={{ marginBottom: 24 }}>
        <Title level={4} style={{ margin: 0 }}>📋 历史会话记录</Title>
        <Text type="secondary">数据存储在本地 SQLite 数据库</Text>
      </Flex>

      {loading ? (
        <Flex vertical align="center" justify="center" style={{ padding: 100 }}>
          <Spin size="large" />
          <Text type="secondary" style={{ marginTop: 12 }}>加载中...</Text>
        </Flex>
      ) : total === 0 ? (
        <Empty description="暂无历史会话" style={{ marginTop: 80 }}>
          <Text type="secondary">上传文档并开始对话后，记录将出现在这里</Text>
        </Empty>
      ) : (
        <>
          <Row gutter={[16, 16]}>
            {paginated.map(conv => (
              <Col key={conv.id} xs={24} sm={12} lg={8} style={{marginTop:'60px'}}>
                <HistoryCard
                  conversation={conv}
                  onView={handleView}
                  onDelete={handleDelete}
                />
              </Col>
            ))}
          </Row>
          {/* 超过一页显示分页器 */}
          {total > PAGE_SIZE && (
            <Flex justify="center" style={{ marginTop: '52px', padding: '16px 0' }}>
              <Pagination
                current={page}
                pageSize={PAGE_SIZE}
                total={total}
                onChange={handlePageChange}
                showSizeChanger={false}
                showTotal={(t: number) => `共 ${t} 条会话`}
              />
            </Flex>
          )}
        </>
      )}
    </Flex>
  );
};

export default HistoryPage;
