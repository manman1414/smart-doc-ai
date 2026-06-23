import React, { useCallback } from 'react';
import { Card, Button, Popconfirm, Space, Typography } from 'antd';
import {
  FilePdfOutlined,
  FileTextOutlined,
  DeleteOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons';
import type { Conversation } from '@/types';

const { Text, Paragraph } = Typography;

interface HistoryCardProps {
  conversation: Conversation;
  onView: (id: string) => void;
  onDelete: (id: string) => void;
}

/**
 * 格式化 ISO 时间戳为可读日期
 * 输入： "2026-06-23T12:30:00.000Z"
 * 输出： "2026-06-23 12:30"
 */
function formatDate(isoStr: string): string {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return isoStr.slice(0, 16);        // 兜底：硬切
  }
}

const HistoryCard: React.FC<HistoryCardProps> = ({ conversation, onView, onDelete }) => {
  const isPdf: boolean = conversation.documentName.toLowerCase().endsWith('.pdf');
  const dialogCount: number = conversation.messages.filter(
    (m) => m.role !== 'system',
  ).length;

  const handleViewClick = useCallback(
    (e: React.MouseEvent): void => {
      e.stopPropagation();
      onView(conversation.id);
    },
    [conversation.id, onView],
  );

  const handleCardClick = useCallback((): void => {
    onView(conversation.id);
  }, [conversation.id, onView]);

  const handleDeleteConfirm = useCallback(
    (e?: React.MouseEvent): void => {
      e?.stopPropagation();
      onDelete(conversation.id);
    },
    [conversation.id, onDelete],
  );

  const handleDeleteCancel = useCallback((e?: React.MouseEvent): void => {
    e?.stopPropagation();
  }, []);

  const handleDeleteBtnClick = useCallback((e: React.MouseEvent): void => {
    e.stopPropagation();
  }, []);

  return (
    <Card
      hoverable
      style={{ borderRadius: 12, height: '100%' }}
      onClick={handleCardClick}
      styles={{
        body: {
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
        },
      }}
      actions={[
        <Button
          key="view"
          type="link"
          icon={<ArrowRightOutlined />}
          onClick={handleViewClick}
        >
          查看
        </Button>,
        <Popconfirm
          key="delete"
          title="确定要删除此会话？"
          description="删除后不可恢复"
          onConfirm={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
          okText="确定删除"
          cancelText="取消"
          okButtonProps={{ danger: true }}
        >
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={handleDeleteBtnClick}
          >
            删除
          </Button>
        </Popconfirm>,
      ]}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 12 }}>
        {isPdf ? (
          <FilePdfOutlined
            style={{ fontSize: 36, color: '#ff4d4f', flexShrink: 0 }}
          />
        ) : (
          <FileTextOutlined
            style={{ fontSize: 36, color: '#1677ff', flexShrink: 0 }}
          />
        )}
        <div style={{ minWidth: 0 }}>
          <Text strong style={{ fontSize: 14, wordBreak: 'break-word' }}>
            {conversation.documentName}
          </Text>
        </div>
      </div>

      <Paragraph
        type="secondary"
        style={{ fontSize: 13, marginBottom: 12, flex: 1 }}
        ellipsis={{ rows: 2 }}
      >
        {conversation.summary}
      </Paragraph>

      <Space
        style={{
          borderTop: '1px solid #f0f0f0',
          paddingTop: 12,
          width: '100%',
          justifyContent: 'space-between',
        }}
      >
        <Text type="secondary" style={{ fontSize: 12 }}>
          🕐 {formatDate(conversation.createdAt)}
        </Text>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {dialogCount > 0 ? `${dialogCount} 条对话` : '尚未提问'}
        </Text>
      </Space>
    </Card>
  );
};

export default HistoryCard;
