import React from 'react';
import { Avatar, Card, Flex, Typography, Spin } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import type { Message } from '@/types';
import Markdown from './Markdown';

const { Text } = Typography;

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isUser: boolean = message.role === 'user';

  // 系统消息 → Typography 居中文本
  if (message.role === 'system') {
    return (
      <Flex justify="center" style={{ width: '100%', padding: '4px 16px' }}>
        <Text type="secondary" style={{ fontSize: 13 }}>
          <Markdown content={message.content} />
          {message.isStreaming && <span className="typing-cursor" />}
        </Text>
      </Flex>
    );
  }

  // 用户/AI 消息 → Flex + Card + Avatar
  return (
    <Flex vertical gap={2} style={{ maxWidth: '75%', alignSelf: isUser ? 'flex-end' : 'flex-start' }}>
      <Flex
        gap={12}
        align="flex-start"
        style={{
          flexDirection: isUser ? 'row-reverse' : 'row',
          animation: 'fadeInUp 0.3s ease',
        }}
      >
        <Avatar
          size={36}
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          style={{
            flexShrink: 0,
            backgroundColor: isUser ? '#1677ff' : '#b37feb',
          }}
        />
        <Card
          size="small"
          styles={{
            body: {
              padding: '10px 16px',
              background: isUser ? '#1677ff' : '#f5f5f5',
              borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
              border: 'none',
              lineHeight: 1.7,
              wordBreak: 'break-word',
              fontSize: 14,
            },
          }}
          style={
            isUser
              ? { border: 'none', background: 'transparent' }
              : { border: 'none', background: 'transparent' }
          }
        >
          {isUser ? (
            <Text style={{ color: '#fff' }}>{message.content}</Text>
          ) : message.isStreaming && !message.content ? (
            // ★ 等待首 token：显示 loading 动画
            <Flex align="center" gap={8}>
              <Spin size="small" />
              <Text type="secondary" style={{ fontSize: 13 }}>思考中…</Text>
            </Flex>
          ) : (
            <span style={{ color: 'rgba(0,0,0,0.88)' }}>
              <Markdown content={message.content} />
              {message.isStreaming && <span className="typing-cursor" />}
            </span>
          )}
        </Card>
      </Flex>
      {/* 时间戳 */}
      <Text
        type="secondary"
        style={{
          fontSize: 11,
          textAlign: isUser ? 'right' : 'left',
          paddingInline: 52,            // 留出 avatar 宽度
        }}
      >
        {new Date(message.timestamp).toLocaleTimeString('zh-CN', {
          hour: '2-digit',
          minute: '2-digit',
        })}
      </Text>
    </Flex>
  );
};

export default ChatBubble;
