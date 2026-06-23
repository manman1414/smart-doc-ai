import React from 'react';
import { Card, Button, Space, Typography, Spin, Alert } from 'antd';
import { RobotOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { isSummaryError, formatSummaryErrorMessage } from '@/services/api';

const { Text } = Typography;

interface AISummaryCardProps {
  summary: string;
  streaming: boolean;
  suggestedQuestions?: string[];
  onAskQuestion?: (question: string) => void;
}

const AISummaryCard: React.FC<AISummaryCardProps> = ({
  summary,
  streaming,
  suggestedQuestions = [],
  onAskQuestion,
}) => {
  const handleAskClick = (q: string): void => {
    onAskQuestion?.(q);
  };

  const summaryFailed = !streaming && !!summary && isSummaryError(summary);
  const summaryDetail = summaryFailed ? formatSummaryErrorMessage(summary) : summary;

  return (
    <div>
      <Card
        size="small"
        title={
          <Space>
            <RobotOutlined style={{ color: '#1677ff' }} />
            <span>AI 摘要</span>
          </Space>
        }
        style={{
          marginBottom: 12,
          background: summaryFailed
            ? '#fffbe6'
            : 'linear-gradient(135deg, #f0f5ff 0%, #e6f4ff 100%)',
          borderColor: summaryFailed ? '#ffe58f' : '#bae0ff',
        }}
      >
        <div style={{
          lineHeight: 1.8,
          maxHeight: 200,
          overflowY: 'auto',
          fontSize: 13,
        }}>
          {streaming ? (
            <Space>
              <Spin size="small" />
              <Text type="secondary">正在生成摘要...</Text>
            </Space>
          ) : summaryFailed ? (
            <Alert
              type="warning"
              showIcon
              message="摘要生成失败"
              description={summaryDetail}
              style={{ padding: '4px 8px' }}
            />
          ) : summary ? (
            <Text>{summary}</Text>
          ) : (
            <Text type="secondary">等待文档上传...</Text>
          )}
        </div>
      </Card>

      {!streaming && suggestedQuestions.length > 0 && (
        <Card
          size="small"
          title={
            <Space>
              <QuestionCircleOutlined style={{ color: '#1677ff' }} />
              <span>试试这些问题</span>
            </Space>
          }
          style={{ background: '#fafafa' }}
        >
          <Space direction="vertical" style={{ width: '100%' }} size={6}>
            {suggestedQuestions.map((q: string) => (
              <Button
                key={q}
                type="default"
                block
                size="small"
                onClick={() => handleAskClick(q)}
                style={{
                  textAlign: 'left',
                  height: 'auto',
                  padding: '8px 12px',
                  whiteSpace: 'normal',
                  wordBreak: 'break-word',
                }}
              >
                {q}
              </Button>
            ))}
          </Space>
        </Card>
      )}
    </div>
  );
};

export default AISummaryCard;
