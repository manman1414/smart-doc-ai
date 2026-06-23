import React from 'react';
import { Card, Tag, Space, Typography } from 'antd';
import {
  FilePdfOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import type { DocInfo } from '@/types';

const { Text } = Typography;

interface DocInfoCardProps {
  doc: DocInfo;
  /** 是否仍在处理中 */
  processing?: boolean;
  /** 摘要是否成功（false 表示向量化可能已完成，但摘要失败） */
  summaryOk?: boolean;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const DocInfoCard: React.FC<DocInfoCardProps> = ({ doc, processing = false, summaryOk = true }) => {
  const isPdf: boolean = doc.name.toLowerCase().endsWith('.pdf');

  const statusTag = processing ? (
    <Tag icon={<LoadingOutlined spin />} color="processing">解析中…</Tag>
  ) : summaryOk ? (
    <Tag icon={<CheckCircleOutlined />} color="success">解析完成</Tag>
  ) : (
    <Tag icon={<WarningOutlined />} color="warning">摘要失败</Tag>
  );

  return (
    <Card
      size="small"
      title={
        <Space>
          {isPdf ? (
            <FilePdfOutlined style={{ color: '#ff4d4f' }} />
          ) : (
            <FileTextOutlined style={{ color: '#1677ff' }} />
          )}
          <span>文档信息</span>
        </Space>
      }
    >
      <Text
        strong
        ellipsis={{ tooltip: doc.name }}
        style={{ display: 'block', marginBottom: 8, fontSize: 14 }}
      >
        {doc.name}
      </Text>
      <Space size="middle" wrap>
        <Text type="secondary" style={{ fontSize: 12 }}>
          大小 {formatSize(doc.size)}
        </Text>
        {statusTag}
      </Space>
      {!processing && !summaryOk && (
        <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
          文档已准备好，可以直接提问；启动 LM Studio 后发送问题将自动重试生成摘要。
        </Text>
      )}
    </Card>
  );
};

export default DocInfoCard;
