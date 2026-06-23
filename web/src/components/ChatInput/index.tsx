import React, { useState, useCallback } from 'react';
import { Input, Button, Space, Typography, Flex } from 'antd';
import { SendOutlined, StopOutlined } from '@ant-design/icons';

const { TextArea } = Input;
const { Text } = Typography;

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  streaming = false,
  disabled = false,
}) => {
  const [value, setValue] = useState<string>('');

  const handleSend = useCallback((): void => {
    const text: string = value.trim();
    if (!text || streaming) return;
    onSend(text);
    setValue('');
  }, [value, streaming, onSend]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
      setValue(e.target.value);
    },
    [],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const isSendDisabled: boolean = !value.trim() || disabled || streaming;

  return (
    <div style={{ padding: '12px 24px', borderTop: '1px solid #f0f0f0', background: '#fff' }}>
      <Space.Compact style={{ width: '100%', display: 'flex' }}>
        <TextArea
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="💬 输入你的问题…  Enter 发送 / Shift+Enter 换行"
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={disabled || streaming}
          style={{ flex: 1 }}
        />
        {streaming ? (
          <Button icon={<StopOutlined />} onClick={onStop} danger>
            停止
          </Button>
        ) : (
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            disabled={isSendDisabled}
          >
            发送
          </Button>
        )}
      </Space.Compact>
      <Flex justify="center" style={{ marginTop: 6 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          Enter 发送 · Shift+Enter 换行 · 支持 Markdown 渲染
        </Text>
      </Flex>
    </div>
  );
};

export default ChatInput;
