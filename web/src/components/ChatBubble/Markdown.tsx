import React from 'react';
import { Typography } from 'antd';

const { Paragraph, Text } = Typography;

interface MarkdownProps {
  content: string;
}

/** 渲染行内加粗 **text** */
function renderInline(text: string): React.ReactNode {
  const parts: string[] = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part: string, i: number) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <Text strong key={i}>{part.slice(2, -2)}</Text>;
    }
    return <Text key={i}>{part}</Text>;
  });
}

const Markdown: React.FC<MarkdownProps> = ({ content }) => {
  const lines: string[] = content.split('\n');
  const elements: React.ReactNode[] = [];
  let i: number = 0;

  while (i < lines.length) {
    const line: string = lines[i];

    // 空行 → 段落间距
    if (line.trim() === '') {
      elements.push(<div key={i} style={{ height: 8 }} />);
      i++;
      continue;
    }

    // 无序列表
    if (/^[-*•]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*•]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*•]\s+/, ''));
        i++;
      }
      elements.push(
        <ul key={i} style={{ margin: '4px 0', paddingLeft: 18 }}>
          {items.map((item: string, idx: number) => (
            <li key={idx} style={{ marginBottom: 2 }}>
              <Text>{renderInline(item)}</Text>
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // 普通段落
    elements.push(
      <Paragraph
        key={i}
        style={{ marginBottom: 2 }}
      >
        {renderInline(line)}
      </Paragraph>,
    );
    i++;
  }

  return <>{elements}</>;
};

export default Markdown;
