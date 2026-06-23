import React from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadProps, RcFile } from 'antd/es/upload';

const { Dragger } = Upload;

interface FileUploadProps {
  onUpload: (file: File) => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUpload, disabled = false }) => {
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: '.pdf,.txt',
    showUploadList: false,
    disabled,
    beforeUpload: (file: RcFile): boolean | typeof Upload.LIST_IGNORE => {
      const isValidType: boolean =
        file.type === 'application/pdf' ||
        file.type === 'text/plain' ||
        file.name.toLowerCase().endsWith('.txt');

      if (!isValidType) {
        message.error('仅支持 .pdf 和 .txt 格式文件');
        return Upload.LIST_IGNORE;
      }

      const isLt20M: boolean = file.size / 1024 / 1024 <= 20;
      if (!isLt20M) {
        message.error('文件大小不能超过 20MB');
        return Upload.LIST_IGNORE;
      }
      
      onUpload(file as File);
      return false;
    },
  };

  return (
    <Dragger {...uploadProps}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
      <p className="ant-upload-hint">支持 .pdf / .txt 格式，大小不超过 20MB</p>
    </Dragger>
  );
};

export default FileUpload;
