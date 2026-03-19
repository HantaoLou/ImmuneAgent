'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button, message } from 'antd';
import { SendOutlined, ClearOutlined } from '@ant-design/icons';
import { FileAttachment } from '@/types';
import { FileUploadButton, FilePreview } from '@/components/files';
import { useSessionStore } from '@/store/sessionStore';
import { fileService } from '@/services/fileService';
import { fileUtils } from '@/lib/fileUtils';
import styles from './InputPanel.module.css';

interface InputPanelProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onClear: () => void;
  disabled?: boolean;
  attachments?: FileAttachment[];
  onAttachmentsChange?: (files: FileAttachment[]) => void;
}

export const InputPanel: React.FC<InputPanelProps> = ({
  value,
  onChange,
  onSend,
  onClear,
  disabled = false,
  attachments = [],
  onAttachmentsChange,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [uploadingCount, setUploadingCount] = useState(0);
  const { activeSessionId } = useSessionStore();

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleClear = () => {
    onChange('');
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleFilesSelected = async (files: FileList) => {
    if (!activeSessionId) {
      message.warning('请先创建会话');
      return;
    }

    const newAttachments: FileAttachment[] = [];
    setUploadingCount(files.length);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      const validation = fileUtils.validateFile(file);
      if (!validation.valid) {
        message.error(`${file.name}: ${validation.error}`);
        setUploadingCount(prev => prev - 1);
        continue;
      }

      try {
        const result = await fileService.upload({
          file,
          sessionId: activeSessionId,
          onProgress: (progress) => {
            // 可选：显示上传进度
          },
        });

        const attachment: FileAttachment = {
          id: result.fileId,
          name: result.filename,
          size: result.size,
          type: result.mimeType,
          url: result.url,
          sessionId: result.sessionId,
          uploadTime: Date.now(),
          category: fileUtils.getCategory(file),
          sandboxPath: (result as any).sandboxPath,
          localPath: (result as any).localPath,
        };

        newAttachments.push(attachment);
        message.success(`文件 ${file.name} 上传成功`);
      } catch (error) {
        message.error(`文件 ${file.name} 上传失败`);
      } finally {
        setUploadingCount(prev => prev - 1);
      }
    }

    if (onAttachmentsChange && newAttachments.length > 0) {
      onAttachmentsChange([...attachments, ...newAttachments]);
    }
  };

  const handleRemoveAttachment = (fileId: string) => {
    if (onAttachmentsChange) {
      onAttachmentsChange(attachments.filter(f => f.id !== fileId));
    }
  };

  const canSend = !disabled && (value.trim() || attachments.length > 0) && uploadingCount === 0;

  return (
    <div className={styles.inputPanel}>
      {/* 文件预览区域 */}
      {attachments.length > 0 && (
        <div className={styles.attachments}>
          {attachments.map(file => (
            <FilePreview
              key={file.id}
              file={file}
              onRemove={() => handleRemoveAttachment(file.id)}
              compact
            />
          ))}
        </div>
      )}

      {/* 上传提示 */}
      {uploadingCount > 0 && (
        <div className={styles.uploadingHint}>
          正在上传 {uploadingCount} 个文件...
        </div>
      )}

      {/* 输入框 */}
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="输入消息... (Enter 发送, Shift+Enter 换行)"
        disabled={disabled}
        className={styles.textarea}
        rows={3}
      />
      
      {/* 控制按钮 */}
      <div className={styles.controls}>
        <div className={styles.leftControls}>
          <FileUploadButton
            sessionId={activeSessionId || ''}
            onFilesSelected={handleFilesSelected}
            disabled={disabled || uploadingCount > 0}
            multiple
          />
          <Button
            icon={<ClearOutlined />}
            onClick={handleClear}
            disabled={disabled || (!value && attachments.length === 0)}
            className={styles.clearBtn}
            type="text"
          >
            清空
          </Button>
        </div>
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={onSend}
          disabled={!canSend}
          className={styles.sendBtn}
        >
          发送
        </Button>
      </div>
    </div>
  );
};
