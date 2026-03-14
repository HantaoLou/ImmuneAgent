'use client';

import React from 'react';
import { Button } from 'antd';
import { DownloadOutlined, DeleteOutlined } from '@ant-design/icons';
import { FileAttachment } from '@/types';
import { FileIcon } from './FileIcon';
import { fileUtils } from '@/lib/fileUtils';
import { fileService } from '@/services/fileService';
import styles from './FileAttachmentCard.module.css';

interface FileAttachmentCardProps {
  file: FileAttachment;
  showDownload?: boolean;
  showDelete?: boolean;
  onDelete?: () => void;
  compact?: boolean;
}

export const FileAttachmentCard: React.FC<FileAttachmentCardProps> = ({
  file,
  showDownload = true,
  showDelete = false,
  onDelete,
  compact = false,
}) => {
  const truncatedName = fileUtils.truncateFileName(file.name);
  const formattedSize = fileUtils.formatFileSize(file.size);

  const handleDownload = async () => {
    try {
      const blob = await fileService.download({
        fileId: file.id,
        sessionId: file.sessionId,
      });
      fileService.triggerDownload(blob, file.name);
    } catch (error) {
      console.error('文件下载失败:', error);
    }
  };

  return (
    <div className={`${styles.card} ${compact ? styles.compact : ''}`}>
      <div className={styles.icon}>
        <FileIcon category={file.category} size={compact ? 'small' : 'medium'} />
      </div>
      
      <div className={styles.info}>
        <div className={styles.name} title={file.name}>
          {truncatedName}
        </div>
        <div className={styles.meta}>
          <span className={styles.size}>{formattedSize}</span>
          <span className={styles.dot}>•</span>
          <span className={styles.time}>
            {new Date(file.uploadTime).toLocaleString('zh-CN')}
          </span>
        </div>
      </div>
      
      <div className={styles.actions}>
        {showDownload && (
          <Button
            icon={<DownloadOutlined />}
            onClick={handleDownload}
            className={styles.actionBtn}
            type="text"
            size="small"
          />
        )}
        {showDelete && onDelete && (
          <Button
            icon={<DeleteOutlined />}
            onClick={onDelete}
            className={`${styles.actionBtn} ${styles.deleteBtn}`}
            type="text"
            size="small"
            danger
          />
        )}
      </div>
    </div>
  );
};
