'use client';

import React, { useState } from 'react';
import { Input, Button, Empty } from 'antd';
import { SearchOutlined, DownloadOutlined, DeleteOutlined, CloseOutlined } from '@ant-design/icons';
import { useSessionStore } from '@/store/sessionStore';
import { FileAttachmentCard } from './FileAttachmentCard';
import { fileService } from '@/services/fileService';
import { fileUtils } from '@/lib/fileUtils';
import { FileAttachment, FileCategory } from '@/types';
import styles from './FileManager.module.css';

interface FileManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const FileManager: React.FC<FileManagerProps> = ({ isOpen, onClose }) => {
  const { activeSessionId, sessionFiles, removeFile } = useSessionStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<FileCategory | 'all'>('all');

  const files = activeSessionId ? sessionFiles[activeSessionId] || [] : [];

  const filteredFiles = files.filter((file) => {
    const matchesSearch = file.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || file.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const totalSize = files.reduce((sum, file) => sum + file.size, 0);

  const handleDownload = async (file: FileAttachment) => {
    try {
      const blob = await fileService.download({
        fileId: file.id,
        sessionId: file.sessionId,
      });
      fileService.triggerDownload(blob, file.name);
    } catch (error) {
      console.error('下载失败:', error);
    }
  };

  const handleDelete = async (fileId: string) => {
    try {
      await fileService.delete({
        fileId,
        sessionId: activeSessionId!,
      });
      removeFile(activeSessionId!, fileId);
    } catch (error) {
      console.error('删除失败:', error);
    }
  };

  const handleBatchDownload = async () => {
    if (filteredFiles.length === 0) return;
    
    try {
      const blob = await fileService.batchDownload({
        fileIds: filteredFiles.map(f => f.id),
        sessionId: activeSessionId!,
      });
      fileService.triggerDownload(blob, `files-${activeSessionId}.zip`);
    } catch (error) {
      console.error('批量下载失败:', error);
    }
  };

  const categories: Array<{ key: FileCategory | 'all'; label: string; color: string }> = [
    { key: 'all', label: '全部', color: 'var(--text-primary)' },
    { key: 'image', label: '图片', color: 'var(--accent-cyan)' },
    { key: 'document', label: '文档', color: 'var(--accent-purple)' },
    { key: 'code', label: '代码', color: 'var(--accent-lime)' },
    { key: 'data', label: '数据', color: 'var(--accent-gold)' },
  ];

  if (!isOpen) return null;

  return (
    <div className={styles.fileManager}>
      {/* 头部 */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h3 className={styles.title}>文件管理器</h3>
          <button className={styles.closeBtn} onClick={onClose}>
            <CloseOutlined />
          </button>
        </div>
        <div className={styles.stats}>
          <span className={styles.count}>{files.length} 个文件</span>
          <span className={styles.divider}>•</span>
          <span className={styles.size}>{fileUtils.formatFileSize(totalSize)}</span>
        </div>
      </div>

      {/* 搜索和过滤 */}
      <div className={styles.toolbar}>
        <Input
          placeholder="搜索文件..."
          prefix={<SearchOutlined />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
        <div className={styles.categories}>
          {categories.map((cat) => (
            <button
              key={cat.key}
              className={`${styles.categoryBtn} ${selectedCategory === cat.key ? styles.active : ''}`}
              style={{ 
                borderColor: selectedCategory === cat.key ? cat.color : 'var(--glass-border)',
                color: selectedCategory === cat.key ? cat.color : 'var(--text-secondary)'
              }}
              onClick={() => setSelectedCategory(cat.key)}
            >
              {cat.label}
            </button>
          ))}
        </div>
        {filteredFiles.length > 0 && (
          <Button
            icon={<DownloadOutlined />}
            onClick={handleBatchDownload}
            className={styles.batchBtn}
            block
          >
            批量下载
          </Button>
        )}
      </div>

      {/* 文件列表 */}
      <div className={styles.fileList}>
        {filteredFiles.length > 0 ? (
          <div className={styles.fileGrid}>
            {filteredFiles.map((file) => (
              <div key={file.id} className={styles.fileItem}>
                <FileAttachmentCard
                  file={file}
                  showDownload
                  showDelete
                  onDelete={() => handleDelete(file.id)}
                  compact
                />
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>
            <Empty
              description={searchQuery || selectedCategory !== 'all' ? '没有找到匹配的文件' : '暂无文件'}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          </div>
        )}
      </div>
    </div>
  );
};
