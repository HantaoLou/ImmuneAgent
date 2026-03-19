'use client';

import React, { useState, useEffect } from 'react';
import { Input, Button, Empty, Spin, message } from 'antd';
import { SearchOutlined, DownloadOutlined, CloseOutlined, ReloadOutlined, LoadingOutlined } from '@ant-design/icons';
import { useSessionStore } from '@/store/sessionStore';
import { fileService } from '@/services/fileService';
import { SandboxFile } from '@/types';
import styles from './FileManager.module.css';

interface FileManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const getFileIcon = (type: string): React.ReactNode => {
  const iconMap: Record<string, string> = {
    csv: '📊',
    json: '📋',
    txt: '📄',
    md: '📝',
    pdf: '📕',
    png: '🖼️',
    jpg: '🖼️',
    jpeg: '🖼️',
    tsv: '📊',
    fasta: '🧬',
    fa: '🧬',
    h5ad: '🔬',
    rds: '🔬',
  };
  return iconMap[type] || '📁';
};

export const FileManager: React.FC<FileManagerProps> = ({ isOpen, onClose }) => {
  const { activeSessionId } = useSessionStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [sandboxFiles, setSandboxFiles] = useState<SandboxFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<string>('');
  const [downloadingFiles, setDownloadingFiles] = useState<Set<string>>(new Set());

  const fetchFiles = async () => {
    if (!activeSessionId) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fileService.getSandboxFiles(activeSessionId);
      setSandboxFiles(response.files);
      setSource(response.source);
    } catch (err: any) {
      console.error('Failed to fetch file list:', err);
      setError(err.response?.data?.detail || 'Failed to fetch file list');
      setSandboxFiles([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && activeSessionId) {
      fetchFiles();
    }
  }, [isOpen, activeSessionId]);

  const filteredFiles = sandboxFiles.filter((file) => {
    return file.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
           file.path.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const totalSize = sandboxFiles.reduce((sum, file) => sum + file.size, 0);

  const handleDownload = async (file: SandboxFile) => {
    if (!activeSessionId) return;
    
    const downloadPath = file.relative_path || file.path;
    const fileKey = file.path;
    
    setDownloadingFiles(prev => new Set(prev).add(fileKey));
    message.loading({ content: 'Preparing download, please wait...', key: fileKey, duration: 0 });
    
    try {
      const blob = await fileService.downloadSandboxFile(activeSessionId, downloadPath);
      fileService.triggerDownload(blob, file.name);
      message.success({ content: 'Download successful', key: fileKey });
    } catch (error: any) {
      console.error('Download failed:', error);
      message.error({ content: error.response?.data?.detail || 'Download failed', key: fileKey });
    } finally {
      setDownloadingFiles(prev => {
        const next = new Set(prev);
        next.delete(fileKey);
        return next;
      });
    }
  };

  if (!isOpen) return null;

  return (
    <div className={styles.fileManager}>
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h3 className={styles.title}>Sandbox File Manager</h3>
          <button className={styles.closeBtn} onClick={onClose}>
            <CloseOutlined />
          </button>
        </div>
        <div className={styles.stats}>
          <span className={styles.count}>{sandboxFiles.length} files</span>
          <span className={styles.divider}>•</span>
          <span className={styles.size}>{formatFileSize(totalSize)}</span>
          {source && (
            <>
              <span className={styles.divider}>•</span>
              <span className={styles.source}>{source}</span>
            </>
          )}
        </div>
      </div>

      <div className={styles.toolbar}>
        <div className={styles.searchRow}>
          <Input
            placeholder="Search files..."
            prefix={<SearchOutlined />}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={styles.searchInput}
          />
          <Button
            icon={<ReloadOutlined spin={isLoading} />}
            onClick={fetchFiles}
            disabled={isLoading}
            className={styles.refreshBtn}
          />
        </div>
      </div>

      <div className={styles.fileList}>
        {isLoading ? (
          <div className={styles.loadingState}>
            <Spin size="large" />
            <p>Loading file list...</p>
          </div>
        ) : error ? (
          <div className={styles.errorState}>
            <p>{error}</p>
            <Button onClick={fetchFiles}>Retry</Button>
          </div>
        ) : filteredFiles.length > 0 ? (
          <div className={styles.fileGrid}>
            {filteredFiles.map((file, index) => (
              <div key={`${file.path}-${index}`} className={styles.fileItem}>
                <div className={styles.fileCard}>
                  <div className={styles.fileIcon}>
                    {getFileIcon(file.type)}
                  </div>
                  <div className={styles.fileInfo}>
                    <div className={styles.fileName} title={file.name}>
                      {file.name}
                    </div>
                    <div className={styles.fileMeta}>
                      <span className={styles.fileSize}>{formatFileSize(file.size)}</span>
                      <span className={styles.filePath} title={file.path}>
                        {file.path.length > 30 ? `...${file.path.slice(-27)}` : file.path}
                      </span>
                    </div>
                  </div>
                  <button
                    className={styles.downloadBtn}
                    onClick={() => handleDownload(file)}
                    disabled={downloadingFiles.has(file.path)}
                    title={downloadingFiles.has(file.path) ? 'Downloading...' : 'Download file'}
                  >
                    {downloadingFiles.has(file.path) ? <LoadingOutlined /> : <DownloadOutlined />}
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>
            <Empty
              description={searchQuery ? 'No matching files found' : 'No output files yet'}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
            {!searchQuery && (
              <p className={styles.emptyTip}>
                Generated files will appear here after task execution
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
