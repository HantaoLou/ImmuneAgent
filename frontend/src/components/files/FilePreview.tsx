'use client';

import React from 'react';
import { FileAttachment } from '@/types';
import { FileIcon } from './FileIcon';
import { fileUtils } from '@/lib/fileUtils';
import styles from './FilePreview.module.css';

interface FilePreviewProps {
  file: FileAttachment;
  onRemove?: () => void;
  compact?: boolean;
}

export const FilePreview: React.FC<FilePreviewProps> = ({
  file,
  onRemove,
  compact = false,
}) => {
  const isImage = file.category === 'image' && file.url;
  const truncatedName = fileUtils.truncateFileName(file.name);
  const formattedSize = fileUtils.formatFileSize(file.size);

  return (
    <div className={`${styles.preview} ${compact ? styles.compact : ''}`}>
      {isImage ? (
        <div className={styles.imagePreview}>
          <img src={file.url} alt={file.name} className={styles.image} />
        </div>
      ) : (
        <div className={styles.iconWrapper}>
          <FileIcon category={file.category} size={compact ? 'small' : 'medium'} />
        </div>
      )}
      
      <div className={styles.info}>
        <div className={styles.name} title={file.name}>
          {truncatedName}
        </div>
        <div className={styles.size}>{formattedSize}</div>
        {file.uploadProgress !== undefined && file.uploadProgress < 100 && (
          <div className={styles.progress}>
            <div 
              className={styles.progressBar} 
              style={{ width: `${file.uploadProgress}%` }}
            />
          </div>
        )}
      </div>
      
      {onRemove && (
        <button 
          className={styles.removeBtn}
          onClick={onRemove}
          aria-label="Remove file"
        >
          ✕
        </button>
      )}
    </div>
  );
};
