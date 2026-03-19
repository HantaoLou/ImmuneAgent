'use client';

import React, { useState, useCallback } from 'react';
import { FileUploadButton } from './FileUploadButton';
import styles from './FileDropZone.module.css';

interface FileDropZoneProps {
  sessionId: string;
  onFilesSelected: (files: FileList) => void;
  disabled?: boolean;
  children?: React.ReactNode;
}

export const FileDropZone: React.FC<FileDropZoneProps> = ({
  sessionId,
  onFilesSelected,
  disabled = false,
  children,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFilesSelected(files);
    }
  }, [disabled, onFilesSelected]);

  return (
    <div
      className={`${styles.dropZone} ${isDragOver ? styles.dragOver : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {children || (
        <div className={styles.defaultContent}>
          <div className={styles.icon}>📁</div>
          <div className={styles.text}>
            Drag and drop files here, or
            <FileUploadButton
              sessionId={sessionId}
              onFilesSelected={onFilesSelected}
              disabled={disabled}
            />
          </div>
          <div className={styles.hint}>
            Supports images, documents, code, and data files (max 10MB)
          </div>
        </div>
      )}
    </div>
  );
};
