'use client';

import React, { useRef } from 'react';
import { Button, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import styles from './FileUploadButton.module.css';

interface FileUploadButtonProps {
  sessionId: string;
  onFilesSelected: (files: FileList) => void;
  multiple?: boolean;
  accept?: string;
  disabled?: boolean;
}

export const FileUploadButton: React.FC<FileUploadButtonProps> = ({
  sessionId,
  onFilesSelected,
  multiple = true,
  accept,
  disabled = false,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (!disabled && inputRef.current) {
      inputRef.current.click();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFilesSelected(files);
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple={multiple}
        accept={accept}
        onChange={handleChange}
        className={styles.hiddenInput}
      />
      <Button
        icon={<UploadOutlined />}
        onClick={handleClick}
        disabled={disabled || !sessionId}
        className={styles.uploadBtn}
        type="text"
      >
        Upload File
      </Button>
    </>
  );
};
