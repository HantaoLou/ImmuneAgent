'use client';

import React from 'react';
import { FileCategory } from '@/types';
import styles from './FileIcon.module.css';

interface FileIconProps {
  category: FileCategory;
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

const CATEGORY_ICONS: Record<FileCategory, string> = {
  image: '🖼️',
  document: '📄',
  code: '💻',
  data: '📊',
  other: '📎',
};

const CATEGORY_COLORS: Record<FileCategory, string> = {
  image: 'var(--accent-cyan)',
  document: 'var(--accent-purple)',
  code: 'var(--accent-lime)',
  data: 'var(--accent-gold)',
  other: 'var(--text-muted)',
};

export const FileIcon: React.FC<FileIconProps> = ({
  category,
  size = 'medium',
  className = '',
}) => {
  return (
    <div
      className={`${styles.icon} ${styles[size]} ${className}`}
      style={{ color: CATEGORY_COLORS[category] }}
    >
      {CATEGORY_ICONS[category]}
    </div>
  );
};
