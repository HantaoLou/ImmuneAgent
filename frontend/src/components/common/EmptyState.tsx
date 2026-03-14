'use client';

import React from 'react';
import styles from './EmptyState.module.css';

interface EmptyStateProps {
  tip?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  tip = '暂无数据',
}) => {
  return (
    <div className={styles.emptyState}>
      <div className={styles.icon}>💬</div>
      <div className={styles.content}>
        <div className={styles.title}>开始对话</div>
        <div className={styles.tip}>{tip}</div>
      </div>
      <div className={styles.glow}></div>
    </div>
  );
};
