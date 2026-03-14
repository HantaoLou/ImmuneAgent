'use client';

import React from 'react';
import styles from './LoadingTyping.module.css';

export const LoadingTyping: React.FC = () => {
  return (
    <div className={styles.typingIndicator}>
      <span className={styles.dot}></span>
      <span className={`${styles.dot} ${styles.dotPurple}`}></span>
      <span className={`${styles.dot} ${styles.dotLime}`}></span>
    </div>
  );
};
