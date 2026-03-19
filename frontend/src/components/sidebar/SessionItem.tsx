'use client';

import React from 'react';
import { Session } from '@/types';
import { Button, Popconfirm } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import styles from './SessionItem.module.css';

interface SessionItemProps {
  session: Session;
  isActive: boolean;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}

export const SessionItem: React.FC<SessionItemProps> = ({
  session,
  isActive,
  onSelect,
  onDelete,
}) => {
  return (
    <div
      onClick={() => onSelect(session.id)}
      className={`${styles.sessionItem} ${isActive ? styles.active : ''}`}
    >
      <div className={styles.content}>
        <div className={styles.title}>{session.title}</div>
        <div className={styles.meta}>
          {session.messages.length} messages
        </div>
      </div>
      <Popconfirm
        title="Are you sure to delete this session?"
        onConfirm={(e) => {
          e?.stopPropagation();
          onDelete(session.id);
        }}
        onCancel={(e) => e?.stopPropagation()}
        okText="Yes"
        cancelText="No"
      >
        <Button
          type="text"
          icon={<DeleteOutlined />}
          size="small"
          onClick={(e) => e.stopPropagation()}
          className={styles.deleteBtn}
        />
      </Popconfirm>
    </div>
  );
};
