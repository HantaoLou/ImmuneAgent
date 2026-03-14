'use client';

import React from 'react';
import { Message } from '@/types';
import { formatTime } from '@/utils/format';
import { FileAttachmentCard } from '@/components/files';
import { ExecutionLog } from './ExecutionLog';
import styles from './MessageBubble.module.css';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const hasAttachments = message.attachments && message.attachments.length > 0;
  const hasExecutionLogs = !isUser && message.executionLogs && message.executionLogs.length > 0;
  
  return (
    <div className={`${styles.messageWrapper} ${isUser ? styles.userWrapper : styles.agentWrapper}`}>
      <div className={styles.messageContainer}>
        <div className={`${styles.messageBubble} ${isUser ? styles.userBubble : styles.agentBubble}`}>
          {message.content && (
            <div className={styles.messageContent}>
              {message.content}
            </div>
          )}
          
          {hasAttachments && (
            <div className={styles.attachments}>
              {message.attachments!.map((file) => (
                <FileAttachmentCard
                  key={file.id}
                  file={file}
                  showDownload
                  compact
                />
              ))}
            </div>
          )}
          
          {hasExecutionLogs && (
            <ExecutionLog logs={message.executionLogs!} />
          )}
        </div>
        <div className={`${styles.timestamp} ${isUser ? styles.timestampRight : styles.timestampLeft}`}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};
