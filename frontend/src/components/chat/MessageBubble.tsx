'use client';

import React from 'react';
import { Message } from '@/types';
import { formatTime } from '@/utils/format';
import { FileAttachmentCard } from '@/components/files';
import { ExecutionLog } from './ExecutionLog';
import { HITLBubble } from '@/components/hitl';
import styles from './MessageBubble.module.css';

interface MessageBubbleProps {
  message: Message;
  onHITLConfirm?: (sessionId: string, feedback?: string, parameters?: Record<string, any>) => void;
  onHITLReject?: (sessionId: string, feedback: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onHITLConfirm, onHITLReject }) => {
  const isUser = message.role === 'user';
  const hasAttachments = message.attachments && message.attachments.length > 0;
  const hasExecutionLogs = !isUser && message.executionLogs && message.executionLogs.length > 0;
  const hasHITLRequest = !isUser && message.hitlRequest;

  console.log('[MessageBubble] Rendering message:', {
    id: message.id,
    role: message.role,
    hasHITLRequest,
    hasExecutionLogs,
    hitlRequest: message.hitlRequest,
  });

  if (hasHITLRequest) {
    console.log('[MessageBubble] Rendering HITLBubble');
    return (
      <div className={`${styles.messageWrapper} ${styles.agentWrapper}`}>
        <div className={styles.messageContainer}>
          <HITLBubble
            hitlRequest={message.hitlRequest!}
            onConfirm={onHITLConfirm}
            onReject={onHITLReject}
          />
          <div className={`${styles.timestamp} ${styles.timestampLeft}`}>
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

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
