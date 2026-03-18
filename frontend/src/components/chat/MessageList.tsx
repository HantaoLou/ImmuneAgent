'use client';

import React from 'react';
import { Message } from '@/types';
import { MessageBubble } from './MessageBubble';
import { LoadingTyping } from './LoadingTyping';

interface MessageListProps {
  messages: Message[];
  onHITLConfirm?: (sessionId: string, feedback?: string, parameters?: Record<string, any>) => void;
  onHITLReject?: (sessionId: string, feedback: string) => void;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, onHITLConfirm, onHITLReject }) => {
  console.log('[MessageList] Rendering messages:', messages);
  return (
    <div className="space-y-2">
      {messages.map((message) => {
        console.log('[MessageList] Rendering message:', {
          id: message.id,
          role: message.role,
          status: message.status,
          hasHitlRequest: !!message.hitlRequest,
          hasExecutionLogs: !!(message.executionLogs && message.executionLogs.length > 0),
        });

        if (message.status === 'loading') {
          const hasExecutionLogs = message.executionLogs && message.executionLogs.length > 0;
          if (hasExecutionLogs) {
            return <MessageBubble key={message.id} message={message} onHITLConfirm={onHITLConfirm} onHITLReject={onHITLReject} />;
          }
          return <LoadingTyping key={message.id} />;
        }
        return <MessageBubble key={message.id} message={message} onHITLConfirm={onHITLConfirm} onHITLReject={onHITLReject} />;
      })}
    </div>
  );
};
