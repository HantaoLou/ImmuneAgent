'use client';

import React from 'react';
import { Message } from '@/types';
import { MessageBubble } from './MessageBubble';
import { LoadingTyping } from './LoadingTyping';

interface MessageListProps {
  messages: Message[];
}

export const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  return (
    <div className="space-y-2">
      {messages.map((message) => {
        if (message.status === 'loading') {
          const hasExecutionLogs = message.executionLogs && message.executionLogs.length > 0;
          if (hasExecutionLogs) {
            return <MessageBubble key={message.id} message={message} />;
          }
          return <LoadingTyping key={message.id} />;
        }
        return <MessageBubble key={message.id} message={message} />;
      })}
    </div>
  );
};
