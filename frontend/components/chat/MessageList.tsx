import React, { useEffect, useRef } from 'react';
import { Message } from '@/lib/types';
import { MessageBubble } from './MessageBubble';
import { StreamingMessage } from './StreamingMessage';

interface MessageListProps {
  messages: Message[];
  isStreaming?: boolean;
  streamingStatus?: string;
}

export function MessageList({ messages, isStreaming, streamingStatus }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <div className="text-6xl mb-4">🧬</div>
        <h2 className="text-2xl font-bold mb-2">Welcome to Bio-Agent</h2>
        <p className="text-center max-w-md">
          Ask me anything about bioinformatics, immunology, or data analysis.
          I can help you analyze data, run predictions, and generate insights.
        </p>
      </div>
    );
  }

  return (
    <>
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isStreaming && <StreamingMessage status={streamingStatus} />}
      <div ref={bottomRef} />
    </>
  );
}
