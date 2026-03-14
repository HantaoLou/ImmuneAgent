'use client';

import React from 'react';
import { Session } from '@/types';
import { SessionItem } from './SessionItem';
import { EmptyState } from '@/components/common/EmptyState';

interface SessionListProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
  onSessionDelete: (sessionId: string) => void;
}

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  activeSessionId,
  onSessionSelect,
  onSessionDelete,
}) => {
  if (sessions.length === 0) {
    return <EmptyState tip="暂无会话，点击新建会话开始聊天" />;
  }

  return (
    <div className="flex-1 overflow-y-auto p-3 space-y-2">
      {sessions.map((session) => (
        <SessionItem
          key={session.id}
          session={session}
          isActive={session.id === activeSessionId}
          onSelect={onSessionSelect}
          onDelete={onSessionDelete}
        />
      ))}
    </div>
  );
};
