import React from 'react';
import { Session } from '@/lib/types';
import { SessionItem } from './SessionItem';

interface SessionListProps {
  sessions: Session[];
  activeSessionId?: string;
  onSessionSelect: (sessionId: string) => void;
  onSessionDelete: (sessionId: string) => void;
}

export function SessionList({
  sessions,
  activeSessionId,
  onSessionSelect,
  onSessionDelete,
}: SessionListProps) {
  if (sessions.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        No sessions yet. Start a new chat!
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {sessions.map((session) => (
        <SessionItem
          key={session.id}
          session={session}
          isActive={session.id === activeSessionId}
          onSelect={() => onSessionSelect(session.id)}
          onDelete={() => onSessionDelete(session.id)}
        />
      ))}
    </div>
  );
}
