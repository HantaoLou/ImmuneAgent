import React from 'react';
import clsx from 'clsx';
import { Trash2 } from 'lucide-react';
import { Session } from '@/lib/types';

interface SessionItemProps {
  session: Session;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

export function SessionItem({ session, isActive, onSelect, onDelete }: SessionItemProps) {
  const timeAgo = getTimeAgo(session.createdAt);

  return (
    <div
      className={clsx(
        'group flex items-center justify-between p-3 cursor-pointer hover:bg-gray-100 transition-colors',
        isActive && 'bg-blue-50 border-l-2 border-blue-600'
      )}
      onClick={onSelect}
    >
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {session.title}
        </p>
        <p className="text-xs text-gray-500">{timeAgo}</p>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity"
        aria-label="Delete session"
      >
        <Trash2 className="h-4 w-4 text-red-600" />
      </button>
    </div>
  );
}

function getTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} min ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  return date.toLocaleDateString();
}
