import { Session } from '@/types';

export const fixtures = {
  emptySession: {
    id: 'session-empty',
    title: 'New Session',
    messages: [],
    createTime: Date.now(),
    updateTime: Date.now(),
  },
  
  activeSession: {
    id: 'session-active',
    title: 'Active Chat',
    messages: [
      {
        id: 'msg-1',
        role: 'user' as const,
        content: 'First message',
        timestamp: Date.now() - 1000,
        status: 'success' as const,
      },
      {
        id: 'msg-2',
        role: 'agent' as const,
        content: 'Agent response',
        timestamp: Date.now() - 500,
        status: 'success' as const,
      },
    ],
    createTime: Date.now() - 10000,
    updateTime: Date.now() - 500,
  },
  
  oldSession: {
    id: 'session-old',
    title: 'Old Chat',
    messages: [],
    createTime: Date.now() - 86400000,
    updateTime: Date.now() - 86400000,
  },
  
  sessionWithLongTitle: {
    id: 'session-long-title',
    title: 'This is a very long session title that should be truncated in the UI display',
    messages: [],
    createTime: Date.now(),
    updateTime: Date.now(),
  },
};

export const sessionList: Session[] = [
  fixtures.emptySession,
  fixtures.activeSession,
  fixtures.oldSession,
];
