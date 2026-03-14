import { describe, it, expect, beforeEach, vi } from 'vitest';
import { storage } from '@/utils/storage';
import { Session } from '@/types';

describe('storage', () => {
  let localStorageMock: Record<string, string>;

  beforeEach(() => {
    localStorageMock = {};
    
    Object.defineProperty(global, 'localStorage', {
      value: {
        getItem: (key: string) => localStorageMock[key] || null,
        setItem: (key: string, value: string) => {
          localStorageMock[key] = value;
        },
        removeItem: (key: string) => {
          delete localStorageMock[key];
        },
        clear: () => {
          Object.keys(localStorageMock).forEach(key => delete localStorageMock[key]);
        },
        get length() {
          return Object.keys(localStorageMock).length;
        },
        key: (index: number) => {
          const keys = Object.keys(localStorageMock);
          return keys[index] || null;
        },
      },
      writable: true,
    });
  });

  describe('getSessions', () => {
    it('should return empty array when no sessions in storage', () => {
      const sessions = storage.getSessions();
      expect(sessions).toEqual([]);
    });

    it('should return parsed sessions from localStorage', () => {
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Test Session',
          messages: [],
          createdAt: Date.now(),
        },
      ];
      
      localStorageMock['agent-chat-sessions'] = JSON.stringify(mockSessions);
      
      const sessions = storage.getSessions();
      expect(sessions).toEqual(mockSessions);
    });

    it('should handle invalid JSON in storage', () => {
      localStorageMock['agent-chat-sessions'] = 'invalid json';
      
      expect(() => storage.getSessions()).toThrow();
    });

    it('should return empty array in non-browser environment', () => {
      const originalWindow = global.window;
      Object.defineProperty(global, 'window', {
        value: undefined,
        writable: true,
      });
      
      const sessions = storage.getSessions();
      expect(sessions).toEqual([]);
      
      Object.defineProperty(global, 'window', {
        value: originalWindow,
        writable: true,
      });
    });

    it('should handle sessions with multiple messages', () => {
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Session with Messages',
          messages: [
            {
              id: 'msg-1',
              role: 'user',
              content: 'Hello',
              timestamp: Date.now(),
              status: 'success',
            },
            {
              id: 'msg-2',
              role: 'agent',
              content: 'Hi there',
              timestamp: Date.now(),
              status: 'success',
            },
          ],
          createdAt: Date.now(),
        },
      ];
      
      localStorageMock['agent-chat-sessions'] = JSON.stringify(mockSessions);
      
      const sessions = storage.getSessions();
      expect(sessions[0].messages).toHaveLength(2);
    });
  });

  describe('saveSessions', () => {
    it('should save sessions to localStorage', () => {
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Test Session',
          messages: [],
          createdAt: Date.now(),
        },
      ];
      
      storage.saveSessions(mockSessions);
      
      expect(localStorageMock['agent-chat-sessions']).toBe(JSON.stringify(mockSessions));
    });

    it('should overwrite existing sessions', () => {
      const oldSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Old Session',
          messages: [],
          createdAt: Date.now() - 1000,
        },
      ];
      
      const newSessions: Session[] = [
        {
          id: 'session-2',
          title: 'New Session',
          messages: [],
          createdAt: Date.now(),
        },
      ];
      
      localStorageMock['agent-chat-sessions'] = JSON.stringify(oldSessions);
      
      storage.saveSessions(newSessions);
      
      const stored = JSON.parse(localStorageMock['agent-chat-sessions']);
      expect(stored).toEqual(newSessions);
      expect(stored).toHaveLength(1);
      expect(stored[0].id).toBe('session-2');
    });

    it('should save empty array', () => {
      storage.saveSessions([]);
      
      expect(localStorageMock['agent-chat-sessions']).toBe('[]');
    });

    it('should not throw in non-browser environment', () => {
      const originalWindow = global.window;
      Object.defineProperty(global, 'window', {
        value: undefined,
        writable: true,
      });
      
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Test Session',
          messages: [],
          createdAt: Date.now(),
        },
      ];
      
      expect(() => storage.saveSessions(mockSessions)).not.toThrow();
      
      Object.defineProperty(global, 'window', {
        value: originalWindow,
        writable: true,
      });
    });

    it('should handle sessions with complex message data', () => {
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Complex Session',
          messages: [
            {
              id: 'msg-1',
              role: 'user',
              content: 'Test with special chars: <>&"\'',
              timestamp: Date.now(),
              status: 'success',
              files: [
                {
                  id: 'file-1',
                  name: 'test.pdf',
                  size: 1024,
                  type: 'application/pdf',
                  url: 'blob:test',
                  sessionId: 'session-1',
                  uploadTime: Date.now(),
                  category: 'document',
                },
              ],
            },
          ],
          createdAt: Date.now(),
        },
      ];
      
      storage.saveSessions(mockSessions);
      
      const stored = JSON.parse(localStorageMock['agent-chat-sessions']);
      expect(stored[0].messages[0].files).toHaveLength(1);
      expect(stored[0].messages[0].files![0].name).toBe('test.pdf');
    });
  });

  describe('clearSessions', () => {
    it('should remove sessions from localStorage', () => {
      localStorageMock['agent-chat-sessions'] = JSON.stringify([
        { id: 'session-1', title: 'Test', messages: [], createdAt: Date.now() },
      ]);
      
      storage.clearSessions();
      
      expect(localStorageMock['agent-chat-sessions']).toBeUndefined();
    });

    it('should not throw when storage is already empty', () => {
      expect(() => storage.clearSessions()).not.toThrow();
    });

    it('should not throw in non-browser environment', () => {
      const originalWindow = global.window;
      Object.defineProperty(global, 'window', {
        value: undefined,
        writable: true,
      });
      
      expect(() => storage.clearSessions()).not.toThrow();
      
      Object.defineProperty(global, 'window', {
        value: originalWindow,
        writable: true,
      });
    });

    it('should only clear agent-chat-sessions key', () => {
      localStorageMock['agent-chat-sessions'] = '[]';
      localStorageMock['other-key'] = 'other-value';
      
      storage.clearSessions();
      
      expect(localStorageMock['agent-chat-sessions']).toBeUndefined();
      expect(localStorageMock['other-key']).toBe('other-value');
    });
  });

  describe('Integration', () => {
    it('should save and retrieve sessions correctly', () => {
      const mockSessions: Session[] = [
        {
          id: 'session-1',
          title: 'Integration Test',
          messages: [
            {
              id: 'msg-1',
              role: 'user',
              content: 'Test message',
              timestamp: Date.now(),
              status: 'success',
            },
          ],
          createdAt: Date.now(),
        },
      ];
      
      storage.saveSessions(mockSessions);
      const retrieved = storage.getSessions();
      
      expect(retrieved).toEqual(mockSessions);
    });

    it('should handle full CRUD lifecycle', () => {
      const session1: Session = {
        id: 'session-1',
        title: 'First Session',
        messages: [],
        createdAt: Date.now(),
      };
      
      const session2: Session = {
        id: 'session-2',
        title: 'Second Session',
        messages: [],
        createdAt: Date.now(),
      };
      
      storage.saveSessions([session1]);
      expect(storage.getSessions()).toHaveLength(1);
      
      storage.saveSessions([session1, session2]);
      expect(storage.getSessions()).toHaveLength(2);
      
      storage.clearSessions();
      expect(storage.getSessions()).toHaveLength(0);
    });
  });
});
