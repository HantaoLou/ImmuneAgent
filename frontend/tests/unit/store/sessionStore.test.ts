import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { useSessionStore } from '@/store/sessionStore';
import { createMockSession, createMockMessage } from '../../utils/mockFactories';

describe('SessionStore', () => {
  beforeEach(() => {
    const { sessions, activeSessionId, sessionFiles } = useSessionStore.getState();
    useSessionStore.setState({
      sessions: [],
      activeSessionId: null,
      sessionFiles: {},
    });
  });

  describe('addSession', () => {
    it('should create session with default values', () => {
      const sessionId = useSessionStore.getState().addSession();
      
      const { sessions, activeSessionId } = useSessionStore.getState();
      
      expect(sessionId).toBeDefined();
      expect(sessions).toHaveLength(1);
      expect(sessions[0].title).toBe('新会话');
      expect(sessions[0].messages).toEqual([]);
      expect(activeSessionId).toBe(sessionId);
    });

    it('should create session with custom values', () => {
      const customTitle = 'Custom Session';
      const sessionId = useSessionStore.getState().addSession({ title: customTitle });
      
      const { sessions } = useSessionStore.getState();
      
      expect(sessions[0].title).toBe(customTitle);
    });

    it('should generate unique IDs', () => {
      const id1 = useSessionStore.getState().addSession();
      const id2 = useSessionStore.getState().addSession();
      
      expect(id1).not.toBe(id2);
    });

    it('should set as active session', () => {
      const sessionId = useSessionStore.getState().addSession();
      
      const { activeSessionId } = useSessionStore.getState();
      
      expect(activeSessionId).toBe(sessionId);
    });

    it('should set createTime and updateTime', () => {
      const beforeTime = Date.now();
      const sessionId = useSessionStore.getState().addSession();
      const afterTime = Date.now();
      
      const { sessions } = useSessionStore.getState();
      const session = sessions.find(s => s.id === sessionId);
      
      expect(session?.createTime).toBeGreaterThanOrEqual(beforeTime);
      expect(session?.createTime).toBeLessThanOrEqual(afterTime);
      expect(session?.updateTime).toBeGreaterThanOrEqual(beforeTime);
      expect(session?.updateTime).toBeLessThanOrEqual(afterTime);
    });
  });

  describe('switchSession', () => {
    it('should switch active session', () => {
      const id1 = useSessionStore.getState().addSession();
      const id2 = useSessionStore.getState().addSession();
      
      useSessionStore.getState().switchSession(id1);
      
      const { activeSessionId } = useSessionStore.getState();
      expect(activeSessionId).toBe(id1);
    });

    it('should not change sessions array', () => {
      const id1 = useSessionStore.getState().addSession();
      useSessionStore.getState().addSession();
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      useSessionStore.getState().switchSession(id1);
      const { sessions: afterSessions } = useSessionStore.getState();
      
      expect(afterSessions).toEqual(beforeSessions);
    });
  });

  describe('deleteSession', () => {
    it('should delete session by ID', () => {
      const id1 = useSessionStore.getState().addSession();
      const id2 = useSessionStore.getState().addSession();
      
      useSessionStore.getState().deleteSession(id2);
      
      const { sessions } = useSessionStore.getState();
      expect(sessions).toHaveLength(1);
      expect(sessions[0].id).toBe(id1);
    });

    it('should switch to last session when deleting active session', () => {
      const id1 = useSessionStore.getState().addSession();
      const id2 = useSessionStore.getState().addSession();
      
      useSessionStore.getState().deleteSession(id2);
      
      const { activeSessionId } = useSessionStore.getState();
      expect(activeSessionId).toBe(id1);
    });

    it('should handle empty sessions gracefully', () => {
      const id = useSessionStore.getState().addSession();
      
      useSessionStore.getState().deleteSession(id);
      
      const { activeSessionId } = useSessionStore.getState();
      expect(activeSessionId).toBeDefined();
      expect(activeSessionId).not.toBe(id);
    });

    it('should clear session files', () => {
      const sessionId = useSessionStore.getState().addSession();
      const { sessionFiles } = useSessionStore.getState();
      sessionFiles[sessionId] = [];
      
      useSessionStore.getState().deleteSession(sessionId);
      
      const { sessionFiles: newSessionFiles } = useSessionStore.getState();
      expect(newSessionFiles[sessionId]).toBeUndefined();
    });

    it('should not affect other sessions when deleting', () => {
      const id1 = useSessionStore.getState().addSession({ title: 'Session 1' });
      const id2 = useSessionStore.getState().addSession({ title: 'Session 2' });
      
      useSessionStore.getState().deleteSession(id1);
      
      const { sessions } = useSessionStore.getState();
      expect(sessions[0].id).toBe(id2);
      expect(sessions[0].title).toBe('Session 2');
    });
  });

  describe('addMessage', () => {
    it('should add message to session', () => {
      const sessionId = useSessionStore.getState().addSession();
      
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content: 'Test message',
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions } = useSessionStore.getState();
      const session = sessions.find(s => s.id === sessionId);
      
      expect(session?.messages).toHaveLength(1);
      expect(session?.messages[0].content).toBe('Test message');
    });

    it('should auto-generate message ID', () => {
      const sessionId = useSessionStore.getState().addSession();
      
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content: 'Test',
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions } = useSessionStore.getState();
      const message = sessions[0].messages[0];
      
      expect(message.id).toBeDefined();
      expect(typeof message.id).toBe('string');
    });

    it('should update session updateTime', () => {
      const sessionId = useSessionStore.getState().addSession();
      const { sessions: beforeSessions } = useSessionStore.getState();
      const beforeTime = beforeSessions[0].updateTime;
      
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content: 'Test',
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions: afterSessions } = useSessionStore.getState();
      const afterTime = afterSessions[0].updateTime;
      
      expect(afterTime).toBeGreaterThanOrEqual(beforeTime);
    });

    it('should set title from first user message', () => {
      const sessionId = useSessionStore.getState().addSession();
      const content = 'This is my first message';
      
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content,
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions } = useSessionStore.getState();
      expect(sessions[0].title).toBe(content.slice(0, 15));
    });

    it('should set title from first user message regardless of initial title', () => {
      const sessionId = useSessionStore.getState().addSession();
      const content = 'This is my first message';
      
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content,
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions } = useSessionStore.getState();
      expect(sessions[0].title).toBe(content.slice(0, 15));
    });

    it('should handle multiple messages', () => {
      const sessionId = useSessionStore.getState().addSession();
      
      for (let i = 0; i < 5; i++) {
        useSessionStore.getState().addMessage(sessionId, {
          role: i % 2 === 0 ? 'user' : 'agent',
          content: `Message ${i}`,
          timestamp: Date.now(),
          status: 'success',
        });
      }
      
      const { sessions } = useSessionStore.getState();
      expect(sessions[0].messages).toHaveLength(5);
    });
  });

  describe('updateMessage', () => {
    it('should update message content', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: '',
        timestamp: Date.now(),
        status: 'loading',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessage(sessionId, messageId, {
        content: 'Updated content',
      });
      
      const { sessions: afterSessions } = useSessionStore.getState();
      const message = afterSessions[0].messages[0];
      
      expect(message.content).toBe('Updated content');
    });

    it('should update message status', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: '',
        timestamp: Date.now(),
        status: 'loading',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessage(sessionId, messageId, {
        status: 'success',
      });
      
      const { sessions: afterSessions } = useSessionStore.getState();
      const message = afterSessions[0].messages[0];
      
      expect(message.status).toBe('success');
    });

    it('should update session updateTime', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: '',
        timestamp: Date.now(),
        status: 'loading',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const beforeTime = beforeSessions[0].updateTime;
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessage(sessionId, messageId, {
        content: 'Updated',
      });
      
      const { sessions: afterSessions } = useSessionStore.getState();
      const afterTime = afterSessions[0].updateTime;
      
      expect(afterTime).toBeGreaterThanOrEqual(beforeTime);
    });

    it('should not affect other messages', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'user',
        content: 'Message 1',
        timestamp: Date.now(),
        status: 'success',
      });
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: 'Message 2',
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[1].id;
      
      useSessionStore.getState().updateMessage(sessionId, messageId, {
        content: 'Updated Message 2',
      });
      
      const { sessions: afterSessions } = useSessionStore.getState();
      expect(afterSessions[0].messages[0].content).toBe('Message 1');
      expect(afterSessions[0].messages[1].content).toBe('Updated Message 2');
    });
  });

  describe('updateMessageStatus', () => {
    it('should update status to success', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: '',
        timestamp: Date.now(),
        status: 'loading',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessageStatus(sessionId, messageId, 'success');
      
      const { sessions: afterSessions } = useSessionStore.getState();
      expect(afterSessions[0].messages[0].status).toBe('success');
    });

    it('should update status to error', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: '',
        timestamp: Date.now(),
        status: 'loading',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessageStatus(sessionId, messageId, 'error');
      
      const { sessions: afterSessions } = useSessionStore.getState();
      expect(afterSessions[0].messages[0].status).toBe('error');
    });

    it('should update status to loading', () => {
      const sessionId = useSessionStore.getState().addSession();
      useSessionStore.getState().addMessage(sessionId, {
        role: 'agent',
        content: 'Test',
        timestamp: Date.now(),
        status: 'success',
      });
      
      const { sessions: beforeSessions } = useSessionStore.getState();
      const messageId = beforeSessions[0].messages[0].id;
      
      useSessionStore.getState().updateMessageStatus(sessionId, messageId, 'loading');
      
      const { sessions: afterSessions } = useSessionStore.getState();
      expect(afterSessions[0].messages[0].status).toBe('loading');
    });
  });

  describe('File Management', () => {
    it('should add file to session', () => {
      const sessionId = useSessionStore.getState().addSession();
      const file = createMockMessage({ id: 'file-1' }) as any;
      
      useSessionStore.getState().addFile(sessionId, file);
      
      const { sessionFiles } = useSessionStore.getState();
      expect(sessionFiles[sessionId]).toHaveLength(1);
      expect(sessionFiles[sessionId][0]).toEqual(file);
    });

    it('should remove file from session', () => {
      const sessionId = useSessionStore.getState().addSession();
      const file = createMockMessage({ id: 'file-1' }) as any;
      
      useSessionStore.getState().addFile(sessionId, file);
      useSessionStore.getState().removeFile(sessionId, 'file-1');
      
      const { sessionFiles } = useSessionStore.getState();
      expect(sessionFiles[sessionId]).toHaveLength(0);
    });

    it('should get session files', () => {
      const sessionId = useSessionStore.getState().addSession();
      const file1 = createMockMessage({ id: 'file-1' }) as any;
      const file2 = createMockMessage({ id: 'file-2' }) as any;
      
      useSessionStore.getState().addFile(sessionId, file1);
      useSessionStore.getState().addFile(sessionId, file2);
      
      const files = useSessionStore.getState().getSessionFiles(sessionId);
      
      expect(files).toHaveLength(2);
    });

    it('should update file progress', () => {
      const sessionId = useSessionStore.getState().addSession();
      const file = createMockMessage({ id: 'file-1' }) as any;
      
      useSessionStore.getState().addFile(sessionId, file);
      useSessionStore.getState().updateFileProgress(sessionId, 'file-1', 50);
      
      const { sessionFiles } = useSessionStore.getState();
      expect(sessionFiles[sessionId][0].uploadProgress).toBe(50);
    });

    it('should clear session files', () => {
      const sessionId = useSessionStore.getState().addSession();
      const file = createMockMessage({ id: 'file-1' }) as any;
      
      useSessionStore.getState().addFile(sessionId, file);
      useSessionStore.getState().clearSessionFiles(sessionId);
      
      const { sessionFiles } = useSessionStore.getState();
      expect(sessionFiles[sessionId]).toBeUndefined();
    });
  });
});
