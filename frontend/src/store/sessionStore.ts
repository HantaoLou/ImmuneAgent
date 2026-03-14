import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Session, Message, FileAttachment } from '@/types';
import { v4 as uuidv4 } from 'uuid';

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  sessionFiles: Record<string, FileAttachment[]>;
  
  addSession: (session?: Partial<Session>) => string;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  addMessage: (sessionId: string, message: Omit<Message, 'id'>) => void;
  updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
  updateMessageStatus: (sessionId: string, messageId: string, status: Message['status']) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
  
  addFile: (sessionId: string, file: FileAttachment) => void;
  removeFile: (sessionId: string, fileId: string) => void;
  getSessionFiles: (sessionId: string) => FileAttachment[];
  updateFileProgress: (sessionId: string, fileId: string, progress: number) => void;
  clearSessionFiles: (sessionId: string) => void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      sessionFiles: {},
      
      addSession: (session) => {
        const newSession: Session = {
          id: session?.id || uuidv4(),
          title: session?.title || '新会话',
          messages: session?.messages || [],
          createTime: session?.createTime || Date.now(),
          updateTime: session?.updateTime || Date.now(),
        };
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: newSession.id,
        }));
        return newSession.id;
      },
      
      switchSession: (sessionId) => {
        set({ activeSessionId: sessionId });
      },
      
      deleteSession: (sessionId) => {
        const { sessions, activeSessionId, sessionFiles } = get();
        const newSessions = sessions.filter((s) => s.id !== sessionId);
        const newSessionFiles = { ...sessionFiles };
        delete newSessionFiles[sessionId];
        
        let newActiveId = activeSessionId;
        
        if (activeSessionId === sessionId) {
          newActiveId = newSessions.length > 0 ? newSessions[newSessions.length - 1].id : null;
          if (!newActiveId) {
            newActiveId = get().addSession();
          }
        }
        
        set({
          sessions: newSessions,
          activeSessionId: newActiveId,
          sessionFiles: newSessionFiles,
        });
      },
      
      addMessage: (sessionId, message) => {
        const newMessage: Message = {
          id: uuidv4(),
          ...message,
        };
        
        set((state) => {
          const sessions = state.sessions.map((s) => {
            if (s.id === sessionId) {
              const session = {
                ...s,
                messages: [...s.messages, newMessage],
                updateTime: Date.now()
              };
              
              if (message.role === 'user' && s.messages.length === 0) {
                session.title = message.content.slice(0, 15) || '新会话';
              }
              
              return session;
            }
            return s;
          });
          
          return { sessions };
        });
      },
      
      updateMessage: (sessionId, messageId, updates) => {
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id === sessionId) {
              return {
                ...s,
                messages: s.messages.map((m) => 
                  m.id === messageId ? { ...m, ...updates } : m
                ),
                updateTime: Date.now()
              };
            }
            return s;
          }),
        }));
      },
      
      updateMessageStatus: (sessionId, messageId, status) => {
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id === sessionId) {
              return {
                ...s,
                messages: s.messages.map((m) => 
                  m.id === messageId ? { ...m, status } : m
                ),
              };
            }
            return s;
          }),
        }));
      },
      
      updateSessionTitle: (sessionId, title) => {
        set((state) => ({
          sessions: state.sessions.map((s) => 
            s.id === sessionId ? { ...s, title } : s
          ),
        }));
      },
      
      addFile: (sessionId, file) => {
        set((state) => ({
          sessionFiles: {
            ...state.sessionFiles,
            [sessionId]: [...(state.sessionFiles[sessionId] || []), file],
          },
        }));
      },
      
      removeFile: (sessionId, fileId) => {
        set((state) => ({
          sessionFiles: {
            ...state.sessionFiles,
            [sessionId]: (state.sessionFiles[sessionId] || []).filter(
              (f) => f.id !== fileId
            ),
          },
        }));
      },
      
      getSessionFiles: (sessionId) => {
        return get().sessionFiles[sessionId] || [];
      },
      
      updateFileProgress: (sessionId, fileId, progress) => {
        set((state) => ({
          sessionFiles: {
            ...state.sessionFiles,
            [sessionId]: (state.sessionFiles[sessionId] || []).map((f) =>
              f.id === fileId ? { ...f, uploadProgress: progress } : f
            ),
          },
        }));
      },
      
      clearSessionFiles: (sessionId) => {
        set((state) => {
          const newSessionFiles = { ...state.sessionFiles };
          delete newSessionFiles[sessionId];
          return { sessionFiles: newSessionFiles };
        });
      },
    }),
    {
      name: 'agent-chat-sessions',
      version: 1,
      partialize: (state) => ({ 
        // 只持久化最近 50 条消息，避免 localStorage 膨胀导致性能问题
        // 完整历史可通过服务器 API 获取
        sessions: state.sessions.map(s => ({
          ...s,
          messages: s.messages.slice(-50),
        })), 
        activeSessionId: state.activeSessionId,
      }),
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          // 从版本 0 迁移到版本 1：限制消息数量
          if (persistedState.sessions) {
            persistedState.sessions = persistedState.sessions.map((s: Session) => ({
              ...s,
              messages: s.messages.slice(-50),
            }));
          }
          // 移除旧的 sessionFiles（不再持久化）
          delete persistedState.sessionFiles;
        }
        return persistedState;
      },
    }
  )
);
