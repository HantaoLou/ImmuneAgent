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
  deleteMessage: (sessionId: string, messageId: string) => void;
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
          title: session?.title || 'New Session',
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
        
        // Check if current activeSessionId is still valid (exists in newSessions)
        const isActiveValid = newSessions.some(s => s.id === activeSessionId);
        
        if (!isActiveValid) {
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
                session.title = message.content.slice(0, 15) || 'New Session';
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
      
      deleteMessage: (sessionId, messageId) => {
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id === sessionId) {
              return {
                ...s,
                messages: s.messages.filter((m) => m.id !== messageId),
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
        // Only persist the most recent 50 messages to avoid localStorage bloat causing performance issues
        // Complete history can be retrieved via server API
        sessions: state.sessions.map(s => ({
          ...s,
          messages: s.messages.slice(-50),
        })), 
        activeSessionId: state.activeSessionId,
      }),
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          // Migrate from version 0 to version 1: limit message count
          if (persistedState.sessions) {
            persistedState.sessions = persistedState.sessions.map((s: Session) => ({
              ...s,
              messages: s.messages.slice(-50),
            }));
          }
          // Remove old sessionFiles (no longer persisted)
          delete persistedState.sessionFiles;
        }
        return persistedState;
      },
      onRehydrateStorage: () => (state) => {
        // Check if activeSessionId is valid after rehydration
        if (state) {
          const { sessions, activeSessionId } = state;
          const isActiveValid = sessions.some(s => s.id === activeSessionId);
          if (!isActiveValid && sessions.length > 0) {
            state.activeSessionId = sessions[sessions.length - 1].id;
          } else if (!isActiveValid) {
            state.activeSessionId = null;
          }
        }
      },
    }
  )
);
