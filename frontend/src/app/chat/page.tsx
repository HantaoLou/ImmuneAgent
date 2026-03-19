'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button, message as antMessage } from 'antd';
import { PlusOutlined, MenuOutlined, CloseOutlined, FolderOutlined } from '@ant-design/icons';
import { useSessionStore } from '@/store/sessionStore';
import { useChatStream } from '@/hooks/useChatStream';
import { v4 as uuidv4 } from 'uuid';
import { FileAttachment, LogEntry, HITLRequest } from '@/types';
import { SessionList } from '@/components/sidebar/SessionList';
import { MessageList } from '@/components/chat/MessageList';
import { InputPanel } from '@/components/chat/InputPanel';
import { EmptyState } from '@/components/common/EmptyState';
import { FileManager } from '@/components/files';
import { resumeHITL } from '@/services/hitlService';
import { parseSSEEventData } from '@/utils/sseParser';
import styles from './page.module.css';

export default function ChatPage() {
  const [inputValue, setInputValue] = useState('');
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isFileManagerOpen, setIsFileManagerOpen] = useState(false);
  const messageListRef = useRef<HTMLDivElement>(null);
  const executionLogsRef = useRef<LogEntry[]>([]);
  const agentMessageIdRef = useRef<string | null>(null);

  const {
    sessions,
    activeSessionId,
    addSession,
    switchSession,
    deleteSession,
    addMessage,
    updateMessage,
    deleteMessage,
  } = useSessionStore();

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  const handleProgress = useCallback((log: LogEntry) => {
    if (!activeSessionId || !agentMessageIdRef.current) return;
    
    const { sessions: currentSessions } = useSessionStore.getState();
    const currentSession = currentSessions.find(s => s.id === activeSessionId);
    const currentMessage = currentSession?.messages.find(m => m.id === agentMessageIdRef.current);
    const currentLogs = currentMessage?.executionLogs || [];
    
    updateMessage(activeSessionId, agentMessageIdRef.current, {
      executionLogs: [...currentLogs, log],
    });
  }, [activeSessionId, updateMessage]);

  const handleComplete = useCallback((result: any) => {
    if (activeSessionId && agentMessageIdRef.current) {
      const { sessions } = useSessionStore.getState();
      const currentSession = sessions.find(s => s.id === activeSessionId);
      const currentMessage = currentSession?.messages.find(m => m.id === agentMessageIdRef.current);
      const currentLogs = currentMessage?.executionLogs || [];
      
      const responseContent = result?.answer || result?.summary?.answer || result?.result?.merged_result?.message || 'Task completed';
      updateMessage(activeSessionId, agentMessageIdRef.current, {
        content: typeof responseContent === 'string' ? responseContent : JSON.stringify(responseContent),
        status: 'success',
        executionLogs: currentLogs,
      });
    }
    agentMessageIdRef.current = null;
  }, [activeSessionId, updateMessage]);

  const handleError = useCallback((error: string) => {
    if (activeSessionId && agentMessageIdRef.current) {
      updateMessage(activeSessionId, agentMessageIdRef.current, {
        content: error,
        status: 'error',
      });
    }
    executionLogsRef.current = [];
    agentMessageIdRef.current = null;
  }, [activeSessionId, updateMessage]);

  const handleHITLRequest = useCallback((hitlRequest: HITLRequest) => {
    console.log('[page] handleHITLRequest called:', hitlRequest?.hitl_id);
    
    if (activeSessionId && agentMessageIdRef.current) {
      updateMessage(activeSessionId, agentMessageIdRef.current, {
        status: 'success',
        hitlRequest,
      });
    }
    agentMessageIdRef.current = null;
  }, [activeSessionId, updateMessage]);

  const handleHITLConfirm = useCallback(async (sessionId: string, feedback?: string, parameters?: Record<string, any>, taskMd?: string) => {
    console.log('[page] handleHITLConfirm called:', sessionId);
    
    const { sessions: currentSessions } = useSessionStore.getState();
    const session = currentSessions.find(s => s.id === sessionId);
    const hitlMessage = session?.messages.filter(m => m.hitlRequest).pop();
    const oldHitlId = hitlMessage?.hitlRequest?.hitl_id;

    if (!hitlMessage || !oldHitlId) {
      antMessage.error('HITL request not found');
      return;
    }

    console.log('[page] Found hitlMessage:', hitlMessage.id, 'hitl_id:', oldHitlId);

    const agentMessageId = hitlMessage.id;

    // Add user confirmation message (before agent message)
    const userConfirmMessage = {
      role: 'user' as const,
      content: `user confirm the plan:\n${taskMd || hitlMessage.hitlRequest?.task_md || ''}`,
      timestamp: Date.now(),
      status: 'success' as const,
    };
    addMessage(sessionId, userConfirmMessage);

    // Clear hitlRequest, keep executionLogs, set to loading state
    updateMessage(sessionId, agentMessageId, {
      hitlRequest: undefined,
      status: 'loading' as const,
    });
    
    // Set ref to point to original message
    agentMessageIdRef.current = agentMessageId;

    // Progress callback
    const handleResumeProgress = (event: MessageEvent) => {
      console.log('[page] handleResumeProgress (confirm) called');
      try {
        const parsed = parseSSEEventData(event.data);
        if (parsed.data.event_type) {
          const logEntry: LogEntry = {
            id: uuidv4(),
            event_type: parsed.data.event_type,
            message: parsed.data.message,
            timestamp: parsed.data.timestamp,
            node_name: parsed.data.node_name,
            details: parsed.data.details,
          };
          // Get latest logs from store and append to original message
          const { sessions: latestSessions } = useSessionStore.getState();
          const latestSession = latestSessions.find(s => s.id === sessionId);
          const currentMessage = latestSession?.messages.find(m => m.id === agentMessageId);
          const currentLogs = currentMessage?.executionLogs || [];
          updateMessage(sessionId, agentMessageId, {
            executionLogs: [...currentLogs, logEntry],
          });
        }
      } catch (e) {
        console.error('[page] Failed to parse resume progress (confirm):', e);
      }
    };

    try {
      const result = await resumeHITL({
        session_id: sessionId,
        hitl_id: oldHitlId,
        confirmed: true,
        feedback,
        parameters,
      }, handleResumeProgress);

      console.log('[page] resumeHITL result:', result);
      console.log('[page] result.hitlRequest:', result.hitlRequest?.hitl_id);

      if (result.hitlRequest) {
        console.log('[page] New HITL request, updating message');
        updateMessage(sessionId, agentMessageId, {
          hitlRequest: result.hitlRequest,
          status: 'success' as const,
        });
        antMessage.info('New confirmation request received');
      } else {
        console.log('[page] No new HITL, task complete');
        updateMessage(sessionId, agentMessageId, {
          content: 'Task completed successfully',
          status: 'success' as const,
        });
        antMessage.success('Task completed');
      }
    } catch (error: any) {
      console.error('[page] Failed to confirm HITL:', error);
      updateMessage(sessionId, agentMessageId, {
        content: error.message || 'Confirmation failed',
        status: 'error' as const,
      });
      antMessage.error('Confirmation failed: ' + error.message);
    } finally {
      agentMessageIdRef.current = null;
    }
  }, [updateMessage, addMessage]);

  const handleHITLReject = useCallback(async (sessionId: string, feedback: string) => {
    console.log('[page] handleHITLReject called:', sessionId);
    
    const { sessions: currentSessions } = useSessionStore.getState();
    const session = currentSessions.find(s => s.id === sessionId);
    const hitlMessage = session?.messages.filter(m => m.hitlRequest).pop();
    const oldHitlId = hitlMessage?.hitlRequest?.hitl_id;

    if (!hitlMessage || !oldHitlId) {
      antMessage.error('HITL request not found');
      return;
    }

    const agentMessageId = hitlMessage.id;

    // Add user rejection message (before agent message)
    const userRejectMessage = {
      role: 'user' as const,
      content: `user reject the plan with feedback:\n${feedback}`,
      timestamp: Date.now(),
      status: 'success' as const,
    };
    addMessage(sessionId, userRejectMessage);

    // Clear hitlRequest, keep executionLogs, set to loading state
    updateMessage(sessionId, agentMessageId, {
      hitlRequest: undefined,
      status: 'loading' as const,
    });
    
    // Set ref to point to original message
    agentMessageIdRef.current = agentMessageId;

    // Progress callback
    const handleResumeProgress = (event: MessageEvent) => {
      console.log('[page] handleResumeProgress (reject) called');
      try {
        const parsed = parseSSEEventData(event.data);
        if (parsed.data.event_type) {
          const logEntry: LogEntry = {
            id: uuidv4(),
            event_type: parsed.data.event_type,
            message: parsed.data.message,
            timestamp: parsed.data.timestamp,
            node_name: parsed.data.node_name,
            details: parsed.data.details,
          };
          // Get latest logs from store and append to original message
          const { sessions: latestSessions } = useSessionStore.getState();
          const latestSession = latestSessions.find(s => s.id === sessionId);
          const currentMessage = latestSession?.messages.find(m => m.id === agentMessageId);
          const currentLogs = currentMessage?.executionLogs || [];
          updateMessage(sessionId, agentMessageId, {
            executionLogs: [...currentLogs, logEntry],
          });
        }
      } catch (e) {
        console.error('[page] Failed to parse resume progress (reject):', e);
      }
    };

    try {
      const result = await resumeHITL({
        session_id: sessionId,
        hitl_id: oldHitlId,
        confirmed: false,
        feedback,
      }, handleResumeProgress);

      console.log('[page] resumeHITL result:', result);

      if (result.hitlRequest) {
        console.log('[page] New HITL request after rejection');
        updateMessage(sessionId, agentMessageId, {
          hitlRequest: result.hitlRequest,
          status: 'success' as const,
        });
        antMessage.info('New confirmation request received');
      } else {
        console.log('[page] Task updated');
        updateMessage(sessionId, agentMessageId, {
          content: 'Task updated based on feedback',
          status: 'success' as const,
        });
        antMessage.success('Modification request submitted');
      }
    } catch (error: any) {
      console.error('[page] Failed to reject HITL:', error);
      updateMessage(sessionId, agentMessageId, {
        content: error.message || 'Submission failed',
        status: 'error' as const,
      });
      antMessage.error('Submission failed: ' + error.message);
    } finally {
      agentMessageIdRef.current = null;
    }
  }, [updateMessage, addMessage]);

  const { submitTask, isLoading, disconnect } = useChatStream({
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError: handleError,
    onHITLRequest: handleHITLRequest,
  });

  const handleNewSession = () => {
    const newSessionId = addSession();
    switchSession(newSessionId);
    setInputValue('');
    setAttachments([]);
    setIsMobileSidebarOpen(false);
  };

  const handleSendMessage = async () => {
    if ((!inputValue.trim() && attachments.length === 0) || !activeSessionId || isLoading) return;

    const userMessage = {
      role: 'user' as const,
      content: inputValue.trim(),
      timestamp: Date.now(),
      status: 'success' as const,
      attachments: attachments.length > 0 ? attachments : undefined,
    };

    addMessage(activeSessionId, userMessage);

    const agentMessage = {
      role: 'agent' as const,
      content: '',
      timestamp: Date.now(),
      status: 'loading' as const,
      executionLogs: [] as LogEntry[],
    };
    
    addMessage(activeSessionId, agentMessage);

    const { sessions } = useSessionStore.getState();
    const currentSession = sessions.find((s) => s.id === activeSessionId);
    const agentMessageId = currentSession?.messages[currentSession.messages.length - 1]?.id;
    
    agentMessageIdRef.current = agentMessageId || null;
    executionLogsRef.current = [];

    setInputValue('');
    setAttachments([]);

    try {
      await submitTask(userMessage.content, activeSessionId);
    } catch (error: any) {
      console.error('Failed to send message:', error);
      antMessage.error(error.message || 'Failed to send message');
    }
  };

  return (
    <div className={styles.container}>
      <a href="#main-content" className={styles.skipLink}>Skip navigation</a>
      <div className={styles.backgroundGrid}></div>

      <div className={`${styles.sidebar} ${isMobileSidebarOpen ? styles.sidebarOpen : ''}`}>
        <div className={styles.sidebarHeader}>
          <h1 className={styles.logo}>AGENT CHAT</h1>
        </div>
        <div className={styles.newSessionBtn}>
          <Button type="primary" icon={<PlusOutlined />} onClick={handleNewSession} block>
            New Session
          </Button>
        </div>
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSessionSelect={(id) => { switchSession(id); setIsMobileSidebarOpen(false); }}
          onSessionDelete={deleteSession}
        />
      </div>

      {isMobileSidebarOpen && <div className={styles.overlay} onClick={() => setIsMobileSidebarOpen(false)} />}

      <div className={styles.mainContent} role="main" id="main-content">
        <header className={styles.topNav}>
          <h2 className={styles.sessionTitle}>{activeSession?.title || 'AGENT CHAT'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className={styles.statusIndicator}>
              <span className={styles.statusDot}></span>
              <span className={styles.statusText}>ONLINE</span>
            </div>
            <Button icon={<FolderOutlined />} onClick={() => setIsFileManagerOpen(true)} disabled={!activeSessionId}>
              File Manager
            </Button>
          </div>
        </header>

        <div className={styles.chatArea}>
          {activeSession?.messages.length ? (
            <div ref={messageListRef} className={styles.messageList}>
              <MessageList 
                messages={activeSession.messages} 
                onHITLConfirm={handleHITLConfirm} 
                onHITLReject={handleHITLReject} 
              />
            </div>
          ) : (
            <div className={styles.emptyState}>
              <EmptyState 
                tip="Send your first message to start the conversation with Agent~" 
                showTemplates={!!activeSessionId}
                onQuestionClick={(question) => setInputValue(question)}
              />
            </div>
          )}
          <div className={styles.inputArea}>
            <InputPanel
              value={inputValue}
              onChange={setInputValue}
              onSend={handleSendMessage}
              onClear={() => { setInputValue(''); setAttachments([]); }}
              disabled={isLoading || !activeSessionId}
              attachments={attachments}
              onAttachmentsChange={setAttachments}
            />
          </div>
        </div>
      </div>

      {isFileManagerOpen && <FileManager isOpen={isFileManagerOpen} onClose={() => setIsFileManagerOpen(false)} />}
    </div>
  );
}
