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
  } = useSessionStore();

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  const handleProgress = useCallback((log: LogEntry) => {
    executionLogsRef.current = [...executionLogsRef.current, log];
    
    if (activeSessionId && agentMessageIdRef.current) {
      updateMessage(activeSessionId, agentMessageIdRef.current, {
        executionLogs: [...executionLogsRef.current],
      });
    }
  }, [activeSessionId, updateMessage]);

  const handleComplete = useCallback((result: any) => {
    if (activeSessionId && agentMessageIdRef.current) {
      const responseContent = result?.answer || result?.summary?.answer || result?.result?.merged_result?.message || 'Task completed';
      updateMessage(activeSessionId, agentMessageIdRef.current, {
        content: typeof responseContent === 'string' ? responseContent : JSON.stringify(responseContent),
        status: 'success',
        executionLogs: [...executionLogsRef.current],
      });
    }
    executionLogsRef.current = [];
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

  const handleHITLConfirm = useCallback(async (sessionId: string, feedback?: string, parameters?: Record<string, any>) => {
    console.log('[page] handleHITLConfirm called:', sessionId);
    
    const { sessions: currentSessions } = useSessionStore.getState();
    const session = currentSessions.find(s => s.id === sessionId);
    const hitlMessage = session?.messages.find(m => m.hitlRequest);
    const oldHitlId = hitlMessage?.hitlRequest?.hitl_id;

    if (!hitlMessage || !oldHitlId) {
      antMessage.error('找不到 HITL 请求');
      return;
    }

    console.log('[page] Found hitlMessage:', hitlMessage.id, 'hitl_id:', oldHitlId);

    try {
      const result = await resumeHITL({
        session_id: sessionId,
        hitl_id: oldHitlId,
        confirmed: true,
        feedback,
        parameters,
      });

      console.log('[page] resumeHITL result:', result);
      console.log('[page] result.hitlRequest:', result.hitlRequest?.hitl_id);

      if (result.hitlRequest) {
        console.log('[page] New HITL request, updating message');
        updateMessage(sessionId, hitlMessage.id, {
          hitlRequest: result.hitlRequest,
          status: 'success',
        });
        antMessage.info('任务有新的确认请求');
      } else {
        console.log('[page] No new HITL, task complete');
        updateMessage(sessionId, hitlMessage.id, {
          hitlRequest: undefined,
          content: 'Task completed successfully',
          status: 'success',
        });
        antMessage.success('任务已完成');
      }
    } catch (error: any) {
      console.error('[page] Failed to confirm HITL:', error);
      antMessage.error('确认失败: ' + error.message);
    }
  }, [updateMessage]);

  const handleHITLReject = useCallback(async (sessionId: string, feedback: string) => {
    console.log('[page] handleHITLReject called:', sessionId);
    
    const { sessions: currentSessions } = useSessionStore.getState();
    const session = currentSessions.find(s => s.id === sessionId);
    const hitlMessage = session?.messages.find(m => m.hitlRequest);
    const oldHitlId = hitlMessage?.hitlRequest?.hitl_id;

    if (!hitlMessage || !oldHitlId) {
      antMessage.error('找不到 HITL 请求');
      return;
    }

    try {
      const result = await resumeHITL({
        session_id: sessionId,
        hitl_id: oldHitlId,
        confirmed: false,
        feedback,
      });

      console.log('[page] resumeHITL result:', result);

      if (result.hitlRequest) {
        console.log('[page] New HITL request after rejection');
        updateMessage(sessionId, hitlMessage.id, {
          hitlRequest: result.hitlRequest,
          status: 'success',
        });
        antMessage.info('任务有新的确认请求');
      } else {
        console.log('[page] Task updated');
        updateMessage(sessionId, hitlMessage.id, {
          hitlRequest: undefined,
          content: `Task updated based on feedback: ${feedback}`,
          status: 'success',
        });
        antMessage.success('修改请求已提交');
      }
    } catch (error: any) {
      console.error('[page] Failed to reject HITL:', error);
      antMessage.error('提交失败: ' + error.message);
    }
  }, [updateMessage]);

  const { submitTask, isLoading, disconnect } = useChatStream({
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError: handleError,
    onHITLRequest: handleHITLRequest,
  });

  useEffect(() => {
    if (sessions.length === 0) {
      addSession();
    }
  }, []);

  const scrollToBottom = () => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    if (activeSession?.messages.length) {
      scrollToBottom();
    }
  }, [activeSession?.messages]);

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
      console.error('发送消息失败:', error);
      antMessage.error(error.message || '消息发送失败');
    }
  };

  return (
    <div className={styles.container}>
      <a href="#main-content" className={styles.skipLink}>跳过导航</a>
      <div className={styles.backgroundGrid}></div>
      
      <button
        onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        className={styles.mobileMenuBtn}
        aria-label={isMobileSidebarOpen ? "关闭侧边栏" : "打开侧边栏"}
      >
        {isMobileSidebarOpen ? <CloseOutlined /> : <MenuOutlined />}
      </button>

      <div className={`${styles.sidebar} ${isMobileSidebarOpen ? styles.sidebarOpen : ''}`}>
        <div className={styles.sidebarHeader}>
          <h1 className={styles.logo}>AGENT CHAT</h1>
        </div>
        <div className={styles.newSessionBtn}>
          <Button type="primary" icon={<PlusOutlined />} onClick={handleNewSession} block>
            新建会话
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
              文件管理
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
              <EmptyState tip="发送第一条消息，开始与Agent的对话吧～" />
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
