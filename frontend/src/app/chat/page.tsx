'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button, message } from 'antd';
import { PlusOutlined, MenuOutlined, CloseOutlined, FolderOutlined } from '@ant-design/icons';
import { useSessionStore } from '@/store/sessionStore';
import { useChatStream } from '@/hooks/useChatStream';
import { v4 as uuidv4 } from 'uuid';
import { FileAttachment, LogEntry } from '@/types';
import { SessionList } from '@/components/sidebar/SessionList';
import { MessageList } from '@/components/chat/MessageList';
import { InputPanel } from '@/components/chat/InputPanel';
import { EmptyState } from '@/components/common/EmptyState';
import { FileManager } from '@/components/files';
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

  const { submitTask, isLoading, disconnect } = useChatStream({
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError: handleError,
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
      message.success('消息发送成功');
    } catch (error: any) {
      console.error('发送消息失败:', error);
      message.error(error.message || '消息发送失败，请重试');
    }
  };

   return (
    <div className={styles.container}>
      {/* 跳过导航链接 */}
      <a href="#main-content" className={styles.skipLink}>
        跳过导航
      </a>
      
      {/* 背景网格 */}
      <div className={styles.backgroundGrid}></div>
      
      {/* 移动端菜单按钮 */}
      <button
        onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        className={styles.mobileMenuBtn}
        aria-label={isMobileSidebarOpen ? "关闭侧边栏" : "打开侧边栏"}
        aria-expanded={isMobileSidebarOpen}
      >
        {isMobileSidebarOpen ? <CloseOutlined /> : <MenuOutlined />}
      </button>

      {/* 侧边栏 */}
      <div className={`${styles.sidebar} ${isMobileSidebarOpen ? styles.sidebarOpen : ''}`}>
        {/* 头部 */}
        <div className={styles.sidebarHeader}>
          <h1 className={styles.logo}>AGENT CHAT</h1>
          <div className={styles.logoAccent}></div>
        </div>

         {/* 新建会话按钮 */}
        <div className={styles.newSessionBtn}>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleNewSession}
            block
            className={styles.primaryBtn}
            aria-label="新建聊天会话"
          >
            新建会话
          </Button>
        </div>

        {/* 会话列表 */}
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSessionSelect={(id) => {
            switchSession(id);
            setIsMobileSidebarOpen(false);
          }}
          onSessionDelete={deleteSession}
        />
      </div>

      {/* 移动端遮罩 */}
      {isMobileSidebarOpen && (
        <div
          className={styles.overlay}
          onClick={() => setIsMobileSidebarOpen(false)}
        />
      )}

      {/* 主内容区 */}
      <div className={styles.mainContent} role="main" id="main-content">
        {/* 顶部导航 */}
        <header className={styles.topNav} role="banner">
          <h2 className={styles.sessionTitle}>
            {activeSession?.title || 'AGENT CHAT'}
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className={styles.statusIndicator}>
              <span className={styles.statusDot}></span>
              <span className={styles.statusText}>ONLINE</span>
            </div>
            <Button
              icon={<FolderOutlined />}
              onClick={() => setIsFileManagerOpen(true)}
              disabled={!activeSessionId}
              className={styles.fileManagerBtn}
              aria-label="打开文件管理器"
            >
              文件管理
            </Button>
          </div>
        </header>

        {/* 聊天区域 */}
        <div className={styles.chatArea}>
          {activeSession?.messages.length ? (
            <div ref={messageListRef} className={styles.messageList}>
              <MessageList messages={activeSession.messages} />
            </div>
          ) : (
            <div className={styles.emptyState}>
              <EmptyState tip="发送第一条消息，开始与Agent的对话吧～" />
            </div>
          )}

          {/* 输入面板 */}
          <div className={styles.inputArea}>
            <InputPanel
              value={inputValue}
              onChange={setInputValue}
              onSend={handleSendMessage}
              onClear={() => {
                setInputValue('');
                setAttachments([]);
              }}
              disabled={isLoading || !activeSessionId}
              attachments={attachments}
              onAttachmentsChange={setAttachments}
            />
          </div>
        </div>
      </div>

      {/* 文件管理器 */}
      {isFileManagerOpen && (
        <FileManager
          isOpen={isFileManagerOpen}
          onClose={() => setIsFileManagerOpen(false)}
        />
      )}
    </div>
  );
}
