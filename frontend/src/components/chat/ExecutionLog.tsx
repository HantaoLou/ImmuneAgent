'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  BulbOutlined,
  CodeOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined,
  ToolOutlined,
} from '@ant-design/icons';
import { LogEntry } from '@/types';
import styles from './ExecutionLog.module.css';

interface ExecutionLogProps {
  logs: LogEntry[];
  defaultExpanded?: boolean;
}

const eventTypeConfig: Record<string, { icon: React.ReactNode; className: string }> = {
  llm_thinking: { icon: <BulbOutlined />, className: 'thinking' },
  llm_reasoning: { icon: <BulbOutlined />, className: 'thinking' },
  llm_streaming: { icon: <BulbOutlined />, className: 'thinking' },
  llm_thinking_start: { icon: <BulbOutlined />, className: 'thinking' },
  llm_thinking_end: { icon: <BulbOutlined />, className: 'thinking' },
  llm_response: { icon: <BulbOutlined />, className: 'thinking' },
  llm_reasoning_complete: { icon: <BulbOutlined />, className: 'thinking' },
  sandbox_stdout: { icon: <CodeOutlined />, className: 'sandbox' },
  sandbox_stderr: { icon: <ExclamationCircleOutlined />, className: 'sandbox' },
  sandbox_init: { icon: <CodeOutlined />, className: 'sandbox' },
  sandbox_complete: { icon: <CheckCircleOutlined />, className: 'sandbox' },
  sandbox_error: { icon: <ExclamationCircleOutlined />, className: 'sandbox' },
  tool_call: { icon: <ToolOutlined />, className: 'tool' },
  tool_result: { icon: <CheckCircleOutlined />, className: 'tool' },
  node_start: { icon: <LoadingOutlined />, className: 'node' },
  node_progress: { icon: <LoadingOutlined />, className: 'node' },
  node_complete: { icon: <CheckCircleOutlined />, className: 'node' },
  task_complete: { icon: <CheckCircleOutlined />, className: 'success' },
  task_start: { icon: <LoadingOutlined />, className: 'node' },
  task_progress: { icon: <LoadingOutlined />, className: 'node' },
  console_output: { icon: <CodeOutlined />, className: 'console' },
  error: { icon: <ExclamationCircleOutlined />, className: 'error' },
  info: { icon: <CodeOutlined />, className: 'console' },
  opencode_init: { icon: <CodeOutlined />, className: 'opencode' },
  opencode_stdout: { icon: <CodeOutlined />, className: 'opencode' },
  opencode_stderr: { icon: <ExclamationCircleOutlined />, className: 'opencode' },
  opencode_result: { icon: <CheckCircleOutlined />, className: 'opencode' },
  opencode_error: { icon: <ExclamationCircleOutlined />, className: 'opencode' },
  opencode_complete: { icon: <CheckCircleOutlined />, className: 'opencode' },
};

export const ExecutionLog: React.FC<ExecutionLogProps> = ({ logs, defaultExpanded = false }) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const logListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logListRef.current && expanded) {
      const element = logListRef.current;
      const isAtBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 50;
      if (isAtBottom) {
        element.scrollTop = element.scrollHeight;
      }
    }
  }, [logs, expanded]);

  if (!logs || logs.length === 0) {
    return null;
  }

  const getEventConfig = (eventType: string) => {
    return eventTypeConfig[eventType] || { icon: <CodeOutlined />, className: 'console' };
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return timestamp;
    }
  };

  const getEntryClassName = (eventType: string) => {
    if (eventType.includes('error')) return styles.errorEntry;
    if (eventType.includes('complete') || eventType.includes('success')) return styles.successEntry;
    if (eventType.startsWith('llm_')) return styles.thinkingEntry;
    return '';
  };

  const handleToggle = () => {
    console.log('ExecutionLog toggle:', !expanded, '->', !expanded);
    setExpanded(!expanded);
  };

  return (
    <div className={styles.container}>
      <button 
        type="button"
        className={styles.header} 
        onClick={handleToggle}
        aria-expanded={expanded}
        aria-label="Toggle execution log"
      >
        <span className={styles.headerLeft}>
          <span className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>▶</span>
          <span className={styles.title}>Execution Log</span>
          <span className={styles.countBadge}>{logs.length}</span>
        </span>
        <span className={styles.hint}>{expanded ? '收起' : '展开'}</span>
      </button>
      
      {expanded && (
        <div className={styles.logList} ref={logListRef}>
          {logs.map((log, index) => {
            const config = getEventConfig(log.event_type);
            return (
              <div
                key={log.id || index}
                className={`${styles.logEntry} ${getEntryClassName(log.event_type)}`}
              >
                <div className={styles.logMeta}>
                  <span className={`${styles.eventTypeTag} ${styles[config.className] || ''}`}>
                    {config.icon}
                    <span>{log.event_type.replace(/_/g, ' ')}</span>
                  </span>
                  {log.node_name && (
                    <span className={styles.nodeName}>{log.node_name}</span>
                  )}
                  <span className={styles.timestamp}>{formatTimestamp(log.timestamp)}</span>
                </div>
                <div className={styles.logContent}>{log.message}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
