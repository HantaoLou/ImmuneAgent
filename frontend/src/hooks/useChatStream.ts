import { useCallback, useRef, useState } from 'react';
import api from '@/services/api';
import { LogEntry } from '@/types';
import { v4 as uuidv4 } from 'uuid';

interface SSEEvent {
  event_type: string;
  message: string;
  timestamp: string;
  node_name?: string;
  details?: Record<string, any>;
  session_id?: string;
}

interface UseChatStreamOptions {
  onProgress?: (log: LogEntry) => void;
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
}

export const useChatStream = (options?: UseChatStreamOptions) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logsRef = useRef<LogEntry[]>([]);

  const handleSSEEvent = useCallback((event: MessageEvent, eventType: string) => {
    try {
      const data: SSEEvent = JSON.parse(event.data);
      
      if (eventType === 'progress') {
        const logEntry: LogEntry = {
          id: uuidv4(),
          event_type: data.event_type,
          message: data.message,
          timestamp: data.timestamp,
          node_name: data.node_name,
          details: data.details,
        };
        
        logsRef.current = [...logsRef.current, logEntry];
        options?.onProgress?.(logEntry);
        
        if (data.event_type === 'task_complete') {
          options?.onComplete?.(data.details?.result);
          disconnect();
        }
      } else if (eventType === 'error') {
        options?.onError?.(data.message || 'Unknown error');
        disconnect();
      }
    } catch (e) {
      console.error('Failed to parse SSE event:', e);
    }
  }, [options]);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
      setIsLoading(false);
    }
  }, []);

  const submitTask = useCallback(async (message: string, sessionId?: string) => {
    if (isLoading) return;
    
    logsRef.current = [];
    setIsLoading(true);

    try {
      const response = await api.post('/api/chat/submit', {
        message,
        session_id: sessionId,
      });

      const { session_id } = response.data;
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const sseUrl = `${apiUrl}/api/chat/stream/${session_id}`;
      
      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;
      setIsConnected(true);

      eventSource.onopen = () => {
        setIsConnected(true);
      };

      eventSource.addEventListener('progress', (event) => {
        handleSSEEvent(event as MessageEvent, 'progress');
      });

      eventSource.addEventListener('status', (event) => {
        handleSSEEvent(event as MessageEvent, 'status');
      });

      eventSource.addEventListener('done', (event) => {
        handleSSEEvent(event as MessageEvent, 'done');
        disconnect();
      });

      eventSource.addEventListener('error', (event) => {
        if ((event as MessageEvent).data) {
          handleSSEEvent(event as MessageEvent, 'error');
        }
        disconnect();
      });

      eventSource.onerror = () => {
        disconnect();
      };

      return session_id;
    } catch (error: any) {
      setIsLoading(false);
      options?.onError?.(error.message || 'Failed to submit task');
      throw error;
    }
  }, [isLoading, handleSSEEvent, disconnect, options]);

  const getLogs = useCallback(() => {
    return logsRef.current;
  }, []);

  return {
    submitTask,
    disconnect,
    isLoading,
    isConnected,
    getLogs,
  };
};
