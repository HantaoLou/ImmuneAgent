import { useCallback, useRef, useState } from 'react';
import api from '@/services/api';
import { LogEntry, HITLRequest } from '@/types';
import { v4 as uuidv4 } from 'uuid';
import { parseSSEEventData, SSEEventData } from '@/utils/sseParser';

interface UseChatStreamOptions {
  onProgress?: (log: LogEntry) => void;
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
  onHITLRequest?: (hitlRequest: HITLRequest) => void;
}

export const useChatStream = (options?: UseChatStreamOptions) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logsRef = useRef<LogEntry[]>([]);

  const handleSSEEvent = useCallback((event: MessageEvent, eventType: string) => {
    console.log('[useChatStream] handleSSEEvent called:', { eventType, data: event.data });

    try {
      const rawData = event.data;
      const parsed = parseSSEEventData(rawData);
      
      console.log('[useChatStream] Parsed result:', parsed);

      if (eventType === 'progress' || eventType === 'message') {
        const data = parsed.data;
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

        if (parsed.type === 'task_complete') {
          console.log('[useChatStream] Handling task_complete');
          options?.onComplete?.(parsed.result);
          disconnect();
        } else if (parsed.type === 'hitl_request') {
          console.log('[useChatStream] Handling hitl_request');
          console.log('[useChatStream] HITL request:', parsed.hitlRequest);
          if (parsed.hitlRequest) {
            options?.onHITLRequest?.(parsed.hitlRequest);
          }
          disconnect();
        }
      } else if (eventType === 'error') {
        options?.onError?.(parsed.data.message || 'Unknown error');
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

    console.log('[useChatStream] submitTask called:', { message, sessionId, isLoading });

    try {
      const response = await api.post('/api/chat/submit', {
        message,
        session_id: sessionId,
      });

      const { session_id } = response.data;
      console.log('[useChatStream] Submit response:', response.data);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const sseUrl = `${apiUrl}/api/chat/stream/${session_id}`;
      console.log('[useChatStream] SSE URL:', sseUrl);

      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;
      setIsConnected(true);

      eventSource.onopen = () => {
        console.log('[useChatStream] EventSource opened');
        setIsConnected(true);
      };

      eventSource.addEventListener('progress', (event) => {
        console.log('[useChatStream] Received progress event:', event);
        handleSSEEvent(event as MessageEvent, 'progress');
      });

      eventSource.addEventListener('status', (event) => {
        console.log('[useChatStream] Received status event:', event);
        handleSSEEvent(event as MessageEvent, 'status');
      });

      eventSource.addEventListener('done', (event) => {
        console.log('[useChatStream] Received done event:', event);
        handleSSEEvent(event as MessageEvent, 'done');
        disconnect();
      });

      eventSource.addEventListener('error', (event) => {
        console.log('[useChatStream] Received error event:', event);
        if ((event as MessageEvent).data) {
          handleSSEEvent(event as MessageEvent, 'error');
        }
        disconnect();
      });

      eventSource.onerror = (error) => {
        console.error('[useChatStream] EventSource error:', error);
        disconnect();
      };

      return session_id;
    } catch (error: any) {
      console.error('[useChatStream] Submit error:', error);
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
