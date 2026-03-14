import { useCallback, useEffect, useRef, useState } from 'react';

interface UseEventSourceOptions {
  onMessage?: (event: MessageEvent) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
}

export const useEventSource = (url: string | null, options?: UseEventSourceOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    if (!url || eventSourceRef.current) return;

    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      options?.onOpen?.();
    };

    eventSource.onmessage = (event) => {
      options?.onMessage?.(event);
    };

    eventSource.onerror = (error) => {
      setIsConnected(false);
      options?.onError?.(error);
      
      if (eventSource.readyState === EventSource.CLOSED) {
        eventSourceRef.current = null;
      }
    };

    eventSource.addEventListener('progress', (event) => {
      options?.onMessage?.(event as MessageEvent);
    });

    eventSource.addEventListener('status', (event) => {
      options?.onMessage?.(event as MessageEvent);
    });

    eventSource.addEventListener('done', (event) => {
      options?.onMessage?.(event as MessageEvent);
      eventSource.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    });

    eventSource.addEventListener('error', (event) => {
      options?.onMessage?.(event as MessageEvent);
    });

    eventSource.addEventListener('heartbeat', () => {
      // Heartbeat received, connection is alive
    });
  }, [url, options]);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    connect,
    disconnect,
  };
};
