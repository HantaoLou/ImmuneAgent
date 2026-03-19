import { useRef, useEffect, useCallback } from 'react';
import api from './api';
import { ChatRequest, ChatResponse } from '@/types';
import axios, { CancelTokenSource } from 'axios';

/**
 * Chat Service Hook
 * Provides chat message sending functionality, automatically manages request cancellation tokens to avoid memory leaks
 */
export const useChatService = () => {
  const cancelTokenRef = useRef<CancelTokenSource | null>(null);

  /**
   * Send chat message
   * @param params Chat request parameters
   * @returns Chat response
   */
  const sendChatMessage = useCallback(async (params: ChatRequest): Promise<ChatResponse> => {
    // Cancel previous request
    if (cancelTokenRef.current) {
      cancelTokenRef.current.cancel('Cancel previous request');
      cancelTokenRef.current = null;
    }

    // Create new cancel token
    cancelTokenRef.current = axios.CancelToken.source();

    try {
      const response = await api.post<ChatResponse>('/api/chat', params, {
        cancelToken: cancelTokenRef.current.token,
      });
      
      return response.data;
    } finally {
      // Clean up after request completes
      cancelTokenRef.current = null;
    }
  }, []);

  /**
   * Cancel current request
   */
  const cancelChatRequest = useCallback(() => {
    if (cancelTokenRef.current) {
      cancelTokenRef.current.cancel('Request cancelled');
      cancelTokenRef.current = null;
    }
  }, []);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (cancelTokenRef.current) {
        cancelTokenRef.current.cancel('Component unmounted');
        cancelTokenRef.current = null;
      }
    };
  }, []);

  return {
    sendChatMessage,
    cancelChatRequest,
  };
};

/**
 * Legacy export for compatibility (deprecated, please use useChatService)
 * @deprecated Use useChatService Hook instead
 */
export const sendChatMessage = async (params: ChatRequest): Promise<ChatResponse> => {
  console.warn('sendChatMessage is deprecated, please use useChatService Hook');
  
  const cancelToken = axios.CancelToken.source();
  
  const response = await api.post<ChatResponse>('/api/chat', params, {
    cancelToken: cancelToken.token,
  });
  
  return response.data;
};

/**
 * Legacy export for compatibility (deprecated)
 * @deprecated Use useChatService Hook instead
 */
export const cancelChatRequest = () => {
  console.warn('cancelChatRequest is deprecated, please use useChatService Hook');
};
