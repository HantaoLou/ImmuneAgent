import { useRef, useEffect, useCallback } from 'react';
import api from './api';
import { ChatRequest, ChatResponse } from '@/types';
import axios, { CancelTokenSource } from 'axios';

/**
 * Chat Service Hook
 * 提供聊天消息发送功能，自动管理请求取消令牌，避免内存泄漏
 */
export const useChatService = () => {
  const cancelTokenRef = useRef<CancelTokenSource | null>(null);

  /**
   * 发送聊天消息
   * @param params 聊天请求参数
   * @returns 聊天响应
   */
  const sendChatMessage = useCallback(async (params: ChatRequest): Promise<ChatResponse> => {
    // 取消之前的请求
    if (cancelTokenRef.current) {
      cancelTokenRef.current.cancel('取消上一个请求');
      cancelTokenRef.current = null;
    }

    // 创建新的取消令牌
    cancelTokenRef.current = axios.CancelToken.source();

    try {
      const response = await api.post<ChatResponse>('/api/chat', params, {
        cancelToken: cancelTokenRef.current.token,
      });
      
      return response.data;
    } finally {
      // 请求完成后清理
      cancelTokenRef.current = null;
    }
  }, []);

  /**
   * 取消当前请求
   */
  const cancelChatRequest = useCallback(() => {
    if (cancelTokenRef.current) {
      cancelTokenRef.current.cancel('请求已取消');
      cancelTokenRef.current = null;
    }
  }, []);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (cancelTokenRef.current) {
        cancelTokenRef.current.cancel('组件卸载');
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
 * 兼容旧版本的导出（已废弃，请使用 useChatService）
 * @deprecated 使用 useChatService Hook 代替
 */
export const sendChatMessage = async (params: ChatRequest): Promise<ChatResponse> => {
  console.warn('sendChatMessage 已废弃，请使用 useChatService Hook');
  
  const cancelToken = axios.CancelToken.source();
  
  const response = await api.post<ChatResponse>('/api/chat', params, {
    cancelToken: cancelToken.token,
  });
  
  return response.data;
};

/**
 * 兼容旧版本的导出（已废弃）
 * @deprecated 使用 useChatService Hook 代替
 */
export const cancelChatRequest = () => {
  console.warn('cancelChatRequest 已废弃，请使用 useChatService Hook');
};
