"use client";

import { AgentResponse, SSEEvent, ProgressEvent, Message } from '@/lib/types';
import { SessionStorage, createMessage } from '@/lib/storage';
import { Terminal } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function getFileDownloadUrl(sessionId: string, filename: string): string {
  return `${API_BASE_URL}/api/download/${sessionId}/${filename}`;
}

interface SSECallbacks {
  onStatus?: (data: any) => void;
  onProgress?: (data: ProgressEvent) => void;
  onOutputFiles?: (data: { files: any; count: number; session_id: string }) => void;
  onDone?: (response: AgentResponse) => void;
  onError?: (error: any) => void;
}

export async function connectToAgent(
  message: string,
  sessionId?: string,
  callbacks: SSECallbacks = {}
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }
  const decoder = new TextDecoder();
  
  let buffer = '';
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          continue;
        }
        
        if (line.startsWith('data: ')) {
          const data = line.substring(6).trim();
          try {
            const parsed = JSON.parse(data);
            
            // 处理进度事件
            if (parsed.event_type && parsed.message && callbacks.onProgress) {
              callbacks.onProgress(parsed);
            }
            
            // 处理状态事件
            if (callbacks.onStatus) {
              callbacks.onStatus({ stage: parsed.stage, message: parsed.message });
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } catch (e) {
    console.error('SSE Reader error:', e);
  }
}

export class AgentAPI {
  static async sendMessage(
    message: string,
    sessionId?: string,
    callbacks: SSECallbacks = {}
  ): Promise<void> {
    return connectToAgent(message, sessionId, callbacks);
  }
  
  static async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}

// 从存储中获取会话信息
export async function getSessionFromStorage(sessionId: string): Promise<any> {
  try {
    const sessions = localStorage.getItem('agent_sessions');
    if (sessions) {
      const sessionsData = JSON.parse(sessions);
      return sessionsData.find((s: any) => s.session_id === sessionId);
    }
    return null;
  } catch (error) {
      console.error('Failed to get session:', error);
      return null;
  }
}

// 创建或获取会话
export async function getOrCreateSession(sessionId?: string): Promise<any> {
  if (sessionId) {
    return await getSessionFromStorage(sessionId);
  }
  
  // 创建新会话
  const newSession = {
    session_id: sessionId || crypto.randomUUID(),
    message_count: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    first_message: null,
    title: 'New Chat',
  };
  
  // 保存会话
  try {
    const sessions = localStorage.getItem('agent_sessions');
    let sessionsData = [];
    if (sessions) {
      sessionsData = JSON.parse(sessions);
    }
    
    // 如果会话已存在，更新它
    const existingIndex = sessionsData.findIndex((s: any) => s.session_id === sessionId);
    if (existingIndex !== -1) {
      sessionsData[existingIndex] = newSession;
    } else {
      sessionsData.push(newSession);
    }
    
    localStorage.setItem('agent_sessions', JSON.stringify(sessionsData));
    return newSession;
  } catch (error) {
      console.error('Failed to save session:', error);
      return newSession;
  }
}

// 保存消息到会话
export async function saveMessageToSession(
  sessionId: string, 
    message: string, 
    isUser: boolean = true
): Promise<void> {
  try {
    const session = await getOrCreateSession(sessionId);
    if (session) {
      session.message_count += 1;
      if (!session.first_message && isUser) {
        session.first_message = message.substring(0, 30);
      }
      
      session.updated_at = new Date().toISOString();
      session.updated_at = new Date().toISOString();
      
      const updatedSession = {
        ...session,
        messages: [...(session.messages || []), createMessage('assistant', '')],
      };
      
      await saveSession(session);
    }
  } catch (error) {
      console.error('Failed to save message:', error);
    }
}

// 保存会话
export async function saveSession(session: any): Promise<void> {
  try {
    const sessions = localStorage.getItem('agent_sessions');
    let sessionsData = [];
    if (sessions) {
      sessionsData = JSON.parse(sessions);
    }
    
    const index = sessionsData.findIndex((s: any) => s.session_id === session.session_id);
    if (index !== -1) {
      sessionsData[index] = session;
    }
    
    localStorage.setItem('agent_sessions', JSON.stringify(sessionsData));
  } catch (error) {
    console.error('Failed to save session:', error);
  }
}