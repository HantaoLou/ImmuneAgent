'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { StreamingMessage } from './StreamingMessage';
import { OutputFilesDisplay } from './OutputFilesDisplay';
import { Session, Message, AgentResponse, ProgressEvent, OutputFile } from '@/lib/types';
import { SessionStorage, createMessage } from '@/lib/storage';
import { AgentAPI } from '@/lib/api';
import { Terminal } from 'lucide-react';

interface ChatContainerProps {
  session: Session;
  onSessionUpdate: (session: Session) => void;
}

export function ChatContainer({ session, onSessionUpdate }: ChatContainerProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([]);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  
  const thinkingStartTimeRef = useRef<number | null>(null);
  const processedStreamingCount = useRef(0);

    const handleSend = async (message: string) => {
        // 先清空控制台输出，准备接收新的
        setProgressEvents([]);
        
        setError(null);
        setIsStreaming(true);
        
        const userMessage = createMessage('user', message);
        const updatedMessages = [...session.messages, userMessage];
        const updatedSession = {
            ...session,
            messages: updatedMessages,
            title: session.messages.length === 0 ? message.substring(0, 30) : session.title,
        };
        
        onSessionUpdate(updatedSession);
        SessionStorage.saveSession(updatedSession);
    
        setIsStreaming(true);

    try {
      await AgentAPI.sendMessage(message, session.id, {
        onStatus: (data) => {
          console.log('[Status]', data.message || data.stage);
        },
        onProgress: (progressEvent: ProgressEvent) => {
          // 添加事件到列表
          setProgressEvents(prev => [...prev, progressEvent]);
          
          // 提取主要内容
          if (progressEvent.event_type === 'llm_thinking' && progressEvent.details?.full_content) {
            setStreamingMessage(progressEvent.details.full_content);
          } else if (progressEvent.event_type === 'final_answer') {
            setStreamingMessage(progressEvent.message);
          }
          
          console.log(`[Progress] ${progressEvent.event_type}: ${progressEvent.message?.substring(0, 50)}...`);
        },
        onOutputFiles: (data) => {
          console.log('[OutputFiles]', data.files.length, 'files');
          setOutputFiles(data.files);
        },
        onDone: (response: AgentResponse) => {
          console.log('收到响应:', response);
          
          const assistantMessage = createMessage('assistant', JSON.stringify(response));
          assistantMessage.metadata = {
            taskType: response.task_type,
            status: 'done',
            sessionId: response.session_id,
            progressEvents: progressEvents,
            outputFiles: outputFiles.length > 0 ? outputFiles : response.output_files,
          };
          
          const finalMessages = [...updatedMessages, assistantMessage];
          const finalSession = {
            ...updatedSession,
            messages: finalMessages,
          };
          
          onSessionUpdate(finalSession);
          SessionStorage.saveSession(finalSession);
          
          setTimeout(() => {
            setIsStreaming(false);
            setProgressEvents([]);
            setStreamingMessage('');
          }, 1000);
        },
        onError: (err) => {
          console.error('Agent error:', err);
          setError(err.error || '处理请求时发生错误');
          setIsStreaming(false);
        },
      });
    } catch (err: any) {
      console.error('Failed to send message:', err);
      setError(err.message || '无法连接到 Agent');
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <MessageList
        messages={session.messages}
        isStreaming={isStreaming}
        streamingStatus=""
      />
      
      {/* 流式消息显示 */}
      {isStreaming && (
        <div className="flex-shrink-0 px-4 py-2">
          <StreamingMessage
            status="Processing..."
            message={streamingMessage}
            progressEvents={progressEvents}
          />
          
          {/* 控制台输出独立显示区域 */}
          {progressEvents.filter(e => e.event_type === 'console_output').length > 0 && (
            <div className="mt-4">
              <div className="bg-white text-gray-900 border border-gray-200 shadow-sm rounded-lg">
                {/* 控制台头部 */}
                <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">
                      控制台输出
                    </span>
                    <span className="text-xs text-gray-400">
                      {progressEvents.filter(e => e.event_type === 'console_output').length} 条
                    </span>
                    <button
                      onClick={() => {
                        // 滚动到控制台区域
                        const consoleEl = document.getElementById('console-output-area');
                        if (consoleEl) {
                          consoleEl.scrollIntoView({ behavior: 'smooth' });
                        }
                      }}
                      className="text-xs text-blue-500 hover:text-blue-700 cursor-pointer"
                    >
                      滚动到此处
                    </button>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => {
                        // 清空控制台内容
                        const consoleEl = document.getElementById('console-output-content');
                        if (consoleEl) {
                          consoleEl.innerHTML = '';
                        }
                      }}
                      className="p-1 hover:bg-gray-200 rounded text-xs text-gray-600"
                    >
                      清空
                    </button>
                    <button
                      onClick={() => {
                        // 切换展开/收起状态
                        const consoleEl = document.getElementById('console-output-area');
                        const contentEl = document.getElementById('console-output-content');
                        if (consoleEl && contentEl) {
                          if (isExpanded) {
                            setIsExpanded(false);
                            contentEl.style.maxHeight = '0px';
                            consoleEl.classList.add('border-t');
                          } else {
                            setIsExpanded(true);
                            contentEl.style.maxHeight = '24rem';
                            consoleEl.classList.remove('border-t');
                          }
                        }
                      }}
                      className="p-1 hover:bg-gray-200 rounded text-xs text-gray-600"
                    >
                      {isExpanded ? '收起' : '展开'}
                    </button>
                  </div>
                </div>
                
                {/* 控制台内容区域 */}
                <div 
                  id="console-output-area"
                  className="bg-gray-900 text-green-400 p-4 max-h-96 overflow-y-auto font-mono text-xs leading-relaxed transition-all duration-200"
                  style={{ 
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}
                >
                  <div 
                    id="console-output-content"
                  >
                    {progressEvents
                      .filter(e => e.event_type === 'console_output' && !e.details?.cleared)
                      .map((event, index) => (
                        <div key={index} className="mb-2">
                          {/* 判断是否是过滤后的内容 */}
                          {event.details?.filtered ? (
                            // 过滤后的用户友好内容
                            <div className="text-yellow-300 bg-gray-800 p-2 rounded">
                              💬 {event.message}
                            </div>
                          ) : (
                            // 原始控制台输出（灰色显示）
                            <div className="text-gray-400 opacity-75">
                              {event.details?.timestamp && (
                                <span className="text-gray-500 mr-2">
                                  [{new Date(event.details.timestamp).toLocaleTimeString()}]
                                </span>
                              )}
                              {event.message}
                            </div>
                          )}
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {outputFiles.length > 0 && !isStreaming && (
        <div className="flex-shrink-0 px-4">
          <OutputFilesDisplay 
            files={outputFiles}
            sessionId={session.id}
          />
        </div>
      )}
      
      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200 flex-shrink-0">
          <p className="text-sm text-red-600 flex items-center gap-2">
            <span>❌</span>
            <span>{error}</span>
          </p>
        </div>
      )}
      
      <div className="flex-shrink-0">
        <MessageInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}