'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { StreamingMessage } from './StreamingMessage';
import { OutputFilesDisplay } from './OutputFilesDisplay';
import { Session, Message, AgentResponse, ProgressEvent, OutputFile } from '@/lib/types';
import { SessionStorage, createMessage } from '@/lib/storage';
import { AgentAPI } from '@/lib/api';
import { Terminal, FolderOpen, RefreshCw, Download, X } from 'lucide-react';
import { getFileDownloadUrl } from '@/lib/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ChatContainerProps {
  session: Session;
  onSessionUpdate: (session: Session) => void;
  pendingMessage?: string | null;
  onPendingMessageSent?: () => void;
}

export function ChatContainer({ session, onSessionUpdate, pendingMessage, onPendingMessageSent }: ChatContainerProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([]);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [showFilesModal, setShowFilesModal] = useState(false);
  
  const thinkingStartTimeRef = useRef<number | null>(null);
  const processedStreamingCount = useRef(0);
  
  // 手动刷新文件列表
  const fetchOutputFiles = async () => {
    if (!session.id) return;
    
    setIsLoadingFiles(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/${session.id}/files`);
      if (response.ok) {
        const data = await response.json();
        setOutputFiles(data.files || []);
        console.log(`[fetchOutputFiles] 找到 ${data.files?.length || 0} 个文件`);
        setShowFilesModal(true);
      } else {
        console.warn('[fetchOutputFiles] 查询失败，返回空列表');
        setOutputFiles([]);
        setShowFilesModal(true);
      }
    } catch (err) {
      console.warn('[fetchOutputFiles] 查询出错，返回空列表:', err);
      setOutputFiles([]);
      setShowFilesModal(true);
    } finally {
      setIsLoadingFiles(false);
    }
  };
  
  // 不再自动查询，只在用户点击时查询
  // useEffect(() => {
  //   if (!isStreaming && session.id) {
  //     const timer = setTimeout(() => {
  //       fetchOutputFiles();
  //     }, 1000);
  //     return () => clearTimeout(timer);
  //   }
  // }, [isStreaming, session.id]);

  // 处理待发送的消息（从模板选择触发）
  useEffect(() => {
    if (pendingMessage && !isStreaming) {
      handleSend(pendingMessage);
      onPendingMessageSent?.();
    }
  }, [pendingMessage]);

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
      {/* 流式消息显示 */}
      <div className="h-[calc(100vh-10rem)] overflow-y-auto">
          <MessageList
            messages={session.messages}
            isStreaming={isStreaming}
            streamingStatus=""
          />
          {isStreaming && <StreamingMessage
            status="Processing..."
              message={streamingMessage}
              progressEvents={progressEvents}
            />
          }
          
          {/* 控制台输出独立显示区域 */}
          {isStreaming && progressEvents.filter(e => e.event_type === 'console_output').length > 0 && (
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
        <MessageInput 
          onSend={handleSend} 
          disabled={isStreaming}
          sessionId={session.id}
          outputFilesCount={outputFiles.length}
          onFetchFiles={fetchOutputFiles}
          isLoadingFiles={isLoadingFiles}
        />
      </div>
      
      {showFilesModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden mx-4">
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold text-gray-800">
                  输出文件 ({outputFiles.length})
                </h3>
              </div>
              <button
                onClick={() => setShowFilesModal(false)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              {outputFiles.length === 0 ? (
                <p className="text-center text-gray-500 py-8">暂无可下载文件</p>
              ) : (
                <div className="space-y-2">
                  {outputFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-blue-300 transition-colors"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <span className="text-gray-500 text-sm flex-shrink-0 font-mono">
                          {index + 1}.
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-gray-800 truncate">
                            {file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {file.type} • {file.size_formatted}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          const url = getFileDownloadUrl(session.id, file.relative_path);
                          window.open(url, '_blank');
                        }}
                        className="flex items-center gap-1 px-3 py-1.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm flex-shrink-0"
                      >
                        <Download className="w-4 h-4" />
                        下载
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}