'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ProgressEvent } from '@/lib/types';
import { 
  ChevronDown,
  Brain,
  CheckCircle2,
  Loader2,
  Terminal,
  FileText,
  AlertCircle,
  Monitor,
  X
} from 'lucide-react';

interface SSEMessageDisplayProps {
  events: ProgressEvent[];
  isStreaming: boolean;
}

interface GroupedEvents {
  thinking: ProgressEvent[];
  sandboxExec: ProgressEvent[];
  finalAnswer: ProgressEvent | null;
  files: ProgressEvent[];
  errors: ProgressEvent[];
  consoleOutput: ProgressEvent[];
}

function groupEvents(events: ProgressEvent[]): GroupedEvents {
  const grouped: GroupedEvents = {
    thinking: [],
    sandboxExec: [],
    finalAnswer: null,
    files: [],
    errors: [],
    consoleOutput: []
  };

  events.forEach(event => {
    switch (event.event_type) {
      case 'llm_thinking':
      case 'llm_streaming':
        grouped.thinking.push(event);
        break;
      case 'sandbox_exec':
      case 'iteration_start':
        grouped.sandboxExec.push(event);
        break;
      case 'final_answer':
        grouped.finalAnswer = event;
        break;
      case 'file_content':
        grouped.files.push(event);
        break;
      case 'error':
        grouped.errors.push(event);
        break;
      case 'console_output':
        grouped.consoleOutput.push(event);
        break;
    }
  });

  return grouped;
}

// AI 思考卡片（豆包风格）
const ThinkingCard = ({ 
  events, 
  isStreaming 
}: { 
  events: ProgressEvent[];
  isStreaming: boolean;
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  if (events.length === 0) return null;

  // 获取最新的思考内容
  const latestEvent = events[events.length - 1];
  const thinkingContent = latestEvent?.details?.full_content || latestEvent?.message || '';
  const phase = latestEvent?.details?.phase || '';

  // 判断是否正在思考
  const isActive = isStreaming && (
    phase.includes('streaming') || 
    phase.includes('progress') ||
    phase.includes('thinking')
  );

  // 判断是否完成
  const isComplete = phase.includes('complete');

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm">
      {/* 头部 */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          {isActive ? (
            <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
          ) : isComplete ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <Brain className="w-4 h-4 text-gray-400" />
          )}
          <span className="text-sm font-medium text-gray-700">
            {isActive ? '正在思考...' : isComplete ? '思考完成' : '思考过程'}
          </span>
          {events.length > 1 && (
            <span className="text-xs text-gray-400">
              {events.length} 步
            </span>
          )}
        </div>
        <ChevronDown 
          className={`w-4 h-4 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* 内容 */}
      {isExpanded && (
        <div 
          ref={contentRef}
          className="border-t border-gray-100 bg-gray-50 p-3 max-h-96 overflow-y-auto"
        >
          <div className="text-sm text-gray-600 whitespace-pre-wrap leading-relaxed">
            {thinkingContent}
          </div>
        </div>
      )}
    </div>
  );
};

// 沙盒执行卡片
const SandboxExecCard = ({ 
  events, 
  isStreaming 
}: { 
  events: ProgressEvent[];
  isStreaming: boolean;
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (events.length === 0) return null;

  // 获取最新状态
  const latestEvent = events[events.length - 1];
  const phase = latestEvent?.details?.phase || '';
  
  // 判断是否正在执行
  const isActive = isStreaming && (
    phase.includes('executing') || 
    phase.includes('waiting') ||
    phase.includes('progress')
  );

  // 判断是否完成
  const isComplete = phase.includes('complete') || phase.includes('done');

  // 统计各阶段数量
  const completedSteps = events.filter(e => 
    e.details?.phase?.includes('complete') || 
    e.message?.includes('✅')
  ).length;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm">
      {/* 头部 */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          {isActive ? (
            <Loader2 className="w-4 h-4 text-amber-500 animate-spin" />
          ) : isComplete ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <Terminal className="w-4 h-4 text-gray-400" />
          )}
          <span className="text-sm font-medium text-gray-700">
            {isActive ? '沙盒执行中...' : isComplete ? '执行完成' : '沙盒执行'}
          </span>
          {events.length > 1 && (
            <span className="text-xs text-gray-400">
              {completedSteps}/{events.length} 步
            </span>
          )}
        </div>
        <ChevronDown 
          className={`w-4 h-4 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* 内容 */}
      {isExpanded && (
        <div className="border-t border-gray-100 bg-gray-50 p-3 max-h-60 overflow-y-auto space-y-2">
          {events.map((event, index) => {
            const isStepComplete = event.message?.includes('✅');
            const isWaiting = event.details?.phase === 'waiting';
            
            return (
              <div 
                key={index}
                className="flex items-start gap-2 text-sm"
              >
                {isStepComplete ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-green-500 mt-0.5 shrink-0" />
                ) : isWaiting ? (
                  <Loader2 className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0 animate-spin" />
                ) : (
                  <div className="w-3.5 h-3.5 rounded-full bg-gray-300 mt-0.5 shrink-0" />
                )}
                <span className="text-gray-600">{event.message}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// 错误卡片
const ErrorCard = ({ events }: { events: ProgressEvent[] }) => {
  if (events.length === 0) return null;

  return (
    <div className="border border-red-200 rounded-lg overflow-hidden bg-red-50">
      <div className="flex items-center gap-2 p-3">
        <AlertCircle className="w-4 h-4 text-red-500" />
        <span className="text-sm font-medium text-red-700">执行出错</span>
      </div>
      <div className="border-t border-red-100 p-3 space-y-2">
        {events.map((event, index) => (
          <div key={index} className="text-sm text-red-600">
            {event.message}
          </div>
        ))}
      </div>
    </div>
  );
};

// 文件卡片
const FileCard = ({ event }: { event: ProgressEvent }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const content = event.details?.content || '';
  const previewContent = content.substring(0, 200) + (content.length > 200 ? '...' : '');

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-700">{event.message}</span>
          {event.details?.file_name && (
            <span className="text-xs text-gray-400">{event.details.file_name}</span>
          )}
        </div>
        <ChevronDown 
          className={`w-4 h-4 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      
      {isExpanded && (
        <div className="border-t border-gray-100 bg-gray-50 p-3">
          <pre className="text-xs text-gray-600 whitespace-pre-wrap overflow-x-auto max-h-60 overflow-y-auto">
            {content}
          </pre>
        </div>
      )}
    </div>
  );
};

// 控制台输出卡片
const ConsoleOutputCard = ({ 
  events, 
  isStreaming 
}: { 
  events: ProgressEvent[];
  isStreaming: boolean;
}) => {
  const [isExpanded, setIsExpanded] = useState(true); // 默认展开
  const [isPinned, setIsPinned] = useState(false); // 是否固定
  const [filter, setFilter] = useState(''); // 过滤器
  const consoleRef = useRef<HTMLDivElement>(null);
  
  if (events.length === 0) return null;
  
  // 合并所有消息
  const allMessages = events.map(e => e.message).join('');
  
  // 过滤事件
  const filteredEvents = filter 
    ? events.filter(e => e.message.toLowerCase().includes(filter.toLowerCase()))
    : events;
  
  const hasMessages = filteredEvents.length > 0;
  
  // 自动滚动到底部
  useEffect(() => {
    if (isExpanded && consoleRef.current && !filter) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [events, isExpanded, filter]);
  
  return (
    <div className={`border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm ${isPinned ? 'ring-2 ring-blue-500' : ''}`}>
      {/* 头部 */}
      <div className="flex items-center justify-between p-3 bg-gray-50">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 hover:bg-gray-100 transition-colors text-left flex-1"
        >
          <Monitor className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-700">
            控制台输出
          </span>
          <span className="text-xs text-gray-400">
            {events.length} 条
          </span>
        </button>
        
        <div className="flex items-center gap-1">
          {/* 过滤器 */}
          <input
            type="text"
            placeholder="过滤..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-2 py-1 text-xs border border-gray-300 rounded"
            onClick={(e) => e.stopPropagation()}
          />
          
          {/* 固定按钮 */}
          <button
            onClick={() => setIsPinned(!isPinned)}
            className="p-1 hover:bg-gray-200 rounded"
            title={isPinned ? '取消固定' : '固定窗口'}
          >
            📌
          </button>
          
          {/* 清空按钮 */}
          <button
            onClick={() => {
              // 清空控制台输出
              const consoleElement = document.getElementById('console-output-content');
              if (consoleElement) {
                consoleElement.innerHTML = '';
              }
            }}
            className="p-1 hover:bg-gray-200 rounded"
            title="清空控制台"
          >
            <X className="w-3 h-3" />
          </button>
          
          {/* 展开/收起按钮 */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-gray-200 rounded"
          >
            <ChevronDown 
              className={`w-4 h-4 transition-transform ${
                isExpanded ? 'rotate-180' : ''
              }`}
            />
          </button>
        </div>
      </div>
      
      {/* 内容 */}
      {isExpanded && (
        <div 
          ref={consoleRef}
          id="console-output-content"
          className="bg-gray-900 text-green-400 p-3 max-h-96 overflow-y-auto font-mono text-xs leading-relaxed"
          style={{ 
            minHeight: '200px',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word'
          }}
        >
          {hasMessages ? (
            filteredEvents.map((event, index) => (
              <div key={index} className="mb-1">
                {/* 时间戳（如果有） */}
                {event.details?.timestamp && (
                  <span className="text-gray-500 mr-2">
                    [{new Date(event.details.timestamp).toLocaleTimeString()}]
                  </span>
                )}
                {/* 消息内容 */}
                <span 
                  className={`
                    ${filter && event.message.toLowerCase().includes(filter.toLowerCase()) 
                      ? 'bg-yellow-900' 
                      : ''
                    }
                  `}
                >
                  {event.message}
                </span>
              </div>
            ))
          ) : (
            <div className="text-gray-500 text-center py-4">
              {filter ? '没有匹配的输出' : '暂无控制台输出'}
            </div>
          )}
        </div>
      )}
      
      {/* 底部统计 */}
      {isExpanded && hasMessages && (
        <div className="bg-gray-100 px-3 py-2 text-xs text-gray-600 flex justify-between">
          <span>总字符数: {allMessages.length}</span>
          <span>已捕获: {events.length} 条</span>
          {filter && (
            <span>过滤结果: {filteredEvents.length} 条</span>
          )}
        </div>
      )}
    </div>
  );
};

// 主组件
export default function SSEMessageDisplay({ events, isStreaming }: SSEMessageDisplayProps) {
  const grouped = groupEvents(events);
  
  return (
    <div className="space-y-3">
      {/* 错误提示（如果有） */}
      <ErrorCard events={grouped.errors} />
      
      {/* 控制台输出 - 新增 */}
      <ConsoleOutputCard events={grouped.consoleOutput} isStreaming={isStreaming} />
      
      {/* 沙盒执行进度 */}
      <SandboxExecCard events={grouped.sandboxExec} isStreaming={isStreaming} />
      
      {/* AI思考过程 */}
      <ThinkingCard events={grouped.thinking} isStreaming={isStreaming} />
      
      {/* 生成的文件（如果有） */}
      {grouped.files.length > 0 && (
        <div className="space-y-3">
          {grouped.files.map((event, index) => (
            <FileCard key={index} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}
