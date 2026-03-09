import React, { useState } from 'react';
import clsx from 'clsx';
import { Download, ChevronDown, Brain, CheckCircle2, Loader2, Terminal, FileText } from 'lucide-react';
import { Message, OutputFile, ProgressEvent } from '@/lib/types';
import { getFileDownloadUrl } from '@/lib/api';

interface MessageBubbleProps {
  message: Message;
}

// Thinking 折叠卡片（豆包风格）
const ThinkingSection = ({ events }: { events: ProgressEvent[] }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!events || events.length === 0) return null;

  const latestEvent = events[events.length - 1];
  const thinkingContent = latestEvent?.details?.full_content || latestEvent?.message || '';
  const phase = latestEvent?.details?.phase || '';
  const isComplete = phase.includes('complete');

  return (
    <div className="mb-3 border border-gray-200 rounded-lg bg-gray-50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2 hover:bg-gray-100 transition-colors text-left rounded-t-lg"
      >
        <div className="flex items-center gap-2">
          {isComplete ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <Brain className="w-4 h-4 text-gray-400" />
          )}
          <span className="text-xs font-medium text-gray-600">
            {isComplete ? '思考完成' : '思考过程'}
          </span>
        </div>
        <ChevronDown 
          className={`w-3.5 h-3.5 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      {isExpanded && (
        <div className="border-t border-gray-200 p-2 max-h-40 overflow-y-auto">
          <div className="text-xs text-gray-600 whitespace-pre-wrap">
            {thinkingContent}
          </div>
        </div>
      )}
    </div>
  );
};

// 沙盒执行折叠卡片
const ExecSection = ({ events }: { events: ProgressEvent[] }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!events || events.length === 0) return null;

  const latestEvent = events[events.length - 1];
  const phase = latestEvent?.details?.phase || '';
  const isComplete = phase.includes('complete') || phase.includes('done');

  const completedSteps = events.filter(e => 
    e.details?.phase?.includes('complete') || 
    e.message?.includes('✅')
  ).length;

  return (
    <div className="mb-3 border border-gray-200 rounded-lg bg-gray-50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2 hover:bg-gray-100 transition-colors text-left rounded-t-lg"
      >
        <div className="flex items-center gap-2">
          {isComplete ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <Terminal className="w-4 h-4 text-gray-400" />
          )}
          <span className="text-xs font-medium text-gray-600">
            {isComplete ? '执行完成' : '执行过程'}
          </span>
          {events.length > 1 && (
            <span className="text-xs text-gray-400">
              {completedSteps}/{events.length} 步
            </span>
          )}
        </div>
        <ChevronDown 
          className={`w-3.5 h-3.5 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      {isExpanded && (
        <div className="border-t border-gray-200 p-2 max-h-40 overflow-y-auto space-y-1">
          {events.slice(-10).map((event, index) => {
            const isStepComplete = event.message?.includes('✅');
            return (
              <div key={index} className="flex items-start gap-2 text-xs">
                {isStepComplete ? (
                  <CheckCircle2 className="w-3 h-3 text-green-500 mt-0.5 shrink-0" />
                ) : (
                  <div className="w-3 h-3 rounded-full bg-gray-300 mt-0.5 shrink-0" />
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

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  
  let displayContent = message.content;
  let answer: string | null = null;
  let taskType: string | null = null;
  let thinkingEvents: ProgressEvent[] = [];
  let execEvents: ProgressEvent[] = [];
  let outputFiles: OutputFile[] = [];
  
  if (!isUser) {
    try {
      const parsed = JSON.parse(message.content);
      
      // 提取答案
      if (parsed.answer) {
        answer = parsed.answer;
      } else if (parsed.summary?.answer) {
        answer = parsed.summary.answer;
      } else if (parsed.result?.merged_result?.general_qa_answer) {
        answer = parsed.result.merged_result.general_qa_answer;
      } else if (parsed.result?.merged_result?.general_qa_conclusion) {
        answer = parsed.result.merged_result.general_qa_conclusion;
      }
      
      // 如果找到答案，显示答案
      if (answer) {
        displayContent = answer;
      } else {
        // 没有明确的答案，显示执行摘要
        displayContent = `✅ 任务执行完成\n\n类型: ${parsed.task_type || '未知'}\n会话ID: ${parsed.session_id || '未知'}`;
      }
      
      // 提取任务类型
      taskType = parsed.task_type || parsed.summary?.task_type || null;
      
      // 提取进度事件并分类
      if (message.metadata?.progressEvents) {
        message.metadata.progressEvents.forEach(event => {
          if (event.event_type === 'llm_thinking' || event.event_type === 'llm_streaming') {
            thinkingEvents.push(event);
          } else if (event.event_type === 'sandbox_exec' || event.event_type === 'iteration_start') {
            execEvents.push(event);
          }
        });
      }
      
      // 提取输出文件
      if (message.metadata?.outputFiles && message.metadata.outputFiles.length > 0) {
        outputFiles = message.metadata.outputFiles;
      } else if (parsed.output_files && parsed.output_files.length > 0) {
        outputFiles = parsed.output_files;
      }
      
    } catch (e) {
      // 不是JSON，直接显示原始内容
    }
  }
  
  const getFileIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'csv data':
      case 'tsv data':
        return '📊';
      case 'json data':
        return '📋';
      case 'markdown':
      case 'text file':
        return '📄';
      case 'pdf report':
        return '📕';
      default:
        return '📁';
    }
  };
  
  return (
    <div
      className={clsx(
        'flex w-full mb-4',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      <div
        className={clsx(
          'max-w-[80%] rounded-lg px-4 py-2',
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-white text-gray-900 border border-gray-200 shadow-sm'
        )}
      >
        {/* Thinking 折叠卡片 */}
        {!isUser && thinkingEvents.length > 0 && (
          <ThinkingSection events={thinkingEvents} />
        )}

        {/* 沙盒执行折叠卡片 */}
        {!isUser && execEvents.length > 0 && (
          <ExecSection events={execEvents} />
        )}

        {/* 主要内容 */}
        <div className="whitespace-pre-wrap break-words">
          {displayContent}
        </div>
        
        {/* 输出文件 */}
        {outputFiles.length > 0 && (
          <div className="mt-3 p-2 bg-gray-50 rounded border border-gray-200">
            <div className="flex items-center gap-2 mb-2 text-xs font-medium text-gray-700">
              <FileText className="w-3.5 h-3.5" />
              <span>输出文件 ({outputFiles.length})</span>
            </div>
            <div className="space-y-1.5">
              {outputFiles.map((file, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-white rounded p-1.5 border border-gray-100"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-sm flex-shrink-0">{getFileIcon(file.type)}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-700 truncate">{file.name}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      const url = getFileDownloadUrl(message.metadata?.sessionId || '', file.relative_path);
                      window.open(url, '_blank');
                    }}
                    className="flex items-center gap-1 px-1.5 py-0.5 bg-blue-500 text-white rounded text-xs hover:bg-blue-600 transition-colors flex-shrink-0"
                  >
                    <Download className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* 元信息 */}
        {message.metadata && !isUser && (
          <div className="mt-2 pt-2 border-t border-gray-200 space-y-0.5">
            {taskType && (
              <div className="text-xs text-gray-500">
                <span className="font-medium">任务:</span> {taskType}
              </div>
            )}
            {message.metadata.status && (
              <div className="text-xs text-gray-500">
                <span className="font-medium">状态:</span>{' '}
                <span className={clsx(
                  message.metadata.status === 'done' && "text-green-600",
                  message.metadata.status === 'error' && "text-red-600"
                )}>
                  {message.metadata.status === 'done' ? '✓ 完成' : message.metadata.status}
                </span>
              </div>
            )}
          </div>
        )}
        
        {/* 时间戳 */}
        <div className={clsx(
          'text-xs mt-2',
          isUser ? 'text-blue-100' : 'text-gray-400'
        )}>
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
