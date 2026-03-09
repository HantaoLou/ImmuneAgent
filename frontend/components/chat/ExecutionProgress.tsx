'use client';

import React, { useState, useEffect } from 'react';
import { ProgressEvent, ProgressEventType } from '@/lib/types';
import ThinkingChain from './ThinkingChain';

interface ExecutionProgressProps {
  progressEvents: ProgressEvent[];
  isStreaming: boolean;
}

const getEventIcon = (eventType: string): string => {
  switch (eventType) {
    case 'node_start':
      return '▶️';
    case 'node_progress':
      return '🔄';
    case 'node_complete':
      return '✅';
    case 'task_start':
      return '🎯';
    case 'task_progress':
      return '⚙️';
    case 'task_complete':
      return '✨';
    case 'code_generation':
      return '💻';
    case 'code_execution':
      return '🚀';
    case 'tool_call':
      return '🔧';
    case 'error':
      return '❌';
    case 'info':
      return 'ℹ️';
    case 'llm_thinking':
      return '🧠';
    case 'llm_reasoning':
      return '💭';
    case 'llm_streaming':
      return '✍️';
    case 'tool_result':
      return '📤';
    case 'subgraph_step':
      return '🔷';
    case 'knowledge_retrieval':
      return '🔍';
    case 'analysis_progress':
      return '📊';
    default:
      return '📍';
  }
};

const getEventColor = (eventType: string): string => {
  switch (eventType) {
    case 'node_start':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    case 'node_progress':
      return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    case 'node_complete':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'task_start':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    case 'task_progress':
      return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    case 'task_complete':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'code_generation':
      return 'text-purple-600 bg-purple-50 border-purple-200';
    case 'code_execution':
      return 'text-purple-600 bg-purple-50 border-purple-200';
    case 'tool_call':
      return 'text-indigo-600 bg-indigo-50 border-indigo-200';
    case 'error':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'info':
      return 'text-gray-600 bg-gray-50 border-gray-200';
    case 'llm_thinking':
      return 'text-violet-600 bg-violet-50 border-violet-200';
    case 'llm_reasoning':
      return 'text-violet-600 bg-violet-50 border-violet-200';
    case 'llm_streaming':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    case 'tool_result':
      return 'text-emerald-600 bg-emerald-50 border-emerald-200';
    case 'subgraph_step':
      return 'text-cyan-600 bg-cyan-50 border-cyan-200';
    case 'knowledge_retrieval':
      return 'text-amber-600 bg-amber-50 border-amber-200';
    case 'analysis_progress':
      return 'text-teal-600 bg-teal-50 border-teal-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

const ThinkingProcess = ({ events }: { events: ProgressEvent[] }) => {
  // 只取思考类事件，去重
  const thinkingEvents = events
    .filter(e => e.event_type === 'llm_thinking' || e.event_type === 'llm_reasoning')
    .slice(-3); // 只显示最近3条

  if (thinkingEvents.length === 0) return null;

  return (
    <div className="mb-3 bg-violet-50 border border-violet-200 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-base">🧠</span>
        <span className="text-sm font-semibold text-violet-700">AI 思考过程</span>
      </div>
      <div className="space-y-1.5">
        {thinkingEvents.map((event, index) => (
          <div key={index} className="text-xs text-violet-700">
            <div className="flex items-start gap-2 mb-1">
              <span className="text-violet-400 mt-0.5">●</span>
              <div className="flex-1 leading-relaxed font-medium">
                {event.message.replace(/[💭🧠]/g, '').trim()}
              </div>
            </div>
            {/* 展示产物详情 */}
            {event.details && Object.keys(event.details).length > 0 && (
              <div className="ml-4 mt-1 pl-4 border-l-2 border-violet-300 text-violet-600 text-xs">
                {event.details.fact_count && (
                  <div>📊 提取了 {event.details.fact_count} 个关键事实</div>
                )}
                {event.details.domains && (
                  <div>📚 检索领域: {event.details.domains.join(', ')}</div>
                )}
                {event.details.conclusion && (
                  <div className="truncate">💡 结论: {event.details.conclusion}</div>
                )}
                {event.details.answer_length && (
                  <div>📝 答案长度: {event.details.answer_length} 字符</div>
                )}
                {event.details.knowledge_preview && (
                  <div className="truncate">📖 知识: {event.details.knowledge_preview}</div>
                )}
                {event.details.key_facts && (
                  <div className="truncate">🔍 事实: {JSON.stringify(event.details.key_facts).substring(0, 100)}...</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const ExecutionSteps = ({ events }: { events: ProgressEvent[] }) => {
  // 过滤掉思考类事件，只保留执行步骤
  const executionEvents = events
    .filter(e => 
      e.event_type !== 'llm_thinking' && 
      e.event_type !== 'llm_reasoning' &&
      e.event_type !== 'llm_streaming'
    )
    .slice(-5); // 只显示最近5条

  if (executionEvents.length === 0) return null;

  return (
    <div className="space-y-1.5">
      {executionEvents.map((event, index) => (
        <div
          key={index}
          className={`flex items-center gap-2 p-2 rounded border text-xs ${getEventColor(event.event_type)}`}
        >
          <span className="text-base">{getEventIcon(event.event_type)}</span>
          <div className="flex-1 leading-relaxed">
            {event.message}
          </div>
          {event.progress_percent !== undefined && event.progress_percent > 0 && (
            <div className="text-xs font-semibold opacity-70">
              {event.progress_percent}%
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default function ExecutionProgress({
  progressEvents,
  isStreaming,
}: ExecutionProgressProps) {
  const [showDetails, setShowDetails] = useState(false);

  if (progressEvents.length === 0 && !isStreaming) {
    return null;
  }

  // 只显示最近的事件
  const recentEvents = progressEvents.slice(-10);
  const latestEvent = recentEvents[recentEvents.length - 1];
  const overallProgress = latestEvent?.progress_percent || 0;

  // 计算完成度（如果有task_complete事件，进度为100%）
  const isComplete = recentEvents.some(e => e.event_type === 'task_complete');
  const displayProgress = isComplete ? 100 : overallProgress;

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm">
      {/* 头部 */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isStreaming && (
            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
          )}
          <span className="text-xs font-semibold text-gray-700">
            {isComplete ? '✅ 执行完成' : 'Agent 执行中...'}
          </span>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          {showDetails ? '收起' : '详情'}
        </button>
      </div>

      {/* 进度条 */}
      {displayProgress > 0 && (
        <div className="mb-2">
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className={`h-1.5 rounded-full transition-all duration-300 ${
                isComplete ? 'bg-green-500' : 'bg-blue-600'
              }`}
              style={{ width: `${displayProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* 最新状态（默认显示） */}
      {!showDetails && latestEvent && (
        <div className="text-xs text-gray-600 flex items-center gap-2">
          <span>{getEventIcon(latestEvent.event_type)}</span>
          <span className="flex-1">{latestEvent.message}</span>
        </div>
      )}

      {/* 思考链（始终显示） */}
      {recentEvents.filter(e => 
        e.event_type === 'llm_thinking' || 
        e.event_type === 'llm_streaming'
      ).length > 0 && (
        <div className="mt-3">
          <ThinkingChain events={recentEvents} isStreaming={isStreaming} />
        </div>
      )}

      {/* 详细信息（展开显示） */}
      {showDetails && (
        <div className="mt-3 space-y-3">
          {/* 执行步骤 */}
          <ExecutionSteps events={recentEvents} />

          {/* 事件统计 */}
          <div className="text-xs text-gray-400 text-right pt-2 border-t border-gray-200">
            共 {progressEvents.length} 条事件
          </div>
        </div>
      )}
    </div>
  );
}
