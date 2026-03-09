import React, { useState } from 'react';
import clsx from 'clsx';
import { ProgressEvent } from '@/lib/types';
import { 
  ChevronDown,
  CheckCircle2,
  Loader2,
  Terminal
} from 'lucide-react';

interface StreamingMessageProps {
  status?: string;
  message?: string;
  progressEvents?: ProgressEvent[];
}

// Exec 折叠卡片
const ExecSection = ({ events }: { events: ProgressEvent[] }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (events.length === 0) return null;
  
  const latestEvent = events[events.length - 1];
  const phase = latestEvent?.details?.phase || '';
  const isActive = phase.includes('executing') || phase.includes('waiting') || phase.includes('progress');
  const isComplete = phase.includes('complete') || phase.includes('done');
  
  const completedSteps = events.filter(e => 
    e.details?.phase?.includes('complete') || 
    e.message?.includes('✅')
  ).length;
  
  return (
    <div className="mb-3 border border-gray-200 rounded-lg bg-white overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2.5 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          {isActive ? (
            <Loader2 className="w-4 h-4 text-amber-500 animate-spin" />
          ) : isComplete ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <Terminal className="w-4 h-4 text-gray-400" />
          )}
          <span className="text-xs font-medium text-gray-700">
            {isActive ? '沙盒执行中...' : isComplete ? '执行完成' : '执行过程'}
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
        <div className="border-t border-gray-100 bg-gray-50 p-2 max-h-32 overflow-y-auto space-y-1">
          {events.slice(-10).map((event, index) => {
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

export function StreamingMessage({ status = 'Thinking...', message, progressEvents = [] }: StreamingMessageProps) {
  // 分类事件
  const execEvents = progressEvents.filter(e => 
    e.event_type === 'sandbox_exec' || e.event_type === 'iteration_start'
  );
  
  return (
    <div className="flex justify-start w-full mb-4">
      <div className="bg-white text-gray-900 border border-gray-200 shadow-sm rounded-lg px-4 py-2 max-w-[80%]">
        {/* Exec 折叠卡片 */}
        <ExecSection events={execEvents} />
        
        {/* 状态指示 */}
        {!message && progressEvents.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span>{status}</span>
          </div>
        )}
        
        {/* 时间戳 */}
        <div className="text-xs text-gray-400 mt-2">
          {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}