'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ProgressEvent } from '@/lib/types';

interface ThinkingChainProps {
  events: ProgressEvent[];
  isStreaming: boolean;
}

interface ThinkingStep {
  id: string;
  eventType: 'llm_thinking' | 'llm_streaming';
  phase: string;
  message: string;
  fullMessage: string;
  timestamp: string;
  displayedText: string;
  isComplete: boolean;
  details?: Record<string, any>;
}

const TypewriterText: React.FC<{
  text: string;
  speed?: number;
  onComplete?: () => void;
  isActive: boolean;
}> = ({ text, speed = 30, onComplete, isActive }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setDisplayedText(text);
      return;
    }

    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.substring(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, speed);

      return () => clearTimeout(timer);
    } else if (onComplete) {
      onComplete();
    }
  }, [currentIndex, text, speed, isActive, onComplete]);

  // Reset when text changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
  }, [text]);

  return (
    <span>
      {displayedText}
      {isActive && currentIndex < text.length && (
        <span className="inline-block w-0.5 h-4 bg-violet-600 ml-0.5 animate-pulse" />
      )}
    </span>
  );
};

const PhaseIcon: React.FC<{ phase: string; isComplete: boolean }> = ({ phase, isComplete }) => {
  const getIcon = () => {
    switch (phase) {
      case 'thinking_start':
      case 'streaming_start':
        return '🤔';
      case 'thinking_progress':
      case 'streaming_progress':
        return '💭';
      case 'thinking_complete':
      case 'streaming_complete':
        return '✅';
      default:
        return '🧠';
    }
  };

  return (
    <span className={`text-lg transition-transform duration-200 ${isComplete ? 'scale-110' : ''}`}>
      {isComplete ? '✅' : getIcon()}
    </span>
  );
};

const ThinkingStepCard: React.FC<{
  step: ThinkingStep;
  isActive: boolean;
  index: number;
}> = ({ step, isActive, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={`relative transition-all duration-300 ${
        isActive ? 'scale-105' : ''
      }`}
      style={{
        animationDelay: `${index * 100}ms`,
      }}
    >
      {/* 连接线 */}
      {index > 0 && (
        <div className="absolute left-4 -top-2 w-0.5 h-2 bg-gradient-to-b from-violet-300 to-transparent" />
      )}

      {/* 卡片内容 */}
      <div
        className={`bg-white border-2 rounded-lg p-3 shadow-sm transition-all duration-300 ${
          isActive
            ? 'border-violet-400 shadow-md'
            : step.isComplete
            ? 'border-green-300 bg-green-50'
            : 'border-gray-200'
        }`}
      >
        {/* 头部 */}
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            <PhaseIcon phase={step.phase} isComplete={step.isComplete} />
          </div>

          <div className="flex-1 min-w-0">
            {/* 消息内容 */}
            <div className="text-sm text-gray-700 leading-relaxed">
              {isActive && !step.isComplete ? (
                <TypewriterText text={step.fullMessage} speed={25} isActive={isActive} />
              ) : (
                <span>{step.fullMessage}</span>
              )}
            </div>

            {/* 详情（可展开） */}
            {step.details && Object.keys(step.details).length > 0 && (
              <div className="mt-2">
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-violet-600 hover:text-violet-800 flex items-center gap-1"
                >
                  <span
                    className={`inline-block transition-transform duration-200 ${
                      isExpanded ? 'rotate-90' : ''
                    }`}
                  >
                    ▶
                  </span>
                  {isExpanded ? '收起详情' : '查看详情'}
                </button>

                {isExpanded && (
                  <div className="mt-2 pl-4 border-l-2 border-violet-300 space-y-1 text-xs text-gray-600">
                    {step.details.chunk_number && step.details.total_chunks && (
                      <div className="flex items-center gap-2">
                        <span className="text-violet-500">📊</span>
                        <span>
                          进度: {step.details.chunk_number}/{step.details.total_chunks}
                        </span>
                      </div>
                    )}
                    {step.details.elapsed_seconds && (
                      <div className="flex items-center gap-2">
                        <span className="text-violet-500">⏱️</span>
                        <span>耗时: {step.details.elapsed_seconds}s</span>
                      </div>
                    )}
                    {step.details.total_length && (
                      <div className="flex items-center gap-2">
                        <span className="text-violet-500">📏</span>
                        <span>长度: {step.details.total_length} 字符</span>
                      </div>
                    )}
                    {step.details.total_time && (
                      <div className="flex items-center gap-2">
                        <span className="text-violet-500">⏰</span>
                        <span>总时长: {step.details.total_time}s</span>
                      </div>
                    )}
                    {step.details.context && (
                      <div className="flex items-center gap-2">
                        <span className="text-violet-500">🏷️</span>
                        <span>上下文: {step.details.context}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* 时间戳 */}
            <div className="mt-1.5 text-xs text-gray-400">
              {new Date(step.timestamp).toLocaleTimeString('zh-CN')}
            </div>
          </div>
        </div>

        {/* 活跃指示器 */}
        {isActive && !step.isComplete && (
          <div className="absolute -right-1 -top-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-violet-500" />
          </div>
        )}
      </div>
    </div>
  );
};

const ThinkingChain: React.FC<ThinkingChainProps> = ({ events, isStreaming }) => {
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [activeStepIndex, setActiveStepIndex] = useState(-1);

  // 处理事件，转换为思维步骤
  useEffect(() => {
    const steps: ThinkingStep[] = [];
    let currentGroup: ProgressEvent[] = [];

    events.forEach((event) => {
      if (event.event_type === 'llm_thinking' || event.event_type === 'llm_streaming') {
        const phase = event.details?.phase || '';
        
        // 移除 emoji 前缀
        const cleanMessage = event.message.replace(/[💭🧠🤔✅]/g, '').trim();
        
        steps.push({
          id: `${event.timestamp}-${event.event_type}`,
          eventType: event.event_type,
          phase: phase,
          message: cleanMessage.substring(0, 150),
          fullMessage: cleanMessage,
          timestamp: event.timestamp,
          displayedText: '',
          isComplete: phase.includes('complete') || !isStreaming,
          details: event.details,
        });
      }
    });

    // 只保留最近的思维链（按开始-进行-完成分组）
    const recentSteps = steps.slice(-10); // 最多显示 10 个步骤

    setThinkingSteps(recentSteps);

    // 找到活跃的步骤（最后一个未完成的）
    const activeIndex = recentSteps.findIndex((step) => !step.isComplete);
    setActiveStepIndex(activeIndex >= 0 ? activeIndex : recentSteps.length - 1);

  }, [events, isStreaming]);

  if (thinkingSteps.length === 0) {
    return null;
  }

  return (
    <div className="bg-gradient-to-br from-violet-50 to-purple-50 border-2 border-violet-200 rounded-lg p-4">
      {/* 头部 */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🧠</span>
          <div>
            <h3 className="text-sm font-bold text-violet-800">AI 思维链</h3>
            <p className="text-xs text-violet-600">
              {isStreaming ? '正在思考...' : '思考完成'}
            </p>
          </div>
        </div>

        {/* 进度指示 */}
        <div className="flex items-center gap-2">
          {isStreaming && (
            <>
              <div className="flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-violet-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500" />
              </div>
              <span className="text-xs text-violet-600">
                {thinkingSteps.filter(s => s.isComplete).length}/{thinkingSteps.length}
              </span>
            </>
          )}
        </div>
      </div>

      {/* 思维步骤列表 */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {thinkingSteps.map((step, index) => (
          <ThinkingStepCard
            key={step.id}
            step={step}
            isActive={index === activeStepIndex && isStreaming}
            index={index}
          />
        ))}
      </div>

      {/* 底部信息 */}
      <div className="mt-3 pt-3 border-t border-violet-200 flex items-center justify-between text-xs text-violet-600">
        <span>共 {thinkingSteps.length} 个思维片段</span>
        {isStreaming && (
          <span className="flex items-center gap-1">
            <span className="inline-block w-1 h-1 rounded-full bg-violet-400 animate-pulse" />
            实时更新中
          </span>
        )}
      </div>
    </div>
  );
};

export default ThinkingChain;
