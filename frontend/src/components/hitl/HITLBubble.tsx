import React, { useState } from 'react';
import { HITLRequest } from '@/types';
import TaskPlanViewer from './TaskPlanViewer';
import HITLInteraction from './HITLInteraction';

interface HITLBubbleProps {
  hitlRequest: HITLRequest;
  onConfirm?: (sessionId: string, feedback?: string, parameters?: Record<string, any>, taskMd?: string) => void;
  onReject?: (sessionId: string, feedback: string) => void;
  isCollapsed?: boolean;
}

const HITLBubble: React.FC<HITLBubbleProps> = ({
  hitlRequest,
  onConfirm,
  onReject,
  isCollapsed = false,
}) => {
  const [collapsed, setCollapsed] = useState(isCollapsed);
  const { task_md, iteration, max_iterations, session_id } = hitlRequest;

  console.log('[HITLBubble] Rendering with hitlRequest:', hitlRequest);
  console.log('[HITLBubble] task_md length:', task_md?.length);
  console.log('[HITLBubble] task_md content (first 200 chars):', task_md?.substring(0, 200));

  const handleConfirm = (feedback?: string, parameters?: Record<string, any>) => {
    console.log('[HITLBubble] handleConfirm called');
    onConfirm?.(session_id, feedback, parameters, task_md);
  };

  const handleReject = (feedback: string) => {
    console.log('[HITLBubble] handleReject called');
    onReject?.(session_id, feedback);
  };

  return (
    <div className="hitl-bubble">
      <div className="hitl-bubble-header" onClick={() => setCollapsed(!collapsed)}>
        <div className="hitl-bubble-title">
          <span className="hitl-icon">📋</span>
          <span>Task Plan Review</span>
          <span className="hitl-iteration">
            ({iteration + 1}/{max_iterations})
          </span>
        </div>
        <button className="hitl-toggle-btn">
          {collapsed ? '▶' : '▼'}
        </button>
      </div>
      
      {!collapsed && (
        <div className="hitl-bubble-content">
          <div className="hitl-plan-section">
            <TaskPlanViewer content={task_md} />
          </div>
          
          <div className="hitl-interaction-section">
            <HITLInteraction
              onConfirm={handleConfirm}
              onReject={handleReject}
              iteration={iteration}
            />
          </div>
        </div>
      )}
      
      <style jsx>{`
        .hitl-bubble {
          background: linear-gradient(135deg, #1a1a2a 0%, #2a2a3a 100%);
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid rgba(139, 92, 246, 0.3);
        }
        .hitl-bubble-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: rgba(139, 92, 246, 0.1);
          cursor: pointer;
        }
        .hitl-bubble-title {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #fff;
          font-weight: 500;
        }
        .hitl-icon {
          font-size: 18px;
        }
        .hitl-iteration {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.7);
          background: rgba(255, 255, 255, 0.1);
          padding: 2px 8px;
          border-radius: 10px;
        }
        .hitl-toggle-btn {
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.7);
          cursor: pointer;
          font-size: 14px;
          padding: 4px 8px;
        }
        .hitl-bubble-content {
          display: flex;
          flex-direction: column;
          max-height: 60vh;
        }
        .hitl-plan-section {
          flex: 1;
          overflow-y: auto;
          min-height: 0;
        }
        .hitl-interaction-section {
          flex-shrink: 0;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(0, 0, 0, 0.1);
        }
      `}</style>
    </div>
  );
};

export default HITLBubble;
