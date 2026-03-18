import React from 'react';

interface TaskPlanViewerProps {
  content: string;
}

const TaskPlanViewer: React.FC<TaskPlanViewerProps> = ({ content }) => {
  console.log('[TaskPlanViewer] Rendering with content length:', content?.length);
  console.log('[TaskPlanViewer] Content (first 200 chars):', content?.substring(0, 200));

  if (!content) {
    console.log('[TaskPlanViewer] No content to render');
    return <div className="task-plan-viewer">No task content available</div>;
  }

  return (
    <div className="task-plan-viewer">
      <pre className="task-content">{content}</pre>

      <style jsx>{`
        .task-plan-viewer {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 12px;
          padding: 16px;
          min-height: 100px;
        }
        .task-content {
          margin: 0;
          font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
          font-size: 13px;
          line-height: 1.6;
          color: rgba(255,255,255, 0.85);
          white-space: pre-wrap;
          word-wrap: break-word;
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
};

export default TaskPlanViewer;
