import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface TaskMDViewerProps {
  content: string;
  maxHeight?: string;
}

const TaskMDViewer: React.FC<TaskMDViewerProps> = ({ content, maxHeight = '300px' }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="task-md-viewer">
      <div className="task-md-header">
        <h4>Task Plan</h4>
        <button className="task-md-expand-btn" onClick={toggleExpand}>
          {isExpanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
      
      <div 
        className="task-md-content"
        style={{ maxHeight: isExpanded ? 'none' : maxHeight }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {content}
        </ReactMarkdown>
      </div>
      
      <style jsx>{`
        .task-md-viewer {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          margin-bottom: 16px;
          overflow: hidden;
        }
        .task-md-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .task-md-header h4 {
          margin: 0;
          color: rgba(255, 255, 255, 0.9);
          font-size: 14px;
          font-weight: 500;
        }
        .task-md-expand-btn {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          color: rgba(255, 255, 255, 0.7);
          padding: 4px 12px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }
        .task-md-expand-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .task-md-content {
          padding: 16px;
          overflow-y: auto;
          color: rgba(255, 255, 255, 0.85);
          font-size: 14px;
          line-height: 1.6;
        }
        .task-md-content h1 {
          color: #a855f7;
          font-size: 18px;
          margin-bottom: 12px;
        }
        .task-md-content h2 {
          color: #c084fc;
          font-size: 16px;
          margin-top: 16px;
          margin-bottom: 8px;
        }
        .task-md-content h3 {
          color: #7dd3fc;
          font-size: 14px;
          margin-top: 12px;
          margin-bottom: 6px;
        }
        .task-md-content ul, .task-md-content ol {
          padding-left: 20px;
        }
        .task-md-content li {
          margin-bottom: 4px;
        }
        .task-md-content code {
          background: rgba(0, 0, 0, 0.3);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: 'Consolas', monospace;
          font-size: 13px;
        }
        .task-md-content pre {
          background: rgba(0, 0, 0, 0.3);
          padding: 12px;
          border-radius: 8px;
          overflow-x: auto;
        }
        .task-md-content pre code {
          background: none;
          padding: 0;
        }
        .task-md-content a {
          color: #60a5fa;
        }
        .task-md-content blockquote {
          border-left: 3px solid rgba(255, 255, 255, 0.3);
          padding-left: 12px;
          margin: 8px 0;
          color: rgba(255, 255, 255, 0.7);
        }
        .task-md-content table {
          width: 100%;
          border-collapse: collapse;
          margin: 8px 0;
        }
        .task-md-content th, .task-md-content td {
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 8px;
          text-align: left;
        }
        .task-md-content th {
          background: rgba(255, 255, 255, 0.1);
        }
      `}</style>
    </div>
  );
};

export default TaskMDViewer;
