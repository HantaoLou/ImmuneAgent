import React from 'react';
import { MissingParameter } from '@/types';

interface MissingParametersPanelProps {
  parameters: MissingParameter[];
}

const MissingParametersPanel: React.FC<MissingParametersPanelProps> = ({ parameters }) => {
  if (!parameters || parameters.length === 0) {
    return null;
  }

  return (
    <div className="missing-parameters-panel">
      <div className="missing-parameters-header">
        <span className="warning-icon">⚠️</span>
        <h4>Missing Parameters</h4>
      </div>
      
      <div className="missing-parameters-list">
        {parameters.map((param, index) => (
          <div key={index} className="missing-parameter-item">
            <div className="param-header">
              <span className="param-name">{param.name}</span>
              <span className={`param-required ${param.required ? 'required' : 'optional'}`}>
                {param.required ? 'Required' : 'Optional'}
              </span>
            </div>
            <div className="param-description">{param.description}</div>
            <div className="param-type">Type: {param.type}</div>
          </div>
        ))}
      </div>
      
      <style jsx>{`
        .missing-parameters-panel {
          background: rgba(255, 193, 7, 0.1);
          border: 1px solid rgba(255, 193, 7, 0.3);
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }
        .missing-parameters-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }
        .warning-icon {
          font-size: 18px;
        }
        .missing-parameters-header h4 {
          margin: 0;
          color: #ffc107;
          font-size: 14px;
          font-weight: 500;
        }
        .missing-parameters-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .missing-parameter-item {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 6px;
          padding: 10px;
        }
        .param-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 4px;
        }
        .param-name {
          font-weight: 500;
          color: rgba(255, 255, 255, 0.9);
          font-family: 'Consolas', monospace;
        }
        .param-required {
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 4px;
          text-transform: uppercase;
        }
        .param-required.required {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        .param-required.optional {
          background: rgba(59, 130, 246, 0.2);
          color: #3b82f6;
        }
        .param-description {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.7);
          margin-bottom: 4px;
        }
        .param-type {
          font-size: 11px;
          color: rgba(255, 255, 255, 0.5);
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

export default MissingParametersPanel;
