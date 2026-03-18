import React, { useState } from 'react';

interface HITLInteractionProps {
  onConfirm: (feedback?: string, parameters?: Record<string, any>) => void | Promise<void>;
  onReject: (feedback: string) => void | Promise<void>;
  iteration: number;
  isSubmitting?: boolean;
}

const HITLInteraction: React.FC<HITLInteractionProps> = ({
  onConfirm,
  onReject,
  iteration,
  isSubmitting = false,
}) => {
  const [feedback, setFeedback] = useState('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [showParams, setShowParams] = useState(false);
  const [submitting, setIsSubmitting] = useState(false);

  const handleConfirm = async () => {
    setIsSubmitting(true);
    try {
      await onConfirm(feedback || undefined, Object.keys(parameters).length > 0 ? parameters : undefined);
    } catch (error) {
      console.error('Failed to submit confirmation:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReject = async () => {
    const hasContent = feedback.trim() || Object.keys(parameters).length > 0;
    if (!hasContent) {
      alert('Please provide feedback or parameters for modifications');
      return;
    }
    
    setIsSubmitting(true);
    try {
      await onReject(feedback);
    } catch (error) {
      console.error('Failed to submit rejection:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleParameterChange = (key: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const [paramKey, setParamKey] = useState('');
  const [paramValue, setParamValue] = useState('');

  const addParameter = () => {
    if (paramKey.trim() && paramValue.trim()) {
      handleParameterChange(paramKey.trim(), paramValue.trim());
      setParamKey('');
      setParamValue('');
    }
  };

  const hasModificationContent = feedback.trim() || Object.keys(parameters).length > 0;

  return (
    <div className="hitl-interaction">
      <div className="hitl-feedback-section">
        <label className="hitl-label">Your Feedback (optional)</label>
        <textarea
          className="hitl-feedback-input"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="Enter your modifications or suggestions..."
          rows={3}
        />
      </div>
      
      <button 
        className="hitl-params-toggle"
        onClick={() => setShowParams(!showParams)}
      >
        {showParams ? '▼ Hide Parameters' : '▶ Add Parameters'}
      </button>
      
      {showParams && (
        <div className="hitl-params-input">
          <input
            type="text"
            className="hitl-param-key"
            value={paramKey}
            onChange={(e) => setParamKey(e.target.value)}
            placeholder="Parameter name"
            onKeyDown={(e) => e.key === 'Enter' && addParameter()}
          />
          <input
            type="text"
            className="hitl-param-value"
            value={paramValue}
            onChange={(e) => setParamValue(e.target.value)}
            placeholder="Parameter value"
            onKeyDown={(e) => e.key === 'Enter' && addParameter()}
          />
          <button className="hitl-param-add" onClick={addParameter}>+</button>
        </div>
      )}
      
      {Object.keys(parameters).length > 0 && (
        <div className="hitl-params-list">
          {Object.entries(parameters).map(([key, value]) => (
            <div key={key} className="hitl-param-item">
              <span className="hitl-param-name">{key}</span>
              <span className="hitl-param-val">{String(value)}</span>
              <button 
                className="hitl-param-remove"
                onClick={() => {
                  const newParams = { ...parameters };
                  delete newParams[key];
                  setParameters(newParams);
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      
      <div className="hitl-buttons">
        <button
          className="hitl-btn hitl-btn-confirm"
          onClick={handleConfirm}
          disabled={submitting || isSubmitting}
        >
          {submitting || isSubmitting ? 'Submitting...' : '✓ Confirm & Execute'}
        </button>
        <button
          className="hitl-btn hitl-btn-reject"
          onClick={handleReject}
          disabled={submitting || isSubmitting || !hasModificationContent}
          title={!hasModificationContent ? 'Please provide feedback or parameters' : ''}
        >
          {submitting || isSubmitting ? 'Submitting...' : '✏ Request Modification'}
        </button>
      </div>
      
      <style jsx>{`
        .hitl-interaction {
          margin-top: 16px;
          padding: 16px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
        }
        .hitl-feedback-section {
          margin-bottom: 12px;
        }
        .hitl-label {
          display: block;
          font-size: 12px;
          color: rgba(255, 255, 255, 0.7);
          margin-bottom: 8px;
        }
        .hitl-feedback-input {
          width: 100%;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 6px;
          padding: 10px 12px;
          color: #fff;
          font-size: 14px;
          resize: vertical;
          min-height: 60px;
        }
        .hitl-params-toggle {
          background: transparent;
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.7);
          padding: 4px 12px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
          margin-bottom: 8px;
        }
        .hitl-params-toggle:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        .hitl-params-input {
          display: flex;
          gap: 8px;
          margin-bottom: 8px;
        }
        .hitl-param-key {
          flex: 1;
          padding: 8px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          color: #fff;
          font-size: 13px;
        }
        .hitl-param-value {
          flex: 2;
          padding: 8px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          color: #fff;
          font-size: 13px;
        }
        .hitl-param-add {
          padding: 8px 12px;
          background: rgba(59, 130, 246, 0.2);
          border: 1px solid rgba(59, 130, 246, 0.5);
          border-radius: 4px;
          color: #3b82f6;
          cursor: pointer;
        }
        .hitl-params-list {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 12px;
        }
        .hitl-param-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 4px 8px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 4px;
          font-size: 12px;
        }
        .hitl-param-name {
          color: #3b82f6;
          font-weight: 500;
        }
        .hitl-param-val {
          color: rgba(255, 255, 255, 0.8);
        }
        .hitl-param-remove {
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.5);
          cursor: pointer;
          padding: 0 4px;
        }
        .hitl-buttons {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
        }
        .hitl-btn {
          padding: 10px 20px;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .hitl-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .hitl-btn-confirm {
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          border: none;
          color: white;
        }
        .hitl-btn-confirm:hover:not(:disabled) {
          background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        }
        .hitl-btn-reject {
          background: rgba(244, 67, 54, 0.2);
          border: 1px solid #f44336;
          color: #ef4444;
        }
        .hitl-btn-reject:hover:not(:disabled) {
          background: rgba(244, 67, 54, 0.3);
        }
      `}</style>
    </div>
  );
};

export default HITLInteraction;
