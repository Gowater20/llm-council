import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''} status-${resp.status || 'unknown'}`}
            onClick={() => setActiveTab(index)}
          >
            <div className="tab-main">
              {resp.model.split('/')[1] || resp.model}
              {resp.status === 'error' && <span className="error-dot">!</span>}
            </div>
            {resp.cost !== undefined && resp.status === 'success' && (
              <span className="tab-cost">${resp.cost.toFixed(5)}</span>
            )}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-header">
          <div className="model-name">
            {responses[activeTab].model}
            {responses[activeTab].usage && responses[activeTab].status === 'success' && (
              <span className="usage-info">
                ({responses[activeTab].usage.prompt_tokens}p + {responses[activeTab].usage.completion_tokens}c tokens)
              </span>
            )}
          </div>
          {responses[activeTab].status === 'error' && (
            <div className="model-error-badge">
              {responses[activeTab].error || 'Response Failed'}
            </div>
          )}
        </div>
        <div className={`response-text markdown-content ${responses[activeTab].status === 'error' ? 'text-error' : ''}`}>
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
