import ReactMarkdown from 'react-markdown';
import './Stage3.css';

export default function Stage3({ finalResponse }) {
  if (!finalResponse) {
    return null;
  }

  return (
    <div className="stage stage3">
      <h3 className="stage-title">Stage 3: Final Council Answer</h3>
      <div className="final-response">
        <div className="chairman-label">
          Chairman: {finalResponse.model.split('/')[1] || finalResponse.model}
          {finalResponse.cost !== undefined && (
            <span className="chairman-cost"> | Cost: ${finalResponse.cost.toFixed(5)}</span>
          )}
          {finalResponse.usage && (
            <span className="usage-info">
              ({finalResponse.usage.prompt_tokens}p + {finalResponse.usage.completion_tokens}c tokens)
            </span>
          )}
        </div>
        <div className="final-text markdown-content">
          <ReactMarkdown>{finalResponse.response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
