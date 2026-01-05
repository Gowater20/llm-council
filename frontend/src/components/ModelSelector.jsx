import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api';
import './ModelSelector.css';

export default function ModelSelector({
  selectedCouncil,
  setSelectedCouncil,
  selectedChairman,
  setSelectedChairman,
  onClose
}) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [filterFree, setFilterFree] = useState(false);

  useEffect(() => {
    async function fetchModels() {
      try {
        setLoading(true);
        const data = await api.listModels();
        // Sort: Free first, then by name
        const sorted = data.sort((a, b) => {
          const isFreeA = parseFloat(a.pricing?.prompt || 0) === 0;
          const isFreeB = parseFloat(b.pricing?.prompt || 0) === 0;
          if (isFreeA && !isFreeB) return -1;
          if (!isFreeA && isFreeB) return 1;
          return (a.name || '').localeCompare(b.name || '');
        });
        setModels(sorted);
        setLoading(false);
      } catch (err) {
        setError('Failed to load models');
        setLoading(false);
      }
    }
    fetchModels();
  }, []);

  const filteredModels = useMemo(() => {
    return models.filter(m => {
      const matchSearch = (m.name || '').toLowerCase().includes(search.toLowerCase()) || 
                          (m.id || '').toLowerCase().includes(search.toLowerCase());
      const isFree = parseFloat(m.pricing?.prompt || 0) === 0;
      return matchSearch && (!filterFree || isFree);
    });
  }, [models, search, filterFree]);

  const toggleCouncil = (modelId) => {
    if (selectedCouncil.includes(modelId)) {
      setSelectedCouncil(selectedCouncil.filter(id => id !== modelId));
    } else {
      setSelectedCouncil([...selectedCouncil, modelId]);
    }
  };

  const formatPrice = (price) => {
    const p = parseFloat(price || 0);
    if (p === 0) return 'Free';
    return `$${(p * 1).toFixed(4)} / 1M`;
  };

  if (loading) return <div className="model-selector-modal loading">Loading models...</div>;
  if (error) return <div className="model-selector-modal error">{error}</div>;

  return (
    <div className="model-selector-overlay" onClick={onClose}>
      <div className="model-selector-modal" onClick={e => e.stopPropagation()}>
        <div className="model-selector-header">
          <h3>Configure LLM Council</h3>
          <button className="close-button" onClick={onClose}>&times;</button>
        </div>

        <div className="model-selector-controls">
          <input 
            type="text" 
            placeholder="Search models..." 
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="model-search"
          />
          <label className="filter-free">
            <input 
              type="checkbox" 
              checked={filterFree}
              onChange={e => setFilterFree(e.target.checked)}
            />
            Show Free Only
          </label>
        </div>

        <div className="selection-summary">
          <div className="summary-item">
            <strong>Council:</strong> {selectedCouncil.length} models
          </div>
          <div className="summary-item">
            <strong>Chairman:</strong> {selectedChairman || 'Not selected'}
          </div>
        </div>

        <div className="models-list-container">
          <table className="models-table">
            <thead>
              <tr>
                <th>Council</th>
                <th>Chairman</th>
                <th>Model Name</th>
                <th>Pricing (Prompt/Comp)</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map(model => (
                <tr key={model.id} className={selectedCouncil.includes(model.id) ? 'selected-row' : ''}>
                  <td>
                    <input 
                      type="checkbox" 
                      checked={selectedCouncil.includes(model.id)}
                      onChange={() => toggleCouncil(model.id)}
                    />
                  </td>
                  <td>
                    <input 
                      type="radio" 
                      name="chairman"
                      checked={selectedChairman === model.id}
                      onChange={() => setSelectedChairman(model.id)}
                    />
                  </td>
                  <td>
                    <div className="model-info">
                      <span className="model-name">{model.name}</span>
                      <span className="model-id">{model.id}</span>
                    </div>
                  </td>
                  <td className="model-pricing">
                    <span className={`price-badge ${parseFloat(model.pricing?.prompt) === 0 ? 'free' : 'paid'}`}>
                      {formatPrice(model.pricing?.prompt)} / {formatPrice(model.pricing?.completion)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="model-selector-footer">
          <button className="confirm-button" onClick={onClose}>Apply Configuration</button>
        </div>
      </div>
    </div>
  );
}
