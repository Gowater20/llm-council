import React, { useState, useEffect } from 'react';
import { API_BASE } from '../api';
import './PromptManager.css';

export default function PromptManager({ onClose, onSelect }) {
  const [prompts, setPrompts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState('list'); // 'list' or 'create'
  
  const [formData, setFormData] = useState({
    id: '',
    name: '',
    description: '',
    system_prompt: '',
    chairman_instruction: ''
  });

  useEffect(() => {
    fetchPrompts();
  }, []);

  const fetchPrompts = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/prompts`);
      const data = await res.json();
      setPrompts(data);
    } catch (err) {
      console.error("Failed to fetch prompts", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name || !formData.system_prompt) return;

    const newId = formData.name.toLowerCase().replace(/[^a-z0-9]/g, '-');
    const newPrompt = { ...formData, id: newId };

    try {
      await fetch(`${API_BASE}/api/prompts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPrompt)
      });
      await fetchPrompts();
      setMode('list');
      setFormData({ id: '', name: '', description: '', system_prompt: '', chairman_instruction: '' });
    } catch (err) {
      console.error("Failed to create prompt", err);
    }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this persona?")) return;
    
    try {
      await fetch(`${API_BASE}/api/prompts/${id}`, { method: 'DELETE' });
      await fetchPrompts();
    } catch (err) {
      console.error("Failed to delete", err);
    }
  };

  return (
    <div className="prompt-manager-overlay">
      <div className="prompt-manager-content">
        <div className="pm-header">
          <h2>Council Personas</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        {mode === 'list' ? (
          <div className="pm-list-view">
            <div className="pm-list">
              {prompts.map(p => (
                <div key={p.id} className="prompt-card" onClick={() => onSelect(p)}>
                  <div className="pc-header">
                    <h3>{p.name}</h3>
                    {p.id !== 'default' && (
                      <button className="delete-btn" onClick={(e) => handleDelete(p.id, e)}>Trash</button>
                    )}
                  </div>
                  <p className="pc-desc">{p.description}</p>
                </div>
              ))}
            </div>
            <button className="create-btn" onClick={() => setMode('create')}>+ Create New Persona</button>
          </div>
        ) : (
          <form className="pm-form" onSubmit={handleCreate}>
            <input 
              placeholder="Name (e.g. Coding Expert)" 
              value={formData.name} 
              onChange={e => setFormData({...formData, name: e.target.value})}
              required
            />
            <input 
              placeholder="Short Description" 
              value={formData.description} 
              onChange={e => setFormData({...formData, description: e.target.value})}
            />
            <textarea 
              placeholder="System Prompt (Who are the council members?)" 
              value={formData.system_prompt} 
              onChange={e => setFormData({...formData, system_prompt: e.target.value})}
              rows={4}
              required
            />
            <textarea 
              placeholder="Chairman Instructions (How should they synthesize?)" 
              value={formData.chairman_instruction} 
              onChange={e => setFormData({...formData, chairman_instruction: e.target.value})}
              rows={3}
            />
            <div className="form-actions">
              <button type="button" onClick={() => setMode('list')}>Cancel</button>
              <button type="submit" className="save-btn">Save Persona</button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
