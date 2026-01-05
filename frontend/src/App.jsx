import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import ModelSelector from './components/ModelSelector';
import { api } from './api';
import PromptManager from './components/PromptManager';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // UI State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [showPrompts, setShowPrompts] = useState(false);

  // Council State
  const [selectedPrompt, setSelectedPrompt] = useState({ id: 'default', name: 'Default Council' });
  
  // Initialize from localStorage or defaults
  const [councilConfig, setCouncilConfig] = useState(() => {
    const savedCouncil = localStorage.getItem('llm_council_selection');
    const savedChairman = localStorage.getItem('llm_chairman_selection');
    return {
      council: savedCouncil ? JSON.parse(savedCouncil) : [
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free"
      ],
      chairman: savedChairman ? JSON.parse(savedChairman) : "meta-llama/llama-3.3-70b-instruct:free"
    };
  });

  // Save selection to localStorage
  useEffect(() => {
    localStorage.setItem('llm_council_selection', JSON.stringify(councilConfig.council));
    localStorage.setItem('llm_chairman_selection', JSON.stringify(councilConfig.chairman));
  }, [councilConfig]);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setActiveConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      // Assuming new conversation has empty messages initially
      setActiveConversation({ ...newConv, messages: [] });
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    loadConversation(id);
  };

  const handleDeleteConversation = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this conversation?')) return;
    
    try {
      await api.deleteConversation(id);
      setConversations(conversations.filter(c => c.id !== id));
      if (activeConversation?.id === id) {
        setActiveConversation(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const handleSendMessage = async (content) => {
    if (!activeConversation) return;

    setIsLoading(true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setActiveConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage3: false,
        },
      };

      // Add the partial assistant message
      setActiveConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(
        activeConversation.id, 
        content, 
        (eventType, event) => {
          switch (eventType) {
            case 'stage1_start':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.loading.stage1 = true;
                return { ...prev, messages };
              });
              break;

            case 'stage1_complete':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.stage1 = event.data;
                lastMsg.loading.stage1 = false;
                return { ...prev, messages };
              });
              break;

            case 'stage2_start':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.loading.stage2 = true;
                return { ...prev, messages };
              });
              break;

            case 'stage2_complete':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.stage2 = event.data;
                lastMsg.metadata = event.metadata;
                lastMsg.loading.stage2 = false;
                return { ...prev, messages };
              });
              break;

            case 'stage3_start':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.loading.stage3 = true;
                return { ...prev, messages };
              });
              break;

            case 'stage3_complete':
              setActiveConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                lastMsg.stage3 = event.data;
                lastMsg.loading.stage3 = false;
                // Add total cost to metadata if present
                if (event.metadata?.total_cost !== undefined) {
                  lastMsg.metadata = { 
                    ...lastMsg.metadata, 
                    total_cost: event.metadata.total_cost 
                  };
                }
                return { ...prev, messages };
              });
              break;

          case 'title_complete':
              // Reload conversations to get updated title
              loadConversations();
              break;

            case 'complete':
              // Stream complete, reload conversations list
              loadConversations();
              setIsLoading(false);
              break;

            case 'error':
              console.error('Stream error:', event.message);
              setIsLoading(false);
              break;

            default:
              // Handled by updateConversationState
              break;
          }
        }, 
        councilConfig.council, 
        councilConfig.chairman,
        selectedPrompt.id // Pass the selected prompt ID
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setActiveConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={activeConversation?.id}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <ChatInterface
        conversation={activeConversation} // Renamed from currentConversation
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        onOpenConfig={() => setShowConfig(true)} // Renamed from setIsConfigOpen
        councilConfig={councilConfig} // Now passes the combined config object
        selectedPrompt={selectedPrompt} // Pass selected prompt
        onOpenPrompts={() => setShowPrompts(true)} // New prop to open prompt manager
      />
      
      {showConfig && ( // Renamed from isConfigOpen
        <ModelSelector // This component will likely be renamed to CouncilConfig
          selectedCouncil={councilConfig.council}
          setSelectedCouncil={(newCouncil) => setCouncilConfig(prev => ({ ...prev, council: newCouncil }))}
          selectedChairman={councilConfig.chairman}
          setSelectedChairman={(newChairman) => setCouncilConfig(prev => ({ ...prev, chairman: newChairman }))}
          onClose={() => setShowConfig(false)}
        />
      )}

      {showPrompts && ( // Conditionally render PromptManager
        <PromptManager
          onSelect={(prompt) => {
            setSelectedPrompt(prompt);
            setShowPrompts(false);
          }}
          onClose={() => setShowPrompts(false)}
        />
      )}
    </div>
  );
}

export default App;
