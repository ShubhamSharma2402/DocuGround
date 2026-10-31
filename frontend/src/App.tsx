import React, { useState, useEffect, useRef } from 'react';
import { api } from './api/client';
import { useAuth } from './context/AuthContext';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { PdfViewerPane } from './components/PdfViewerPane';
import { DocumentUploadModal } from './components/DocumentUploadModal';
import { AuthModal } from './components/AuthModal';
import { Conversation, DocumentItem, Message, Citation } from './types';

export const AppContent: React.FC = () => {
  const { user, isLoading } = useAuth();

  // Documents state
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // Conversations & Chat state
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // PDF Viewer pane state
  const [isPdfViewerOpen, setIsPdfViewerOpen] = useState(false);
  const [activePdfDoc, setActivePdfDoc] = useState<{
    docId: string;
    filename: string;
    page: number;
  } | null>(null);

  // Initial data loading
  const loadDocuments = async () => {
    try {
      const docs = await api.documents.list();
      setDocuments(docs);

      // Auto-select ready documents if none selected yet
      setSelectedDocIds((prev) => {
        if (prev.length === 0) {
          return docs.filter((d) => d.status === 'ready').map((d) => d.id);
        }
        return prev.filter((id) => docs.some((d) => d.id === id && d.status === 'ready'));
      });
    } catch (err) {
      console.error('Failed to load documents', err);
    }
  };

  const loadConversations = async () => {
    try {
      const convs = await api.conversations.list();
      setConversations(convs);
    } catch (err) {
      console.error('Failed to load conversations', err);
    }
  };

  useEffect(() => {
    if (user) {
      loadDocuments();
      loadConversations();
    }
  }, [user]);

  // Handle conversation selection
  const handleSelectConv = async (convId: string) => {
    if (activeConvId === convId) return;
    try {
      setActiveConvId(convId);
      const detail = await api.conversations.get(convId);
      setMessages(detail.messages || []);
      // If conversation has associated documents, update selected documents
      if (detail.document_ids && detail.document_ids.length > 0) {
        setSelectedDocIds(detail.document_ids);
      }
    } catch (err) {
      console.error('Failed to load conversation details', err);
    }
  };

  const handleNewChat = () => {
    setActiveConvId(null);
    setMessages([]);
  };

  const handleDeleteConv = async (convId: string) => {
    try {
      await api.conversations.delete(convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConvId === convId) {
        handleNewChat();
      }
    } catch (err) {
      console.error('Failed to delete conversation', err);
    }
  };

  // Document actions
  const handleToggleSelectDoc = (id: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSelectAllDocs = () => {
    const readyIds = documents.filter((d) => d.status === 'ready').map((d) => d.id);
    setSelectedDocIds(readyIds);
  };

  const handleDeselectAllDocs = () => {
    setSelectedDocIds([]);
  };

  const handleDeleteDoc = async (docId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    try {
      await api.documents.delete(docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
      if (activePdfDoc?.docId === docId) {
        setActivePdfDoc(null);
        setIsPdfViewerOpen(false);
      }
    } catch (err) {
      console.error('Failed to delete document', err);
    }
  };

  const handleViewDoc = (docId: string, filename: string, page: number = 1) => {
    setActivePdfDoc({ docId, filename, page });
    setIsPdfViewerOpen(true);
  };

  // Citation click handler (jumps directly to page in PDF viewer)
  const handleSelectCitation = (docId: string, page: number) => {
    const doc = documents.find((d) => d.id === docId);
    const filename = doc ? doc.filename : 'Document';
    setActivePdfDoc({ docId, filename, page });
    setIsPdfViewerOpen(true);
  };

  // Chat message sending & SSE Streaming
  const handleSendMessage = async (question: string) => {
    if (selectedDocIds.length === 0 || isStreaming) return;

    // Optimistic user message
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      conversation_id: activeConvId || '',
      role: 'user',
      content: question,
      citations: [],
      retrieval_scores: [],
      gate_passed: null,
      verdict: null,
      latency_ms: null,
      created_at: new Date().toISOString(),
    };

    // Placeholder assistant message
    const assistantMsgId = `assistant-${Date.now()}`;
    const assistantMsg: Message = {
      id: assistantMsgId,
      conversation_id: activeConvId || '',
      role: 'assistant',
      content: '',
      citations: [],
      retrieval_scores: [],
      gate_passed: null,
      verdict: null,
      latency_ms: null,
      created_at: new Date().toISOString(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let streamedTokens = '';
    let currentCitations: Citation[] = [];
    let currentVerdict: any = null;

    try {
      await api.chat.streamChat(
        {
          document_ids: selectedDocIds,
          question,
          conversation_id: activeConvId || undefined,
        },
        {
          onCitations: (citations) => {
            currentCitations = citations;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, citations } : m
              )
            );
          },
          onToken: (token) => {
            streamedTokens += token;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, content: streamedTokens } : m
              )
            );
          },
          onVerification: (verdict, rejected) => {
            currentVerdict = verdict;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      verdict: verdict as any,
                      content: rejected ? "I couldn't find this in the document(s)." : m.content,
                    }
                  : m
              )
            );
          },
          onDone: (data) => {
            setIsStreaming(false);
            if (!activeConvId && data.conversation_id) {
              setActiveConvId(data.conversation_id);
              loadConversations();
            }

            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      id: data.message_id || m.id,
                      conversation_id: data.conversation_id,
                      gate_passed: data.gate_passed,
                      latency_ms: data.latency_ms,
                      isStreaming: false,
                    }
                  : m
              )
            );
          },
          onError: (err) => {
            setIsStreaming(false);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: `Error: ${err.message}`,
                      isStreaming: false,
                    }
                  : m
              )
            );
          },
        },
        controller.signal
      );
    } catch (err: any) {
      setIsStreaming(false);
    }
  };

  const handleCancelStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
      );
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen w-screen bg-slate-950 flex flex-col items-center justify-center gap-3 text-slate-400">
        <div className="w-8 h-8 rounded-full border-2 border-brand-500 border-t-transparent animate-spin" />
        <span className="text-xs font-mono">Loading DocuGround...</span>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-slate-950 text-slate-100">
      <AuthModal isOpen={!user} />

      {/* Top Application Header */}
      <Header
        selectedCount={selectedDocIds.length}
        totalDocs={documents.filter((d) => d.status === 'ready').length}
        isPdfViewerOpen={isPdfViewerOpen}
        onTogglePdfViewer={() => setIsPdfViewerOpen((prev) => !prev)}
      />

      {/* 3-Pane Responsive Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          documents={documents}
          selectedDocIds={selectedDocIds}
          onToggleSelectDoc={handleToggleSelectDoc}
          onSelectAllDocs={handleSelectAllDocs}
          onDeselectAllDocs={handleDeselectAllDocs}
          onOpenUpload={() => setIsUploadModalOpen(true)}
          onViewDoc={(id, name) => handleViewDoc(id, name, 1)}
          onDeleteDoc={handleDeleteDoc}
          conversations={conversations}
          activeConvId={activeConvId}
          onSelectConv={handleSelectConv}
          onNewChat={handleNewChat}
          onDeleteConv={handleDeleteConv}
        />

        {/* Center: Live Chat Thread */}
        <main className="flex-1 flex flex-col min-w-0">
          <ChatArea
            messages={messages}
            isStreaming={isStreaming}
            selectedDocCount={selectedDocIds.length}
            onSendMessage={handleSendMessage}
            onCancelStream={handleCancelStream}
            onSelectCitation={handleSelectCitation}
          />
        </main>

        {/* Right: PDF Document Viewer Pane */}
        {isPdfViewerOpen && (
          <aside className="w-[450px] lg:w-[540px] h-full flex-shrink-0 animate-in slide-in-from-right duration-200">
            <PdfViewerPane
              docId={activePdfDoc?.docId || null}
              filename={activePdfDoc?.filename || 'Document'}
              initialPage={activePdfDoc?.page || 1}
              onClose={() => setIsPdfViewerOpen(false)}
            />
          </aside>
        )}
      </div>

      {/* Upload Modal with Live SSE Ingestion Progress */}
      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onSuccess={loadDocuments}
      />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <div className="h-full w-full">
      <AppContent />
    </div>
  );
};

export default App;
