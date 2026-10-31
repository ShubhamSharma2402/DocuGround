import React from 'react';
import { Conversation, DocumentItem } from '../types';
import {
  Upload,
  Plus,
  FileText,
  Trash2,
  Eye,
  CheckSquare,
  Square,
  MessageSquare,
  Clock,
  Layers,
  AlertCircle,
  Loader2,
} from 'lucide-react';

interface SidebarProps {
  documents: DocumentItem[];
  selectedDocIds: string[];
  onToggleSelectDoc: (id: string) => void;
  onSelectAllDocs: () => void;
  onDeselectAllDocs: () => void;
  onOpenUpload: () => void;
  onViewDoc: (docId: string, filename: string) => void;
  onDeleteDoc: (docId: string) => void;

  conversations: Conversation[];
  activeConvId: string | null;
  onSelectConv: (id: string) => void;
  onNewChat: () => void;
  onDeleteConv: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  documents,
  selectedDocIds,
  onToggleSelectDoc,
  onSelectAllDocs,
  onDeselectAllDocs,
  onOpenUpload,
  onViewDoc,
  onDeleteDoc,
  conversations,
  activeConvId,
  onSelectConv,
  onNewChat,
  onDeleteConv,
}) => {
  const readyDocs = documents.filter((d) => d.status === 'ready');
  const allSelected = readyDocs.length > 0 && readyDocs.every((d) => selectedDocIds.includes(d.id));

  return (
    <aside className="w-80 h-full bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0 select-none">
      {/* Action Top Bar */}
      <div className="p-3.5 border-b border-slate-800 flex items-center gap-2">
        <button
          onClick={onNewChat}
          className="flex-1 py-2 px-3 bg-brand-600 hover:bg-brand-500 text-white rounded-lg text-xs font-semibold shadow-md shadow-brand-600/20 transition-all flex items-center justify-center gap-1.5"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New Chat</span>
        </button>

        <button
          onClick={onOpenUpload}
          className="py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5"
          title="Upload PDF Document"
        >
          <Upload className="w-3.5 h-3.5 text-brand-400" />
          <span>Upload</span>
        </button>
      </div>

      {/* Split view: Documents section (top) + Conversations history (bottom) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Documents Section */}
        <div className="flex-1 flex flex-col overflow-hidden border-b border-slate-800/80">
          <div className="px-3.5 py-2.5 bg-slate-900/90 flex items-center justify-between text-xs font-semibold text-slate-400">
            <span className="flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
              <Layers className="w-3.5 h-3.5 text-brand-400" />
              Documents ({selectedDocIds.length}/{readyDocs.length} Active)
            </span>

            {readyDocs.length > 0 && (
              <button
                onClick={allSelected ? onDeselectAllDocs : onSelectAllDocs}
                className="text-[11px] text-brand-400 hover:text-brand-300 font-medium"
              >
                {allSelected ? 'Clear All' : 'Select All'}
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {documents.length === 0 ? (
              <div className="py-8 px-4 text-center">
                <div className="w-10 h-10 rounded-full bg-slate-800 text-slate-500 mx-auto mb-2 flex items-center justify-center">
                  <FileText className="w-5 h-5" />
                </div>
                <p className="text-xs text-slate-400 font-medium">No documents uploaded yet</p>
                <p className="text-[11px] text-slate-500 mt-1">Upload a PDF to ground the chatbot</p>
                <button
                  onClick={onOpenUpload}
                  className="mt-3 text-xs text-brand-400 hover:underline inline-flex items-center gap-1"
                >
                  <Plus className="w-3 h-3" />
                  Upload your first PDF
                </button>
              </div>
            ) : (
              documents.map((doc) => {
                const isSelected = selectedDocIds.includes(doc.id);
                const isReady = doc.status === 'ready';

                return (
                  <div
                    key={doc.id}
                    className={`p-2.5 rounded-xl border transition-all text-xs ${
                      isSelected
                        ? 'bg-brand-500/10 border-brand-500/40'
                        : 'bg-slate-800/40 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2 min-w-0 flex-1">
                        <button
                          type="button"
                          disabled={!isReady}
                          onClick={() => onToggleSelectDoc(doc.id)}
                          className={`mt-0.5 flex-shrink-0 ${!isReady ? 'opacity-30 cursor-not-allowed' : 'text-slate-400 hover:text-brand-400'}`}
                        >
                          {isSelected ? (
                            <CheckSquare className="w-4 h-4 text-brand-400" />
                          ) : (
                            <Square className="w-4 h-4" />
                          )}
                        </button>

                        <div className="min-w-0 flex-1">
                          <p
                            className="font-medium text-slate-200 truncate cursor-pointer hover:text-brand-300"
                            title={doc.filename}
                            onClick={() => onViewDoc(doc.id, doc.filename)}
                          >
                            {doc.filename}
                          </p>

                          <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-1">
                            {doc.status === 'ready' && (
                              <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                                {doc.page_count}p • {doc.chunk_count} chunks
                              </span>
                            )}
                            {doc.status === 'processing' && (
                              <span className="inline-flex items-center gap-1 text-amber-400 font-medium animate-pulse">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Processing...
                              </span>
                            )}
                            {doc.status === 'failed' && (
                              <span className="inline-flex items-center gap-1 text-rose-400 font-medium">
                                <AlertCircle className="w-3 h-3" />
                                Failed
                              </span>
                            )}
                            <span>•</span>
                            <span>{(doc.size_bytes / (1024 * 1024)).toFixed(1)}MB</span>
                          </div>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1">
                        {isReady && (
                          <button
                            type="button"
                            onClick={() => onViewDoc(doc.id, doc.filename)}
                            className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
                            title="View PDF"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => onDeleteDoc(doc.id)}
                          className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Conversations History Section */}
        <div className="h-48 flex flex-col bg-slate-900/50">
          <div className="px-3.5 py-2 flex items-center justify-between text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            <span className="flex items-center gap-1.5">
              <MessageSquare className="w-3.5 h-3.5 text-slate-400" />
              Recent Conversations
            </span>
            <span className="font-mono text-slate-500">{conversations.length}</span>
          </div>

          <div className="flex-1 overflow-y-auto px-2 space-y-1">
            {conversations.length === 0 ? (
              <p className="text-[11px] text-slate-500 text-center py-4 italic">No conversation history</p>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => onSelectConv(conv.id)}
                  className={`group px-2.5 py-1.5 rounded-lg flex items-center justify-between text-xs cursor-pointer transition-colors ${
                    activeConvId === conv.id
                      ? 'bg-slate-800 text-white font-medium'
                      : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                  }`}
                >
                  <span className="truncate max-w-[200px]">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConv(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-500 hover:text-rose-400 transition-opacity"
                    title="Delete Chat"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </aside>
  );
};
