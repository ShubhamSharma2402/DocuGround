import React, { useState, useRef, useEffect } from 'react';
import { Message, Citation } from '../types';
import { CitationChip } from './CitationChip';
import {
  Send,
  Sparkles,
  User,
  ShieldCheck,
  ShieldAlert,
  Clock,
  AlertTriangle,
  StopCircle,
  FileQuestion,
  Info,
} from 'lucide-react';

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
  selectedDocCount: number;
  onSendMessage: (question: string) => void;
  onCancelStream: () => void;
  onSelectCitation: (docId: string, page: number) => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isStreaming,
  selectedDocCount,
  onSendMessage,
  onCancelStream,
  onSelectCitation,
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming || selectedDocCount === 0) return;
    onSendMessage(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 overflow-hidden relative">
      {/* Messages Thread */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto my-auto p-6">
            <div className="w-14 h-14 rounded-2xl bg-brand-500/10 border border-brand-500/20 text-brand-400 flex items-center justify-center mb-4 shadow-inner">
              <Sparkles className="w-7 h-7" />
            </div>
            <h2 className="text-lg font-bold text-white mb-2">Strict Grounded RAG Assistant</h2>
            <p className="text-xs text-slate-400 leading-relaxed mb-6">
              Ask anything about your uploaded documents. DocuGround answers <strong>strictly</strong> from retrieved context with page citations, and explicitly refuses outside queries.
            </p>

            <div className="grid grid-cols-1 gap-2 w-full text-left text-xs">
              <div
                onClick={() => setInput("What was the consolidated revenue and year-over-year growth?")}
                className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white cursor-pointer transition-colors"
              >
                📊 "What was the consolidated revenue and year-over-year growth?"
              </div>
              <div
                onClick={() => setInput("Summarize the primary security frameworks and compliance certifications.")}
                className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white cursor-pointer transition-colors"
              >
                🔒 "Summarize the primary security frameworks and compliance certifications."
              </div>
              <div
                onClick={() => setInput("Who is on the executive leadership team?")}
                className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white cursor-pointer transition-colors"
              >
                👥 "Who is on the executive leadership team?"
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            const isRefusal = msg.content.includes("I couldn't find this in the document(s)") || msg.gate_passed === false;

            return (
              <div
                key={msg.id || idx}
                className={`flex gap-3 text-sm ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-7 h-7 rounded-lg bg-brand-600/20 border border-brand-500/40 text-brand-400 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Sparkles className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] md:max-w-2xl rounded-2xl px-4 py-3 leading-relaxed ${
                    isUser
                      ? 'bg-brand-600 text-white rounded-tr-sm shadow-md'
                      : isRefusal
                      ? 'bg-amber-950/20 border border-amber-500/30 text-amber-200 rounded-tl-sm'
                      : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-sm shadow-md'
                  }`}
                >
                  {/* Gate Refusal Alert Banner */}
                  {!isUser && isRefusal && (
                    <div className="mb-2 flex items-center gap-1.5 text-xs text-amber-400 font-semibold border-b border-amber-500/20 pb-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Refusal Gate: Zero outside knowledge allowed</span>
                    </div>
                  )}

                  {/* Message Content */}
                  <div className="whitespace-pre-wrap text-[13.5px] leading-relaxed">
                    {msg.content}
                    {msg.isStreaming && (
                      <span className="inline-block w-2 h-4 bg-brand-400 ml-1 animate-pulse align-middle" />
                    )}
                  </div>

                  {/* Citations Row: show only sources actually cited in the text when available */}
                  {!isUser && msg.citations && msg.citations.length > 0 && (() => {
                    const matches = Array.from(msg.content.matchAll(/\[(\d+)\]/g));
                    const citedIndexes = new Set(matches.map(m => parseInt(m[1], 10) - 1));

                    // When streaming completes or references exist, show only the cited sources
                    const isCompleted = !msg.isStreaming;
                    const showOnlyCited = citedIndexes.size > 0 && (isCompleted || citedIndexes.size >= 1);
                    const activeCitations = showOnlyCited
                      ? msg.citations.filter((_, idx) => citedIndexes.has(idx))
                      : msg.citations;

                    if (activeCitations.length === 0) return null;

                    return (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/80">
                        <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                          <span>
                            {showOnlyCited
                              ? `Cited Sources (${activeCitations.length})`
                              : `Candidate Sources (${activeCitations.length})`}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {activeCitations.map((c) => {
                            const originalIdx = msg.citations.indexOf(c);
                            return (
                              <CitationChip
                                key={c.chunk_id || originalIdx}
                                index={originalIdx}
                                citation={c}
                                onClick={onSelectCitation}
                              />
                            );
                          })}
                        </div>
                      </div>
                    );
                  })()}

                  {/* Metadata Footer: Verdict & Latency */}
                  {!isUser && (msg.verdict || msg.latency_ms !== null) && (
                    <div className="mt-2.5 flex items-center gap-3 text-[10px] text-slate-500 border-t border-slate-800/50 pt-1.5">
                      {msg.verdict && (
                        <span
                          className={`flex items-center gap-1 font-medium ${
                            msg.verdict === 'supported'
                              ? 'text-emerald-400'
                              : msg.verdict === 'partially'
                              ? 'text-amber-400'
                              : 'text-rose-400'
                          }`}
                        >
                          <ShieldCheck className="w-3 h-3" />
                          <span>Verification: {msg.verdict}</span>
                        </span>
                      )}

                      {msg.latency_ms !== null && (
                        <span className="flex items-center gap-1 font-mono text-slate-500">
                          <Clock className="w-3 h-3" />
                          <span>{msg.latency_ms.toFixed(0)} ms</span>
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Prompt Box */}
      <div className="p-4 bg-slate-900 border-t border-slate-800">
        {selectedDocCount === 0 && (
          <div className="mb-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2">
            <Info className="w-3.5 h-3.5 flex-shrink-0" />
            <span>Select at least one document from the sidebar to query against.</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="relative flex items-end gap-2 max-w-4xl mx-auto">
          <textarea
            ref={textareaRef}
            rows={2}
            value={input}
            disabled={isStreaming}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedDocCount > 0
                ? "Ask a question about the selected document(s)... (Enter to send, Shift+Enter for newline)"
                : "Select document(s) in the sidebar first"
            }
            className="w-full resize-none p-3 pr-24 text-sm bg-slate-950/80 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:opacity-50"
          />

          <div className="absolute right-3 bottom-3 flex items-center gap-1.5">
            {isStreaming ? (
              <button
                type="button"
                onClick={onCancelStream}
                className="p-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition-colors flex items-center gap-1 text-xs font-medium shadow-md shadow-rose-600/20"
                title="Cancel Generation"
              >
                <StopCircle className="w-4 h-4" />
                <span className="hidden sm:inline">Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || selectedDocCount === 0}
                className="p-2 rounded-lg bg-brand-600 hover:bg-brand-500 text-white transition-colors disabled:opacity-30 disabled:hover:bg-brand-600 shadow-md shadow-brand-600/20"
                title="Send Question"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};
