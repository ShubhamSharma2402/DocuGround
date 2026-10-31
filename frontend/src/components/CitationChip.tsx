import React, { useState } from 'react';
import { Citation } from '../types';
import { BookOpen, ExternalLink } from 'lucide-react';

interface CitationChipProps {
  index: number;
  citation: Citation;
  onClick: (docId: string, page: number) => void;
}

export const CitationChip: React.FC<CitationChipProps> = ({ index, citation, onClick }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        onClick={() => onClick(citation.doc_id, citation.page)}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-brand-500/10 text-brand-300 border border-brand-500/30 hover:bg-brand-500/20 hover:border-brand-500/50 transition-all cursor-pointer group"
      >
        <span className="font-mono font-bold text-brand-400">[{index + 1}]</span>
        <BookOpen className="w-3 h-3 text-brand-400" />
        <span className="truncate max-w-[120px]">{citation.doc_name}</span>
        <span className="text-slate-400 font-mono">p.{citation.page}</span>
        <ExternalLink className="w-2.5 h-2.5 opacity-60 group-hover:opacity-100" />
      </button>

      {/* Tooltip Preview */}
      {showTooltip && (
        <div className="absolute bottom-full left-0 mb-2 w-80 p-3 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl z-50 text-left pointer-events-none animate-in fade-in zoom-in-95 duration-100">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5 border-b border-slate-800 pb-1">
            <span className="font-semibold text-brand-300 truncate max-w-[180px]">{citation.doc_name}</span>
            <span className="font-mono">Page {citation.page} • Score: {citation.score.toFixed(3)}</span>
          </div>
          <p className="text-xs text-slate-200 line-clamp-4 italic leading-relaxed">
            "{citation.snippet}"
          </p>
          <div className="mt-2 text-[10px] text-brand-400 flex items-center gap-1">
            <span>Click to jump to page {citation.page} in document</span>
          </div>
        </div>
      )}
    </div>
  );
};
