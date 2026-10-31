import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Shield, BookOpen, LogOut, CheckCircle, PanelRightClose, PanelRightOpen } from 'lucide-react';

interface HeaderProps {
  selectedCount: number;
  totalDocs: number;
  isPdfViewerOpen: boolean;
  onTogglePdfViewer: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  selectedCount,
  totalDocs,
  isPdfViewerOpen,
  onTogglePdfViewer,
}) => {
  const { user, logout } = useAuth();

  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between flex-shrink-0 z-10 select-none">
      {/* Left: Branding */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-brand-500/10 border border-brand-500/30 text-brand-400">
            <Shield className="w-4 h-4" />
          </div>
          <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            DocuGround
            <span className="hidden sm:inline-block text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Strictly Grounded RAG
            </span>
          </h1>
        </div>

        {/* Selected documents badge */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/80 text-xs text-slate-300">
          <BookOpen className="w-3.5 h-3.5 text-brand-400" />
          <span>
            Scope: <strong className="text-white">{selectedCount}</strong> / {totalDocs} docs selected
          </span>
        </div>
      </div>

      {/* Right: Actions & User Info */}
      <div className="flex items-center gap-3 text-xs">
        <button
          onClick={onTogglePdfViewer}
          className={`px-2.5 py-1.5 rounded-lg border flex items-center gap-1.5 transition-colors ${
            isPdfViewerOpen
              ? 'bg-brand-500/15 border-brand-500/40 text-brand-300'
              : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white'
          }`}
          title="Toggle PDF Document Viewer"
        >
          {isPdfViewerOpen ? (
            <>
              <PanelRightClose className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Hide PDF</span>
            </>
          ) : (
            <>
              <PanelRightOpen className="w-3.5 h-3.5 text-brand-400" />
              <span className="hidden sm:inline">View PDF</span>
            </>
          )}
        </button>

        {user && (
          <div className="flex items-center gap-3 pl-2 border-l border-slate-800">
            <div className="hidden lg:flex flex-col text-right">
              <span className="font-medium text-slate-200">{user.email}</span>
              <span className="text-[10px] text-slate-400 flex items-center gap-1 justify-end">
                <CheckCircle className="w-2.5 h-2.5 text-emerald-400" /> Authenticated
              </span>
            </div>

            <button
              onClick={logout}
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
