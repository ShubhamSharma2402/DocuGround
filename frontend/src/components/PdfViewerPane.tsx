import React, { useState, useEffect } from 'react';
import { api, getToken } from '../api/client';
import {
  X,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Download,
  FileText,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { Document, Page, pdfjs } from 'react-pdf';

// Configure pdfjs worker to unpkg/cdnjs CDN for reliable client-side rendering
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PdfViewerPaneProps {
  docId: string | null;
  filename: string;
  initialPage?: number;
  onClose: () => void;
}

export const PdfViewerPane: React.FC<PdfViewerPaneProps> = ({
  docId,
  filename,
  initialPage = 1,
  onClose,
}) => {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(initialPage);
  const [scale, setScale] = useState<number>(1.0);
  const [pdfData, setPdfData] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialPage) {
      setPageNumber(initialPage);
    }
  }, [initialPage]);

  useEffect(() => {
    if (!docId) {
      setPdfData(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    const loadPdf = async () => {
      try {
        const token = getToken();
        const res = await fetch(api.documents.getContentUrl(docId), {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });

        if (!res.ok) {
          throw new Error(`Failed to load PDF (${res.status})`);
        }

        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        if (isMounted) {
          setPdfData(objectUrl);
          setLoading(false);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Error loading PDF document');
          setLoading(false);
        }
      }
    };

    loadPdf();

    return () => {
      isMounted = false;
      if (pdfData) URL.revokeObjectURL(pdfData);
    };
  }, [docId]);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setLoading(false);
    if (initialPage && initialPage <= numPages) {
      setPageNumber(initialPage);
    } else {
      setPageNumber(1);
    }
  };

  const changePage = (offset: number) => {
    setPageNumber((prev) => {
      const next = prev + offset;
      return Math.min(Math.max(1, next), numPages || 1);
    });
  };

  if (!docId) return null;

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 w-full overflow-hidden">
      {/* Top Toolbar */}
      <div className="h-14 px-4 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between gap-2 flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-brand-400 flex-shrink-0" />
          <span className="text-xs font-semibold text-white truncate max-w-[160px] md:max-w-[220px]" title={filename}>
            {filename}
          </span>
        </div>

        {/* Page navigation & zoom */}
        <div className="flex items-center gap-1.5 text-xs text-slate-300">
          <button
            onClick={() => changePage(-1)}
            disabled={pageNumber <= 1}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30"
            title="Previous Page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700">
            {pageNumber} / {numPages || '–'}
          </span>

          <button
            onClick={() => changePage(1)}
            disabled={numPages === null || pageNumber >= numPages}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30"
            title="Next Page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <div className="h-4 w-px bg-slate-800 mx-1" />

          <button
            onClick={() => setScale((s) => Math.max(0.6, s - 0.15))}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono text-[10px] text-slate-400 w-8 text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={() => setScale((s) => Math.min(2.0, s + 0.15))}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          <div className="h-4 w-px bg-slate-800 mx-1" />

          {pdfData && (
            <a
              href={pdfData}
              download={filename}
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
              title="Download PDF"
            >
              <Download className="w-4 h-4" />
            </a>
          )}

          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white ml-1"
            title="Close Viewer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Document Body */}
      <div className="flex-1 overflow-auto p-4 flex items-start justify-center bg-slate-950/70">
        {loading && (
          <div className="h-64 flex flex-col items-center justify-center gap-3 text-slate-400 text-xs">
            <Loader2 className="w-6 h-6 animate-spin text-brand-400" />
            <span>Loading document page {pageNumber}...</span>
          </div>
        )}

        {error && (
          <div className="h-64 flex flex-col items-center justify-center gap-3 text-rose-400 text-xs p-6 text-center">
            <AlertCircle className="w-8 h-8 text-rose-500" />
            <p>{error}</p>
          </div>
        )}

        {pdfData && (
          <div className="relative">
            <Document
              file={pdfData}
              onLoadSuccess={onDocumentLoadSuccess}
              loading=""
              error={
                <div className="p-4 text-xs text-rose-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  <span>Failed to render PDF canvas.</span>
                </div>
              }
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
          </div>
        )}
      </div>
    </div>
  );
};
