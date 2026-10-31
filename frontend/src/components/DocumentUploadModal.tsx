import React, { useState, useRef } from 'react';
import { api } from '../api/client';
import { JobStatus } from '../types';
import { Upload, X, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

interface DocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued in processing stream...',
  parse: 'Stage 1: Extracting text & running OCR...',
  chunk: 'Stage 2: Chunking at sentence boundaries...',
  embed: 'Stage 3: Generating dense & sparse embeddings...',
  index: 'Stage 4: Upserting vectors into Qdrant...',
  finalize: 'Stage 5: Finalizing document records...',
  done: 'Ready for chatting!',
  failed: 'Processing failed',
};

export const DocumentUploadModal: React.FC<DocumentUploadModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const validateFile = (file: File): boolean => {
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setError('Please upload a valid PDF document (.pdf)');
      return false;
    }
    const maxMb = 20;
    if (file.size > maxMb * 1024 * 1024) {
      setError(`File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds limit of ${maxMb}MB`);
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);

    try {
      const res = await api.documents.upload(selectedFile);
      setJobStatus({
        job_id: res.job_id,
        doc_id: res.document_id,
        user_id: '',
        stage: 'queued',
        progress: 0,
        attempts: 0,
        error: '',
      });

      // Listen to SSE job status changes until done or failed
      const unsubscribe = api.jobs.listenEvents(
        res.job_id,
        (status) => {
          setJobStatus(status);
          if (status.stage === 'done') {
            setIsUploading(false);
            onSuccess();
            setTimeout(() => {
              onClose();
              setSelectedFile(null);
              setJobStatus(null);
            }, 1200);
          } else if (status.stage === 'failed') {
            setIsUploading(false);
            setError(status.error || 'Ingestion failed');
          }
        },
        (err) => {
          console.error('Job SSE error', err);
        }
      );

      return () => unsubscribe();
    } catch (err: any) {
      setIsUploading(false);
      setError(err.message || 'Upload failed');
    }
  };

  const currentStageText = jobStatus
    ? STAGE_LABELS[jobStatus.stage] || `Processing stage: ${jobStatus.stage}`
    : 'Uploading...';

  const progressPercent = jobStatus ? jobStatus.progress : (isUploading ? 10 : 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-2xl relative">
        <button
          onClick={onClose}
          disabled={isUploading && jobStatus?.stage !== 'done' && jobStatus?.stage !== 'failed'}
          className="absolute right-4 top-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors disabled:opacity-30"
        >
          <X className="w-5 h-5" />
        </button>

        <h3 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
          <Upload className="w-5 h-5 text-brand-400" />
          Upload Document
        </h3>
        <p className="text-xs text-slate-400 mb-5">
          PDFs are strictly chunked, embedded, and indexed for isolated multi-tenant retrieval.
        </p>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Drag and Drop Zone */}
        {!jobStatus && (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              dragActive
                ? 'border-brand-500 bg-brand-500/10'
                : 'border-slate-700 hover:border-slate-600 bg-slate-800/40'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-12 h-12 mx-auto rounded-full bg-slate-800 flex items-center justify-center text-slate-400 mb-3 border border-slate-700">
              <FileText className="w-6 h-6 text-brand-400" />
            </div>
            {selectedFile ? (
              <div>
                <p className="text-sm font-semibold text-white truncate max-w-xs mx-auto">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
                <span className="inline-block mt-3 px-3 py-1 bg-brand-500/20 text-brand-300 text-xs rounded-full font-medium">
                  Click or drag another file to replace
                </span>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium text-slate-200">
                  Drag and drop your PDF here, or <span className="text-brand-400 font-semibold">browse</span>
                </p>
                <p className="text-xs text-slate-500 mt-1.5">Max size 20MB • Up to 400 pages • Auto OCR</p>
              </div>
            )}
          </div>
        )}

        {/* Live SSE Ingestion Progress */}
        {jobStatus && (
          <div className="p-5 rounded-xl bg-slate-800/60 border border-slate-700/80 mb-2">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-medium text-slate-300 flex items-center gap-2">
                {jobStatus.stage === 'done' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : jobStatus.stage === 'failed' ? (
                  <AlertCircle className="w-4 h-4 text-rose-400" />
                ) : (
                  <Loader2 className="w-4 h-4 text-brand-400 animate-spin" />
                )}
                {currentStageText}
              </span>
              <span className="font-mono text-slate-400">{progressPercent}%</span>
            </div>

            <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  jobStatus.stage === 'failed'
                    ? 'bg-rose-500'
                    : jobStatus.stage === 'done'
                    ? 'bg-emerald-500'
                    : 'bg-brand-500'
                }`}
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400">
              <span className="truncate max-w-[250px]">{selectedFile?.name}</span>
              <span className="font-mono">Stage: {jobStatus.stage}</span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-5 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            disabled={isUploading && jobStatus?.stage !== 'done' && jobStatus?.stage !== 'failed'}
            className="px-4 py-2 text-xs font-medium text-slate-300 hover:text-white rounded-lg border border-slate-700 hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          {!jobStatus && (
            <button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className="px-5 py-2 text-xs font-semibold text-white bg-brand-600 hover:bg-brand-500 rounded-lg shadow-lg shadow-brand-600/30 transition-all disabled:opacity-40 flex items-center gap-1.5"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload & Ingest</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
