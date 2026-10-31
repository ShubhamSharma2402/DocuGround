export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export type DocumentStatus = 'processing' | 'ready' | 'failed';

export interface DocumentItem {
  id: string;
  user_id: string;
  filename: string;
  size_bytes: number;
  status: DocumentStatus;
  page_count: number | null;
  chunk_count: number | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface UploadResponse {
  job_id: string;
  document_id: string;
  filename: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  doc_id: string;
  user_id: string;
  stage: string;
  progress: number;
  attempts: number;
  error: string;
}

export interface Citation {
  chunk_id: string;
  doc_id: string;
  doc_name: string;
  page: number;
  snippet: string;
  score: number;
}

export type MessageRole = 'user' | 'assistant';
export type VerificationVerdict = 'supported' | 'partially' | 'unsupported';

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  citations: Citation[];
  retrieval_scores: number[];
  gate_passed: boolean | null;
  verdict: VerificationVerdict | null;
  latency_ms: number | null;
  created_at: string;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  user_id: string;
  document_ids: string[];
  title: string;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}
