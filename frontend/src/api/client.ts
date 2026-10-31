import {
  AuthResponse,
  Conversation,
  ConversationDetail,
  DocumentItem,
  JobStatus,
  UploadResponse,
  Citation,
} from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || '';

export function getToken(): string | null {
  return localStorage.getItem('docuground_token');
}

export function setToken(token: string): void {
  localStorage.setItem('docuground_token', token);
}

export function clearToken(): void {
  localStorage.removeItem('docuground_token');
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers || {});

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}`;
    try {
      const errData = await response.json();
      errorMsg = errData.message || errData.detail || errData.error || errorMsg;
    } catch {
      // Non-json response
    }
    if (response.status === 401) {
      clearToken();
      window.dispatchEvent(new Event('auth_unauthorized'));
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export const api = {
  // ── Auth ───────────────────────────────────────────────────────────────────
  auth: {
    async register(email: string, password: string): Promise<AuthResponse> {
      const res = await request<AuthResponse>('/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });
      setToken(res.access_token);
      return res;
    },

    async login(email: string, password: string): Promise<AuthResponse> {
      const res = await request<AuthResponse>('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });
      setToken(res.access_token);
      return res;
    },

    async me() {
      return request<{ id: string; email: string; created_at: string }>('/auth/me');
    },

    logout() {
      clearToken();
    },
  },

  // ── Documents ──────────────────────────────────────────────────────────────
  documents: {
    async list(): Promise<DocumentItem[]> {
      return request<DocumentItem[]>('/documents');
    },

    async upload(file: File): Promise<UploadResponse> {
      const formData = new FormData();
      formData.append('file', file);
      return request<UploadResponse>('/documents', {
        method: 'POST',
        body: formData,
      });
    },

    async delete(docId: string): Promise<void> {
      await request<void>(`/documents/${docId}`, {
        method: 'DELETE',
      });
    },

    getContentUrl(docId: string): string {
      return `${API_BASE}/documents/${docId}/content`;
    },
  },

  // ── Jobs ───────────────────────────────────────────────────────────────────
  jobs: {
    async getStatus(jobId: string): Promise<JobStatus> {
      return request<JobStatus>(`/jobs/${jobId}`);
    },

    listenEvents(jobId: string, onEvent: (data: JobStatus) => void, onError: (err: any) => void): () => void {
      const token = getToken();
      // Use EventSource or fetch reader with auth
      const controller = new AbortController();

      const fetchStream = async () => {
        try {
          const res = await fetch(`${API_BASE}/jobs/${jobId}/events`, {
            headers: { Authorization: `Bearer ${token}` },
            signal: controller.signal,
          });

          if (!res.ok) throw new Error(`Events stream failed with status ${res.status}`);
          const reader = res.body?.getReader();
          if (!reader) return;

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const chunk of lines) {
              const dataLine = chunk.split('\n').find((l) => l.startsWith('data: '));
              if (dataLine) {
                try {
                  const parsed = JSON.parse(dataLine.replace('data: ', ''));
                  onEvent(parsed);
                } catch {
                  // Ignore parse error
                }
              }
            }
          }
        } catch (err: any) {
          if (err.name !== 'AbortError') {
            onError(err);
          }
        }
      };

      fetchStream();
      return () => controller.abort();
    },
  },

  // ── Conversations & Chat ───────────────────────────────────────────────────
  conversations: {
    async list(): Promise<Conversation[]> {
      return request<Conversation[]>('/conversations');
    },

    async get(convId: string): Promise<ConversationDetail> {
      return request<ConversationDetail>(`/conversations/${convId}`);
    },

    async delete(convId: string): Promise<void> {
      await request<void>(`/conversations/${convId}`, {
        method: 'DELETE',
      });
    },
  },

  chat: {
    async streamChat(
      params: {
        document_ids: string[];
        question: string;
        conversation_id?: string;
      },
      handlers: {
        onCitations: (citations: Citation[]) => void;
        onToken: (token: string) => void;
        onVerification?: (verdict: string, rejected: boolean) => void;
        onDone: (data: { conversation_id: string; message_id: string; latency_ms: number; gate_passed: boolean }) => void;
        onError: (err: Error) => void;
      },
      signal?: AbortSignal
    ): Promise<void> {
      const token = getToken();

      try {
        const response = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(params),
          signal,
        });

        if (!response.ok) {
          let errText = `HTTP Error ${response.status}`;
          try {
            const errJson = await response.json();
            errText = errJson.message || errJson.detail || errText;
          } catch {}
          throw new Error(errText);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No readable stream available');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split('\n\n');
          buffer = events.pop() || '';

          for (const rawEvent of events) {
            if (!rawEvent.trim()) continue;
            const lines = rawEvent.split('\n');
            let eventType = 'message';
            let dataStr = '';

            for (const line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.replace('event: ', '').trim();
              } else if (line.startsWith('data: ')) {
                dataStr = line.replace('data: ', '').trim();
              }
            }

            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);
              if (eventType === 'citations') {
                handlers.onCitations(data as Citation[]);
              } else if (eventType === 'token') {
                handlers.onToken(data.token || '');
              } else if (eventType === 'verification') {
                handlers.onVerification?.(data.verdict, data.rejected);
              } else if (eventType === 'done') {
                handlers.onDone(data);
              } else if (eventType === 'error') {
                handlers.onError(new Error(data.error || 'Server stream error'));
              }
            } catch (jsonErr) {
              console.error('Error parsing SSE event data', jsonErr, dataStr);
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          handlers.onError(err);
        }
      }
    },
  },
};
