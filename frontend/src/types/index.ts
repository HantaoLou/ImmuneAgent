export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  sessionId: string;
  uploadTime: number;
  category: 'image' | 'document' | 'code' | 'data' | 'other';
  uploadProgress?: number;
}

export interface LogEntry {
  id: string;
  event_type: string;
  message: string;
  timestamp: string;
  node_name?: string;
  details?: Record<string, any>;
}

export interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: number;
  status: 'success' | 'loading' | 'error';
  attachments?: FileAttachment[];
  executionLogs?: LogEntry[];
  hitlRequest?: HITLRequest;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  createTime: number;
  updateTime: number;
}

export interface SessionFiles {
  sessionId: string;
  files: FileAttachment[];
  totalSize: number;
}

export interface ChatRequest {
  sessionId: string;
  messages: Pick<Message, 'role' | 'content'>[];
  attachments?: FileAttachment[];
}

export interface ChatResponse {
  content: string;
  sessionId: string;
}

export type FileCategory = 'image' | 'document' | 'code' | 'data' | 'other';

export interface FileFilter {
  category?: FileCategory;
  searchQuery?: string;
}

export interface SandboxFile {
  name: string;
  path: string;
  relative_path: string;
  size: number;
  size_formatted: string;
  type: string;
  extension: string;
  source: string;
  modified?: string;
}

export interface SandboxFilesResponse {
  session_id: string;
  files: SandboxFile[];
  count: number;
  source: string;
}

export interface MissingParameter {
  name: string;
  description: string;
  type: string;
  required: boolean;
}

export interface HITLRequest {
  type: string;
  session_id: string;
  task_md: string;
  missing_parameters: MissingParameter[];
  iteration: number;
  max_iterations: number;
  timestamp: string;
  previous_feedback: string | null;
  hitl_id: string;
}
