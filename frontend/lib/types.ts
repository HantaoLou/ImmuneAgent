export interface Session {
  id: string;
  title: string;
  createdAt: string;
  messages: Message[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  taskType?: string;
  status?: 'streaming' | 'done' | 'error';
  sessionId?: string;
  progressEvents?: ProgressEvent[];
  outputFiles?: OutputFile[];
}

export interface ProgressEvent {
  event_type: string;
  node_name?: string;
  task_id?: string;
  message: string;
  details: Record<string, any>;
  timestamp: string;
  progress_percent?: number;
}

export type ProgressEventType = 
  | 'node_start'
  | 'node_progress'
  | 'node_complete'
  | 'task_start'
  | 'task_progress'
  | 'task_complete'
  | 'code_generation'
  | 'code_execution'
  | 'tool_call'
  | 'error'
  | 'info'
  | 'llm_thinking'
  | 'llm_reasoning'
  | 'llm_streaming'
  | 'tool_result'
  | 'subgraph_step'
  | 'knowledge_retrieval'
  | 'analysis_progress';

export interface ThinkingStep {
  id: string;
  timestamp: string;
  content: string;
  step_name?: string;
  node_name?: string;
}

export interface ExecutionStep {
  id: string;
  type: ProgressEventType;
  timestamp: string;
  message: string;
  node_name?: string;
  progress_percent?: number;
  details?: Record<string, any>;
}

export interface Template {
  id: string;
  name: string;
  content: string;
  category: string;
}

export interface SSEEvent {
  event: string;
  data: string;
}

export interface AgentResponse {
  session_id: string;
  task_type: string;
  answer?: string; // 友好的答案（优先显示）
  result: {
    merged_result: Record<string, any>;
    completed_tasks: Record<string, any>;
    file_paths: Record<string, string>;
    execution_plan?: string;
  };
  sandbox_dir: string;
  supervisor?: {
    decision?: string;
    reasoning?: string;
  };
  summary?: {
    answer?: string;
    task_type?: string;
    status?: string;
    key_findings?: string[];
  };
  output_files?: OutputFile[];
  output_files_count?: number;
}

export interface OutputFile {
  name: string;
  path: string;
  relative_path: string;
  size: number;
  size_formatted: string;
  type: string;
  extension: string;
}
