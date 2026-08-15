export type OrbState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'blocked' | 'warning';

export type TaskType = 'coding' | 'research' | 'file_ops' | 'automation' | 'building' | 'debugging' | 'refactoring' | 'general';

export type AutomationState = 'init' | 'awaiting_ide' | 'awaiting_folder' | 'awaiting_task_details' | 'executing' | 'paused' | 'completed' | 'failed' | 'cancelled';

export type IDEType = 'vscode' | 'cursor' | 'pycharm' | 'intellij' | 'sublime' | 'vim' | 'neovim' | 'emacs' | 'other';

export interface AutomationSession {
  session_id: string;
  task_description: string;
  task_type: TaskType;
  state: AutomationState;
  ide: IDEType | null;
  project_folder: string | null;
  task_details: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
  execution_log: Array<{
    step: number;
    action: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    result?: string;
    timestamp: string;
  }>;
}

export interface Activity {
  id: string;
  timestamp: string;
  text: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'blocked';
}

export interface SystemStats {
  cpu: number;
  ram: number;
  disk: { free: number; total: number };
  network: { up: number; down: number };
}

export interface ProviderHealth {
  name: string;
  health: number;
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  successes: number;
  failures: number;
  last_check: string;
}

export interface MemoryStats {
  explicit_memories: number;
  vector_entries: number;
  rag_chunks: number;
  graph_entities: number;
  procedures: number;
  associations: number;
}

export interface AutomationTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  steps: Array<{
    id: string;
    action: string;
    params: Record<string, any>;
    description: string;
  }>;
  triggers: Array<{
    type: string;
    config: Record<string, any>;
  }>;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface SessionMeta {
  id: string;
  name: string;
  created: string;
  updated: string;
  message_count: number;
  preview: string;
}

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  ts?: string;
}

export interface WebSocketMessage {
  type: 'audio' | 'end_audio' | 'text' | 'ping' | 'pong' | 'partial' | 'transcribed' | 'final' | 'tool_call' | 'tool_result' | 'reply' | 'tts_start' | 'tts_chunk' | 'tts_done' | 'error';
  data?: any;
  text?: string;
  chunk?: string;
  error?: string;
}

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: 'ok' | 'error';
}

export interface AutomationStartRequest {
  task_description: string;
}

export interface AutomationResponse {
  response: string;
  session_id?: string;
  status: 'success' | 'error' | 'pending';
}