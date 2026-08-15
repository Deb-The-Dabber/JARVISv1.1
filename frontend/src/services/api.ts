import type {
  AutomationSession,
  AutomationTemplate,
  SystemStats,
  ProviderHealth,
  MemoryStats,
  AutomationStartRequest,
  AutomationResponse,
  ApiResponse,
  SessionMeta,
  SessionMessage,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('jarvis-access-token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchJson<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data as ApiResponse<T>;
}

async function fetchText(endpoint: string, options: RequestInit = {}): Promise<string> {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
  });
  const text = await res.text();
  if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
  return text;
}

// Health & system
export async function getSystemInfo(): Promise<SystemStats> {
  const res = await fetchJson<SystemStats>('/system');
  return (res.data as SystemStats) ?? (res as unknown as SystemStats);
}

export async function getHealth(): Promise<any> {
  const res = await fetchJson<any>('/health');
  return res.data;
}

export async function getProviderHealth(): Promise<ProviderHealth[]> {
  const res = await fetchJson<ProviderHealth[]>('/health/providers');
  return (res.data ?? []) as ProviderHealth[];
}

export async function getMemoryStats(): Promise<MemoryStats> {
  const res = await fetchJson<MemoryStats>('/memory/stats');
  return (res.data as MemoryStats) ?? (res as unknown as MemoryStats);
}

export async function getMemories(): Promise<any> {
  const res = await fetchJson<any>('/memories');
  return res.data;
}

// Chat
export async function askText(text: string, tts = 'server', sessionId = 'default'): Promise<any> {
  const res = await fetchJson<any>('/ask', {
    method: 'POST',
    body: JSON.stringify({ text, tts, session_id: sessionId }),
  });
  return res.data;
}

// Sessions (persistent conversation threads, shared across devices)
export async function getSessions(): Promise<SessionMeta[]> {
  const res = await fetchJson<any>('/sessions');
  return (res.data?.sessions ?? []) as SessionMeta[];
}

export async function createSession(name?: string): Promise<SessionMeta> {
  const res = await fetchJson<any>('/sessions', {
    method: 'POST',
    body: JSON.stringify({ name: name ?? null }),
  });
  return res.data?.session as SessionMeta;
}

export async function getSessionHistory(
  id: string
): Promise<{ session: SessionMeta; messages: SessionMessage[] }> {
  const res = await fetchJson<any>(`/sessions/${encodeURIComponent(id)}/history`);
  return res.data as { session: SessionMeta; messages: SessionMessage[] };
}

export async function renameSession(id: string, name: string): Promise<SessionMeta> {
  const res = await fetchJson<any>(`/sessions/${encodeURIComponent(id)}/rename`, {
    method: 'POST',
    body: JSON.stringify({ text: name }),
  });
  return res.data?.session as SessionMeta;
}

export async function resetSession(id: string): Promise<void> {
  await fetchJson(`/sessions/${encodeURIComponent(id)}/reset`, { method: 'POST' });
}

export async function deleteSession(id: string): Promise<void> {
  await fetchJson(`/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

// Remote control
export async function sendRemoteCommand(command: string, params: any): Promise<any> {
  return fetchJson('/remote/command', {
    method: 'POST',
    body: JSON.stringify({ command, params }),
  });
}

export async function remoteConfirm(command: string, params: any): Promise<any> {
  return fetchJson('/remote/confirm', {
    method: 'POST',
    body: JSON.stringify({ command, params }),
  });
}

// OAuth
export async function oauthAuthorize(provider: string): Promise<string> {
  const res = await fetch(`${API_BASE}/oauth/authorize/${provider}`, {
    headers: getAuthHeaders(),
  });
  return res.url;
}

export async function getOAuthStatus(): Promise<any> {
  const res = await fetchJson<any>('/oauth/status');
  return res.data;
}

export async function disconnectOAuth(provider: string): Promise<any> {
  return fetchJson(`/oauth/disconnect/${provider}`, { method: 'POST' });
}

// Learner
export async function triggerLearning(prompt: string): Promise<any> {
  return fetchJson('/learner/trigger', {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  });
}

export async function getLearnerTools(): Promise<any> {
  const res = await fetchJson<any>('/learner/tools');
  return res.data;
}

export async function getLearnerStats(): Promise<any> {
  const res = await fetchJson<any>('/learner/stats');
  return res.data;
}

export async function deleteLearnedTool(name: string): Promise<any> {
  return fetchJson(`/learner/tools/${name}`, { method: 'DELETE' });
}

// Workflows
export async function getWorkflows(): Promise<any> {
  const res = await fetchJson<any>('/workflows');
  return res.data;
}

export async function runWorkflow(name: string, params: any): Promise<any> {
  return fetchJson('/workflows/run', {
    method: 'POST',
    body: JSON.stringify({ name, params }),
  });
}

export async function getWorkflowHistory(limit = 20): Promise<any> {
  const res = await fetchJson<any>(`/workflows/history?limit=${limit}`);
  return res.data;
}

// Agents
export async function getAgentStatus(agentId: string): Promise<any> {
  const res = await fetchJson<any>(`/agents/${agentId}`);
  return res.data;
}

export async function listAgents(): Promise<any[]> {
  const res = await fetchJson<any>('/agents');
  return (res.data?.agents ?? []) as any[];
}

export async function spawnAgent(goal: string): Promise<any> {
  return fetchJson('/agents/spawn', { method: 'POST', body: JSON.stringify({ goal }) });
}

export async function stopAgent(agentId: string): Promise<any> {
  return fetchJson(`/agents/${agentId}/stop`, { method: 'POST' });
}

// Automation
export async function startAutomation(taskDescription: string): Promise<AutomationResponse> {
  const body: AutomationStartRequest = { task_description: taskDescription };
  const res = await fetchJson<AutomationResponse>('/automation/start', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return res.data ?? ({ response: '', status: 'error' } as AutomationResponse);
}

export async function automationRespond(response: string): Promise<string> {
  return fetchText('/automation/respond', {
    method: 'POST',
    body: JSON.stringify({ response }),
  });
}

export async function getAutomationStatus(): Promise<string> {
  return fetchText('/automation/status');
}

export async function cancelAutomation(): Promise<string> {
  return fetchText('/automation/cancel', { method: 'POST' });
}

export async function getAutomationTemplates(): Promise<AutomationTemplate[]> {
  const res = await fetchJson<AutomationTemplate[]>('/automation/templates');
  return (res.data ?? []) as AutomationTemplate[];
}

export async function saveAutomationTemplate(template: any): Promise<any> {
  const res = await fetchJson<any>('/automation/templates', {
    method: 'POST',
    body: JSON.stringify(template),
  });
  return res.data;
}

// Runtime / metrics
export async function getRuntimeStatus(): Promise<any> {
  const res = await fetchJson<any>('/runtime');
  return res.data;
}

export async function getMetrics(): Promise<string> {
  return fetchText('/metrics');
}

export async function getRecap(hours = 8): Promise<string> {
  return fetchText(`/recap?hours=${hours}`);
}

export async function getRecentEvents(hours = 2): Promise<string> {
  return fetchText(`/recent_events?hours=${hours}`);
}

// Screen
export async function getScreenLatest(): Promise<Blob> {
  const res = await fetch(`${API_BASE}/screen/latest?t=${Date.now()}`);
  return res.blob();
}

export async function analyzeScreen(question = 'Describe what is on the screen.'): Promise<any> {
  return fetchJson('/screen/analyze', {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

// Weather
export async function getWeather(): Promise<any> {
  const res = await fetchJson<any>('/weather');
  return res.data;
}

export async function getWeatherDetailed(): Promise<any> {
  const res = await fetchJson<any>('/weather/detailed');
  return res.data;
}

// Audit / priorities / notifications
export async function getAuditLog(limit = 50): Promise<any[]> {
  const res = await fetchJson<any>(`/audit?limit=${limit}`);
  return (res.data?.logs ?? []) as any[];
}

export async function getPriorities(): Promise<any[]> {
  const res = await fetchJson<any>('/priorities');
  return (res.data?.priorities ?? []) as any[];
}

export async function getNotifications(unreadOnly = true, limit = 50): Promise<any[]> {
  const res = await fetchJson<any>(`/notices?unread_only=${unreadOnly}&limit=${limit}`);
  return (res.data?.notifications ?? []) as any[];
}

export async function markNotificationRead(id: number): Promise<void> {
  await fetchJson(`/notices/${id}/read`, { method: 'POST' });
}

export async function markAllNotificationsRead(): Promise<void> {
  await fetchJson('/notices/read-all', { method: 'POST' });
}

// App / system control
export async function openApp(appName: string): Promise<string> {
  return fetchText('/open_app', { method: 'POST', body: JSON.stringify({ app_name: appName }) });
}

export async function quitApp(appName: string): Promise<string> {
  return fetchText('/quit_app', { method: 'POST', body: JSON.stringify({ app_name: appName }) });
}

export async function openUrl(url: string): Promise<string> {
  return fetchText('/open_url', { method: 'POST', body: JSON.stringify({ url }) });
}

export async function runTerminalCommand(cmd: string): Promise<string> {
  return fetchText('/run_terminal_command', { method: 'POST', body: JSON.stringify({ command: cmd }) });
}

export async function webSearch(query: string): Promise<any> {
  return fetchJson('/web_search', { method: 'POST', body: JSON.stringify({ query }) });
}

// Browser automation
export async function browserNavigate(url: string): Promise<string> {
  return fetchText('/browser_navigate', { method: 'POST', body: JSON.stringify({ url }) });
}

export async function browserQuickSearch(query: string): Promise<any> {
  return fetchJson('/browser_quick_search', { method: 'POST', body: JSON.stringify({ query }) });
}

export async function browserClick(x: number, y: number): Promise<string> {
  return fetchText('/browser_click', { method: 'POST', body: JSON.stringify({ x, y }) });
}

export async function browserType(text: string): Promise<string> {
  return fetchText('/browser_type', { method: 'POST', body: JSON.stringify({ text }) });
}

export async function browserPressKey(key: string): Promise<string> {
  return fetchText('/browser_press_key', { method: 'POST', body: JSON.stringify({ key }) });
}

export async function browserScroll(direction: 'up' | 'down', amount = 500): Promise<string> {
  return fetchText('/browser_scroll', {
    method: 'POST',
    body: JSON.stringify({ direction, amount }),
  });
}

export async function readScreen(): Promise<string> {
  return fetchText('/read_screen');
}

export async function findOnScreen(query: string): Promise<any> {
  return fetchJson('/find_on_screen', { method: 'POST', body: JSON.stringify({ query }) });
}

export async function summarizeScreen(): Promise<string> {
  return fetchText('/screen/summarize');
}

export async function takeScreenshot(): Promise<string> {
  return fetchText('/take_screenshot');
}

export async function getTopProcesses(by: 'memory' | 'cpu' = 'memory', count = 10): Promise<string> {
  return fetchText('/get_top_processes', {
    method: 'POST',
    body: JSON.stringify({ by, count }),
  });
}

// File ops
export async function listDirectory(path = '~/Jarvis'): Promise<string> {
  return fetchText('/list_directory', { method: 'POST', body: JSON.stringify({ path }) });
}

export async function readFile(path: string): Promise<string> {
  return fetchText('/read_file', { method: 'POST', body: JSON.stringify({ path }) });
}

export async function writeFile(
  filename: string,
  content: string,
  path = '~/Desktop'
): Promise<string> {
  return fetchText('/create_file', {
    method: 'POST',
    body: JSON.stringify({ filename, content, path }),
  });
}

export async function appendFile(
  filename: string,
  content: string,
  path = '~/Desktop'
): Promise<string> {
  return fetchText('/append_file', {
    method: 'POST',
    body: JSON.stringify({ filename, content, path }),
  });
}

export async function deleteFile(path: string): Promise<string> {
  return fetchText('/delete_file', { method: 'POST', body: JSON.stringify({ path }) });
}

export async function moveFile(src: string, dest: string): Promise<string> {
  return fetchText('/move_file', { method: 'POST', body: JSON.stringify({ src, dest }) });
}

export async function copyFile(src: string, dest: string): Promise<string> {
  return fetchText('/copy_file', { method: 'POST', body: JSON.stringify({ src, dest }) });
}

export async function searchFiles(query: string, path = '~/Jarvis'): Promise<string> {
  return fetchText('/search_in_files', {
    method: 'POST',
    body: JSON.stringify({ query, path }),
  });
}

export async function getLargestFiles(folder = '~/Downloads', count = 5): Promise<string> {
  return fetchText('/get_largest_files', {
    method: 'POST',
    body: JSON.stringify({ folder, count }),
  });
}

export async function organizeDownloads(): Promise<string> {
  return fetchText('/organize_downloads');
}

export async function openInFinder(path: string): Promise<string> {
  return fetchText('/open_in_finder', { method: 'POST', body: JSON.stringify({ path }) });
}

// App focus
export async function getOpenApps(): Promise<string> {
  return fetchText('/get_open_apps');
}

export async function focusApp(appName: string): Promise<string> {
  return fetchText('/focus_app', { method: 'POST', body: JSON.stringify({ app_name: appName }) });
}

// Timers
export async function setTimer(label: string, seconds: number): Promise<string> {
  return fetchText('/set_timer', {
    method: 'POST',
    body: JSON.stringify({ label, seconds }),
  });
}

export async function cancelTimer(): Promise<string> {
  return fetchText('/cancel_timer');
}

// Knowledge / memory
export async function searchMyNotes(query: string): Promise<string> {
  return fetchText('/search_my_notes', { method: 'POST', body: JSON.stringify({ query }) });
}

export async function queryMyKnowledgeGraph(entity: string): Promise<string> {
  return fetchText('/query_my_knowledge_graph', {
    method: 'POST',
    body: JSON.stringify({ entity }),
  });
}

export async function addToKnowledgeGraph(
  entity1: string,
  relationship: string,
  entity2: string
): Promise<string> {
  return fetchText('/add_to_knowledge_graph', {
    method: 'POST',
    body: JSON.stringify({ entity1, relationship, entity2 }),
  });
}

export async function semanticSearchMemory(query: string): Promise<string> {
  return fetchText('/semantic_search_memory', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

export async function warwatchNews(query = 'latest'): Promise<string> {
  return fetchText('/warwatch_news', { method: 'POST', body: JSON.stringify({ query }) });
}

// API keys
export async function saveApiKey(keyName: string, keyValue: string): Promise<string> {
  return fetchText('/save_api_key', {
    method: 'POST',
    body: JSON.stringify({ key_name: keyName, key_value: keyValue }),
  });
}

export async function listConfiguredKeys(masked = true): Promise<string> {
  return fetchText('/list_configured_keys', { method: 'POST', body: JSON.stringify({ masked }) });
}
