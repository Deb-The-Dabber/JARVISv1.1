import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionMeta, SessionMessage } from '../types';
import {
  getSessions,
  createSession as apiCreateSession,
  getSessionHistory,
} from '../services/api';

interface SessionState {
  sessions: SessionMeta[];
  activeId: string;
  history: SessionMessage[];
  loading: boolean;
  fetchSessions: () => Promise<void>;
  setActive: (id: string) => Promise<void>;
  createSession: (name?: string) => Promise<void>;
}

function formatTranscript(history: SessionMessage[]): string {
  return history
    .map((m) => (m.role === 'user' ? `You: ${m.content}` : `Jarvis: ${m.content}`))
    .join('\n\n');
}

function renderReply(text: string) {
  window.dispatchEvent(new CustomEvent('jarvis-reply', { detail: { reply: text } }));
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeId: 'default',
      history: [],
      loading: false,

      fetchSessions: async () => {
        try {
          const sessions = await getSessions();
          set({ sessions, loading: false });
          const current = get().activeId;
          if (!sessions.some((s) => s.id === current)) {
            set({ activeId: sessions[0]?.id || 'default' });
          }
        } catch (e) {
          console.error('fetchSessions failed:', e);
          set({ loading: false });
        }
      },

      setActive: async (id) => {
        set({ activeId: id });
        try {
          const { messages } = await getSessionHistory(id);
          set({ history: messages });
          renderReply(
            messages.length
              ? formatTranscript(messages)
              : 'No messages yet in this session.'
          );
        } catch (e) {
          console.error('setActive failed:', e);
        }
      },

      createSession: async (name) => {
        try {
          await apiCreateSession(name);
          await get().fetchSessions();
          const sessions = get().sessions;
          if (sessions.length) {
            await get().setActive(sessions[0].id);
          }
        } catch (e) {
          console.error('createSession failed:', e);
        }
      },
    }),
    {
      name: 'jarvis-session-store',
      partialize: (s) => ({ activeId: s.activeId }),
    }
  )
);
