import { useState } from 'react';
import { useStore } from '../stores/useAppStore';
import { useSessionStore } from '../stores/useSessionStore';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function InputRow() {
  const [text, setText] = useState('');
  const { setOrbState, addActivity } = useStore();
  const activeId = useSessionStore((s) => s.activeId);

  const setReply = (reply: string) => {
    const el = document.getElementById('reply');
    if (el) el.textContent = reply;
  };

  const handleSend = async () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setText('');

    const lower = trimmed.toLowerCase();
    const yesWords = ['yes', 'yeah', 'go ahead', 'confirm', 'sure', 'ok', 'approved'];
    const noWords = ['no', 'cancel', 'stop', 'nevermind', 'deny'];

    if (yesWords.some((w) => lower.includes(w))) {
      addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: '✅ Confirmed by user', type: 'success' });
      return;
    }
    if (noWords.some((w) => lower.includes(w))) {
      addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: '🚫 Cancelled by user', type: 'warning' });
      return;
    }

    addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: `You: ${trimmed}`, type: 'info' });
    setOrbState('thinking');
    setReply('Processing...');

    try {
      const ttsOnPhone = (document.getElementById('ttsClientToggle') as HTMLInputElement)?.checked;
      const res = await fetch(`${API}/ask${ttsOnPhone ? '?tts=client' : ''}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed, session_id: activeId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setReply(data.reply || '');
      if (data.reply) {
        addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: `🤖 ${data.reply.slice(0, 120)}`, type: 'success' });
      }
      setOrbState('speaking');
      setTimeout(() => setOrbState('idle'), 3000);
    } catch (e) {
      setReply('Error — is Jarvis running?');
      addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: 'Error — is Jarvis running?', type: 'error' });
      setOrbState('idle');
    }
  };

  return (
    <div className="panel">
      <div className="input-row">
        <input
          type="text"
          id="textInput"
          placeholder="Enter command..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button className="btn btn-primary" id="micBtn" onClick={handleSend}>
          SEND
        </button>
        <button className="btn btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent('jarvis-stop'))}>
          STOP
        </button>
      </div>
      <div
        className="tts-toggle"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginTop: '8px',
          fontSize: '0.75rem',
          color: 'var(--fg-dim)',
        }}
      >
        <input type="checkbox" id="ttsClientToggle" defaultChecked />
        <label htmlFor="ttsClientToggle">📱 TTS on this device</label>
        <span id="ttsStatus" style={{ marginLeft: 'auto', fontSize: '0.7rem' }}></span>
      </div>
    </div>
  );
}
