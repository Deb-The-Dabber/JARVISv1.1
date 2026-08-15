import { useState, useEffect } from 'react';
import { useStore } from '../../stores/useAppStore';
import { useSessionStore } from '../../stores/useSessionStore';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function SessionSwitcher() {
  const { sessions, activeId, fetchSessions, setActive, createSession } = useSessionStore();

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const handleNew = async () => {
    const name = window.prompt('New session name (optional):');
    if (name === null) return;
    await createSession(name || undefined);
  };

  return (
    <div className="session-switcher" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <select
        value={activeId}
        onChange={(e) => setActive(e.target.value)}
        title="Switch session"
        style={{
          background: 'var(--bg-elev)',
          color: 'var(--fg)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '2px 6px',
          fontSize: '0.7rem',
          maxWidth: '160px',
        }}
      >
        {sessions.length === 0 && <option value={activeId}>default</option>}
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} ({s.message_count})
          </option>
        ))}
      </select>
      <button
        onClick={handleNew}
        title="New session"
        style={{
          background: 'var(--bg-elev)',
          color: 'var(--fg)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '2px 8px',
          fontSize: '0.7rem',
          cursor: 'pointer',
        }}
      >
        + New
      </button>
    </div>
  );
}

export function Header() {
  const { connected } = useStore();
  const [online, setOnline] = useState(connected);
  const [model, setModel] = useState('MODEL');
  const [time, setTime] = useState('--:--:--');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const dateStr = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
      setTime(`${dateStr} · ${timeStr}`);
      const footerTime = document.getElementById('footerTime');
      if (footerTime) footerTime.textContent = timeStr;
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);

  // Poll /health like the static UI — drives ONLINE/OFFLINE + model name
  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch(`${API}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setOnline(true);
        const runtime = data.runtime || {};
        const last = runtime.model_last_used;
        const pref = runtime.model_preferred;
        setModel((last && last !== 'unknown' ? last : pref) || 'MODEL');
      } catch (e) {
        setOnline(false);
      }
    };
    checkServer();
    const id = setInterval(checkServer, 20000);
    return () => clearInterval(id);
  }, []);

  const isOnline = online;

  return (
    <header className="header">
      <div className="logo">
        <div>
          <h1>J.A.R.V.I.S.</h1>
          <div className="version">JUST A RATHER VERY INTELLIGENT SYSTEM · v3.1</div>
        </div>
      </div>
      <div className="status-bar">
        <div className="status-item">
          <div className={isOnline ? "dot" : "dot offline"} id="serverDot"></div>
          <span id="serverStatus">{isOnline ? "ONLINE" : "OFFLINE"}</span>
        </div>
        <div className="status-item">
          <div className={isOnline ? "dot" : "dot offline"} id="modelDot"></div>
          <span id="modelStatus">{model}</span>
        </div>
        <div className="status-item">
          <SessionSwitcher />
        </div>
        <div className="status-item" id="timeDisplay">{time}</div>
      </div>
    </header>
  );
}
