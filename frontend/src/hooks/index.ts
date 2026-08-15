import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppStore } from '../stores/useAppStore';
import { useAuthStore } from '../stores/useAuthStore';
import { useSessionStore } from '../stores/useSessionStore';

export interface WebSocketMessage {
  type: string;
  data?: any;
  text?: string;
  chunk?: string;
  error?: string;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const { connected, setConnected, fallbackMode, setFallbackMode, addActivity } = useAppStore();
  const { accessToken } = useAuthStore();
  const activeId = useSessionStore((s) => s.activeId);
  const [wsReady, setWsReady] = useState(false);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const sessionId = useSessionStore.getState().activeId || 'default';
    const wsUrl = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ws?session_id=${encodeURIComponent(sessionId)}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected');
      setWsReady(true);
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        console.error('[WS] Parse error:', e);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected');
      setWsReady(false);
      setConnected(false);
      // Reconnect after 3 seconds
      setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };
  }, [setConnected]);

  const handleMessage = (msg: any) => {
    switch (msg.type) {
      case 'partial':
        // Partial transcription
        break;
      case 'transcribed':
        addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: `🎙 You: "${msg.text}"`, type: 'info' });
        break;
      case 'final':
      case 'reply':
        // Handle response in the main component
        window.dispatchEvent(new CustomEvent('jarvis-reply', { detail: msg }));
        break;
      case 'tool_call':
        addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: `🔧 ${msg.tool}(${JSON.stringify(msg.args)})`, type: 'info' });
        break;
      case 'tool_result':
        addActivity({ id: crypto.randomUUID(), time: new Date().toISOString(), text: `✅ ${msg.tool} completed`, type: 'success' });
        break;
      case 'tts_start':
        // TTS started
        break;
      case 'tts_chunk':
        // Audio chunk received - handled by audio player
        break;
      case 'tts_done':
        break;
      case 'error':
        console.error('WS Error:', msg.error);
        break;
      case 'pong':
        break;
    }
  };

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, activeId]);

  const sendAudio = useCallback((audioBlob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const reader = new FileReader();
      reader.onload = () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(reader.result as ArrayBuffer);
        }
      };
      reader.readAsArrayBuffer(audioBlob);
    }
  }, []);

  const sendText = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'text', text }));
    }
  }, []);

  const sendEndAudio = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(new Uint8Array([69, 78, 68])); // "END"
    }
  }, []);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  return {
    wsReady,
    connected,
    sendAudio,
    sendText,
    sendEndAudio,
    sendPing,
    connect,
    disconnect: () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    },
  };
}

export function useAudio() {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const { sendAudio, sendEndAudio, wsReady } = useWebSocket();
  
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      // Safari doesn't support WebM/Opus — pick the first codec this browser supports.
      const SUPPORTED_MIMES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4;codecs=mp4a.40.2', 'audio/mp4'];
      let recorderOptions;
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported) {
        const chosen = SUPPORTED_MIMES.find((m) => MediaRecorder.isTypeSupported(m));
        if (chosen) recorderOptions = { mimeType: chosen };
      }

      const mediaRecorder = new MediaRecorder(stream, recorderOptions);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start(250); // Send chunks every 250ms

      // Store references
      mediaRecorderRef.current = mediaRecorder;
      streamRef.current = stream;

      // Send chunks as they arrive
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          sendAudio(e.data);
        }
      };
    } catch (e) {
      console.error('Audio error:', e);
      return false;
    }
  }, []);

  const stopRecording = async () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  return {
    isRecording,
    startRecording,
    stopRecording,
  };
}

export function useOrb() {
  const { orb, setOrbState } = useAppStore();
  return { orb, setOrbState };
}

export function useAuth() {
  return useAuthStore();
}