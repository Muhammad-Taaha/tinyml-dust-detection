import { useEffect, useRef, useState, useCallback } from 'react';

const RECONNECT_DELAY = 3000;

export function useWebSocket(ip) {
  const ws            = useRef(null);
  const reconnectTimer = useRef(null);
  const [status, setStatus]   = useState('disconnected'); // disconnected | connecting | connected
  const [lastReading, setLastReading] = useState(null);
  const [history, setHistory] = useState([]);             // last 50 readings

  const connect = useCallback(() => {
    if (!ip) return;
    setStatus('connecting');

    const socket = new WebSocket(`ws://${ip}:81`);
    ws.current = socket;

    socket.onopen = () => {
      setStatus('connected');
      clearTimeout(reconnectTimer.current);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'inference') {
          const reading = {
            label:      data.label,
            confidence: parseFloat(data.confidence.toFixed(3)),
            timestamp:  Date.now(),
          };
          setLastReading(reading);
          setHistory(prev => [reading, ...prev].slice(0, 50));
        }
      } catch (e) {
        console.warn('Parse error:', e);
      }
    };

    socket.onerror = (e) => {
      console.warn('WebSocket error:', e.message);
    };

    socket.onclose = () => {
      setStatus('disconnected');
      // Auto-reconnect
      reconnectTimer.current = setTimeout(() => connect(), RECONNECT_DELAY);
    };
  }, [ip]);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    ws.current?.close();
    setStatus('disconnected');
  }, []);

  useEffect(() => {
    if (ip) connect();
    return () => disconnect();
  }, [ip]);

  // Heartbeat ping every 10s
  useEffect(() => {
    const interval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send('ping');
      }
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  return { status, lastReading, history, connect, disconnect };
}
