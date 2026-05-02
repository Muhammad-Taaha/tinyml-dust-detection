import { useEffect, useRef, useState, useCallback } from 'react';

const DEFAULT_WS_PORT = 81;
const BASE_RECONNECT_DELAY = 2000;
const MAX_RECONNECT_DELAY = 30000;
const HISTORY_LIMIT = 50;

function toNumber(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function normalizeConfidence(value) {
  let confidence = toNumber(value);
  if (confidence == null) {
    return null;
  }

  if (confidence > 1 && confidence <= 100) {
    confidence = confidence / 100;
  }

  if (confidence < 0) {
    confidence = 0;
  }

  if (confidence > 1) {
    confidence = 1;
  }

  return confidence;
}

function normalizeLabel(value) {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === 'dusty' || normalized === 'clean') {
    return normalized;
  }

  return null;
}

function parseInferenceMessage(payload) {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  if (payload.type && payload.type !== 'inference') {
    return null;
  }

  let label = normalizeLabel(payload.label ?? payload.result ?? payload.class_name);
  const dustyProb = normalizeConfidence(payload.dusty_prob ?? payload.dustyProbability);
  const cleanProb = normalizeConfidence(payload.clean_prob ?? payload.cleanProbability);
  let confidence = normalizeConfidence(payload.confidence);

  if (!label) {
    if (typeof payload.dust_detected === 'boolean') {
      label = payload.dust_detected ? 'dusty' : 'clean';
    } else if (dustyProb != null && cleanProb != null) {
      label = dustyProb >= cleanProb ? 'dusty' : 'clean';
    } else if (dustyProb != null) {
      label = dustyProb >= 0.5 ? 'dusty' : 'clean';
    }
  }

  if (!label) {
    return null;
  }

  if (confidence == null) {
    if (label === 'dusty' && dustyProb != null) {
      confidence = dustyProb;
    } else if (label === 'clean' && cleanProb != null) {
      confidence = cleanProb;
    } else if (dustyProb != null && cleanProb != null) {
      confidence = Math.max(dustyProb, cleanProb);
    } else {
      confidence = 0;
    }
  }

  return {
    label,
    confidence,
    dusty_prob: dustyProb,
    clean_prob: cleanProb,
    timestamp: Date.now(),
  };
}

function buildWebSocketUrl(ipOrUrl, defaultPort = DEFAULT_WS_PORT) {
  const raw = (ipOrUrl ?? '').trim();
  if (!raw) {
    throw new Error('ESP IP is required. Enter an IP like 192.168.1.42.');
  }

  const hasScheme = /^wss?:\/\//i.test(raw);
  const scheme = hasScheme ? raw.split('://')[0].toLowerCase() : 'ws';
  const withoutScheme = hasScheme ? raw.replace(/^wss?:\/\//i, '') : raw;
  const hostAndMaybePath = withoutScheme.split('/')[0].trim();

  if (!hostAndMaybePath) {
    throw new Error('Invalid endpoint. Use <IP>, <IP>:<port>, or ws://<IP>:<port>.');
  }

  let host = hostAndMaybePath;
  let port = '';

  if (hostAndMaybePath.startsWith('[')) {
    const ipv6End = hostAndMaybePath.indexOf(']');
    if (ipv6End === -1) {
      throw new Error('Invalid IPv6 endpoint format.');
    }
    host = hostAndMaybePath.slice(0, ipv6End + 1);
    if (hostAndMaybePath[ipv6End + 1] === ':') {
      port = hostAndMaybePath.slice(ipv6End + 2);
    }
  } else {
    const parts = hostAndMaybePath.split(':');
    if (parts.length > 2) {
      throw new Error('Invalid endpoint format. Too many ":" separators.');
    }
    host = parts[0];
    port = parts[1] ?? '';
  }

  if (!host) {
    throw new Error('Host/IP is missing in endpoint.');
  }

  if (!port) {
    port = String(defaultPort);
  }

  if (!/^\d+$/.test(port)) {
    throw new Error(`Invalid port "${port}".`);
  }

  const portNumber = Number(port);
  if (portNumber < 1 || portNumber > 65535) {
    throw new Error(`Port out of range: ${portNumber}.`);
  }

  return `${scheme}://${host}:${portNumber}`;
}

export function useWebSocket(ip) {
  const ws = useRef(null);
  const reconnectTimer = useRef(null);
  const shouldReconnect = useRef(true);
  const reconnectAttempts = useRef(0);
  const connectRef = useRef(null);

  const [status, setStatus] = useState('disconnected');
  const [lastReading, setLastReading] = useState(null);
  const [history, setHistory] = useState([]);
  const [lastError, setLastError] = useState(null);
  const [connectionUrl, setConnectionUrl] = useState('');

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
  }, []);

  const closeSocket = useCallback((closeCode = 1000, reason = 'client disconnect') => {
    if (!ws.current) {
      return;
    }

    const socket = ws.current;
    ws.current = null;

    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;

    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close(closeCode, reason);
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!shouldReconnect.current) {
      return;
    }

    clearReconnectTimer();
    const delay = Math.min(
      BASE_RECONNECT_DELAY * (2 ** reconnectAttempts.current),
      MAX_RECONNECT_DELAY
    );
    reconnectAttempts.current += 1;

    console.warn(
      `[WS] Disconnected. Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}).`
    );

    reconnectTimer.current = setTimeout(() => {
      connectRef.current?.();
    }, delay);
  }, [clearReconnectTimer]);

  const connect = useCallback(() => {
    let formattedUrl;
    try {
      formattedUrl = buildWebSocketUrl(ip);
    } catch (error) {
      setStatus('disconnected');
      setConnectionUrl('');
      setLastError(error.message);
      console.error('[WS] URL format error:', error.message);
      return;
    }

    shouldReconnect.current = true;
    clearReconnectTimer();
    closeSocket();
    setStatus('connecting');
    setConnectionUrl(formattedUrl);
    setLastError(null);

    console.log(`[WS] Connecting to ${formattedUrl}`);
    const socket = new WebSocket(formattedUrl);
    ws.current = socket;

    socket.onopen = () => {
      setStatus('connected');
      reconnectAttempts.current = 0;
      setLastError(null);
      console.log(`[WS] Connected: ${formattedUrl}`);
    };

    socket.onmessage = (event) => {
      let payload;
      if (typeof event?.data === 'string') {
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          console.warn('[WS] JSON parse error:', event?.data, error);
          return;
        }
      } else if (event?.data && typeof event.data === 'object') {
        payload = event.data;
      } else {
        console.warn('[WS] Unsupported message payload:', event?.data);
        return;
      }

      if (payload?.type === 'pong') {
        return;
      }

      const reading = parseInferenceMessage(payload);
      if (!reading) {
        return;
      }

      setLastReading(reading);
      setHistory(prev => [reading, ...prev].slice(0, HISTORY_LIMIT));
    };

    socket.onerror = (event) => {
      const message =
        event?.message ||
        'WebSocket connection error. Check ESP IP, port, and local network permissions.';
      setLastError(message);
      console.error('[WS] Error event:', { message, url: formattedUrl, event });
    };

    socket.onclose = (event) => {
      if (ws.current === socket) {
        ws.current = null;
      }

      setStatus('disconnected');
      if (event?.code && event.code !== 1000) {
        const closeReason = event?.reason ? ` - ${event.reason}` : '';
        setLastError(`Socket closed (${event.code}${closeReason})`);
      }

      console.warn(
        `[WS] Closed: code=${event?.code ?? 'n/a'} reason=${event?.reason || 'none'}`
      );

      if (shouldReconnect.current) {
        scheduleReconnect();
      }
    };
  }, [ip, clearReconnectTimer, closeSocket, scheduleReconnect]);

  connectRef.current = connect;

  const disconnect = useCallback(() => {
    shouldReconnect.current = false;
    clearReconnectTimer();
    closeSocket();
    setStatus('disconnected');
  }, [clearReconnectTimer, closeSocket]);

  useEffect(() => {
    if (!ip?.trim()) {
      disconnect();
      setLastError('Set ESP IP first. Expected endpoint: ws://<IP>:81');
      return;
    }

    connect();

    return () => {
      shouldReconnect.current = false;
      clearReconnectTimer();
      closeSocket();
    };
  }, [ip, connect, disconnect, clearReconnectTimer, closeSocket]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        try {
          ws.current.send('ping');
        } catch (error) {
          console.warn('[WS] Ping failed:', error);
        }
      }
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  return {
    status,
    lastReading,
    history,
    connect,
    disconnect,
    lastError,
    connectionUrl,
    defaultPort: DEFAULT_WS_PORT,
  };
}
