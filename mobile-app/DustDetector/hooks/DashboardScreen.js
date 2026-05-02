import React from 'react';
import {
  View, Text, StyleSheet,
  TouchableOpacity, ScrollView
} from 'react-native';

const STATUS_CONFIG = {
  clean: { color: '#1D9E75', bg: '#E1F5EE', emoji: '✅', label: 'Clean' },
  dusty: { color: '#BA7517', bg: '#FAEEDA', emoji: '⚠️', label: 'Dusty' },
};

function ConfidenceBar({ value, color }) {
  const widthValue = Math.max(0, Math.min(1, Number(value) || 0));

  return (
    <View style={styles.barBg}>
      <View style={[styles.barFill, { width: `${widthValue * 100}%`, backgroundColor: color }]} />
    </View>
  );
}

function StatusCard({ reading, status, connectionUrl, lastError, defaultPort }) {
  const cfg = STATUS_CONFIG[reading?.label] ?? STATUS_CONFIG.clean;

  return (
    <View style={[styles.card, { borderColor: cfg.color, backgroundColor: cfg.bg }]}>
      <Text style={styles.cardEmoji}>{cfg.emoji}</Text>
      <Text style={[styles.cardLabel, { color: cfg.color }]}>{cfg.label}</Text>
      {reading && (
        <>
          <ConfidenceBar value={reading.confidence} color={cfg.color} />
          <Text style={styles.confidence}>
            {(reading.confidence * 100).toFixed(1)}% confidence
          </Text>
        </>
      )}
      <Text style={[styles.connStatus, { color: status === 'connected' ? '#0F6E56' : '#A32D2D' }]}>
        {status === 'connected' ? '● Live' : status === 'connecting' ? '◌ Connecting…' : '○ Disconnected'}
      </Text>
      <Text style={styles.endpoint}>Endpoint: {connectionUrl || `ws://<ESP-IP>:${defaultPort}`}</Text>
      {lastError ? <Text style={styles.errorText}>Last error: {lastError}</Text> : null}
    </View>
  );
}

export default function DashboardScreen({
  lastReading,
  status,
  onReconnect,
  connectionUrl,
  lastError,
  defaultPort = 81,
}) {
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Solar Panel Monitor</Text>
      <StatusCard
        reading={lastReading}
        status={status}
        connectionUrl={connectionUrl}
        lastError={lastError}
        defaultPort={defaultPort}
      />

      {status !== 'connected' && (
        <TouchableOpacity style={styles.reconnectBtn} onPress={onReconnect}>
          <Text style={styles.reconnectText}>Reconnect</Text>
        </TouchableOpacity>
      )}

      {lastReading && (
        <View style={styles.details}>
          <Text style={styles.detailRow}>
            Last checked: {new Date(lastReading.timestamp).toLocaleTimeString()}
          </Text>
          <Text style={styles.detailRow}>
            Result: {lastReading.label.toUpperCase()}
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 24, alignItems: 'center' },
  title: { fontSize: 22, fontWeight: '600', marginBottom: 28, color: '#2C2C2A' },
  card: {
    width: '100%',
    borderRadius: 16,
    borderWidth: 2,
    padding: 28,
    alignItems: 'center',
    marginBottom: 20,
  },
  cardEmoji: { fontSize: 56, marginBottom: 8 },
  cardLabel: { fontSize: 32, fontWeight: '700', marginBottom: 16 },
  barBg: {
    width: '100%',
    height: 10,
    backgroundColor: '#D3D1C7',
    borderRadius: 5,
    overflow: 'hidden',
    marginBottom: 8,
  },
  barFill: { height: '100%', borderRadius: 5 },
  confidence: { fontSize: 14, color: '#5F5E5A', marginBottom: 12 },
  connStatus: { fontSize: 13, fontWeight: '500' },
  endpoint: { fontSize: 12, color: '#5F5E5A', marginTop: 10, textAlign: 'center' },
  errorText: { fontSize: 12, color: '#A32D2D', marginTop: 6, textAlign: 'center' },
  reconnectBtn: {
    backgroundColor: '#534AB7',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 10,
    marginBottom: 20,
  },
  reconnectText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  details: {
    width: '100%',
    backgroundColor: '#F1EFE8',
    borderRadius: 12,
    padding: 16,
    gap: 6,
  },
  detailRow: { fontSize: 14, color: '#444441' },
});
