import React from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';

const DOT = { dusty: '#BA7517', clean: '#0F6E56' };

function HistoryItem({ item }) {
  return (
    <View style={styles.row}>
      <View style={[styles.dot, { backgroundColor: DOT[item.label] }]} />
      <View style={styles.info}>
        <Text style={styles.label}>{item.label.toUpperCase()}</Text>
        <Text style={styles.time}>{new Date(item.timestamp).toLocaleTimeString()}</Text>
      </View>
      <Text style={styles.conf}>{(item.confidence * 100).toFixed(0)}%</Text>
    </View>
  );
}

export default function HistoryScreen({ history }) {
  if (!history.length) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No readings yet</Text>
      </View>
    );
  }

  return (
    <FlatList
      data={history}
      keyExtractor={(_, i) => i.toString()}
      renderItem={({ item }) => <HistoryItem item={item} />}
      contentContainerStyle={{ padding: 16 }}
    />
  );
}

const styles = StyleSheet.create({
  row:       { flexDirection: 'row', alignItems: 'center', padding: 14,
               backgroundColor: '#fff', borderRadius: 10, marginBottom: 8,
               shadowColor: '#000', shadowOpacity: 0.04, shadowRadius: 4, elevation: 1 },
  dot:       { width: 12, height: 12, borderRadius: 6, marginRight: 12 },
  info:      { flex: 1 },
  label:     { fontSize: 15, fontWeight: '600', color: '#2C2C2A' },
  time:      { fontSize: 12, color: '#888780' },
  conf:      { fontSize: 14, fontWeight: '500', color: '#5F5E5A' },
  empty:     { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyText: { fontSize: 16, color: '#888780' },
});
