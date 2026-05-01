import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity, Switch } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function SettingsScreen({ ip, onIpChange }) {
  const [inputIp, setInputIp]         = useState(ip);
  const [notifEnabled, setNotifEnabled] = useState(true);

  const save = async () => {
    await AsyncStorage.setItem('esp32_ip', inputIp);
    onIpChange(inputIp);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.section}>ESP32 Connection</Text>
      <Text style={styles.hint}>
        Find the IP in Serial Monitor after flashing.{'\n'}
        Example: 192.168.1.42
      </Text>
      <TextInput
        style={styles.input}
        value={inputIp}
        onChangeText={setInputIp}
        placeholder="192.168.1.xx"
        keyboardType="numeric"
        autoCorrect={false}
      />
      <TouchableOpacity style={styles.saveBtn} onPress={save}>
        <Text style={styles.saveTxt}>Save & Connect</Text>
      </TouchableOpacity>

      <Text style={styles.section}>Notifications</Text>
      <View style={styles.row}>
        <Text style={styles.rowLabel}>Alert when dusty</Text>
        <Switch value={notifEnabled} onValueChange={setNotifEnabled}
                trackColor={{ true: '#534AB7' }} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24 },
  section:   { fontSize: 13, fontWeight: '600', color: '#888780',
               textTransform: 'uppercase', letterSpacing: 0.8, marginTop: 28, marginBottom: 8 },
  hint:      { fontSize: 13, color: '#5F5E5A', marginBottom: 12, lineHeight: 20 },
  input:     { borderWidth: 1, borderColor: '#D3D1C7', borderRadius: 10,
               padding: 12, fontSize: 16, color: '#2C2C2A', backgroundColor: '#fff' },
  saveBtn:   { backgroundColor: '#534AB7', borderRadius: 10, padding: 14,
               alignItems: 'center', marginTop: 12 },
  saveTxt:   { color: '#fff', fontWeight: '600', fontSize: 16 },
  row:       { flexDirection: 'row', justifyContent: 'space-between',
               alignItems: 'center', paddingVertical: 12 },
  rowLabel:  { fontSize: 16, color: '#2C2C2A' },
});
