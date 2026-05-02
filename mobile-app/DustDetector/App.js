import React, { useState, useEffect, useCallback } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Text } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { useWebSocket } from './hooks/useWebSocket';
import { useNotifications, requestNotificationPermission } from './hooks/useNotifications';
import DashboardScreen from './hooks/DashboardScreen';
import HistoryScreen from './hooks/HistoryScreen';
import SettingsScreen from './hooks/SettingsScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  const [ip, setIp] = useState('');

  // Load saved IP on startup
  useEffect(() => {
    requestNotificationPermission();

    async function loadSavedIp() {
      try {
        const saved = await AsyncStorage.getItem('esp32_ip');
        if (saved) {
          setIp(saved.trim()); // ✅ triggers auto connect
        }
      } catch (error) {
        console.error('Failed to load saved ESP IP:', error);
      }
    }

    loadSavedIp();
  }, []);

  // WebSocket hook
  const {
    status,
    lastReading,
    history,
    connect,
    disconnect,
    lastError,
    connectionUrl,
    defaultPort,
  } = useWebSocket(ip);

  // Notifications
  useNotifications(lastReading);

  // ✅ FIXED — NO manual connect call
  const handleIpChange = useCallback(async (newIp) => {
    const normalizedIp = (newIp ?? '').trim();

    // Disconnect current socket
    disconnect();

    try {
      await AsyncStorage.setItem('esp32_ip', normalizedIp);
      setIp(normalizedIp); // ✅ this alone triggers connection
    } catch (e) {
      console.error('Failed to save IP', e);
    }
  }, [disconnect]);

  return (
    <NavigationContainer>
      <Tab.Navigator screenOptions={{ tabBarActiveTintColor: '#534AB7' }}>

        <Tab.Screen
          name="Dashboard"
          options={{ tabBarIcon: () => <Text>📊</Text> }}
        >
          {() => (
            <DashboardScreen
              lastReading={lastReading}
              status={status}
              onReconnect={connect}
              connectionUrl={connectionUrl}
              lastError={lastError}
              defaultPort={defaultPort}
            />
          )}
        </Tab.Screen>

        <Tab.Screen
          name="History"
          options={{ tabBarIcon: () => <Text>📋</Text> }}
        >
          {() => <HistoryScreen history={history} />}
        </Tab.Screen>

        <Tab.Screen
          name="Settings"
          options={{ tabBarIcon: () => <Text>⚙️</Text> }}
        >
          {() => (
            <SettingsScreen
              ip={ip}
              onIpChange={handleIpChange}
              defaultPort={defaultPort}
            />
          )}
        </Tab.Screen>

      </Tab.Navigator>
    </NavigationContainer>
  );
}
