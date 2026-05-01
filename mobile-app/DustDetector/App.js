import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Text } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { useWebSocket }    from './hooks/useWebSocket';
import { useNotifications, requestNotificationPermission } from './hooks/useNotifications';
// To:
import DashboardScreen from './hooks/DashboardScreen';
import HistoryScreen from './hooks/HistoryScreen';
import SettingsScreen from './hooks/SettingsScreen';const Tab = createBottomTabNavigator();

export default function App() {
  const [ip, setIp] = useState('');

  // Load saved IP on startup
  useEffect(() => {
    requestNotificationPermission();
    AsyncStorage.getItem('esp32_ip').then(saved => {
      if (saved) setIp(saved);
    });
  }, []);

  const { status, lastReading, history, connect, disconnect } = useWebSocket(ip);
  useNotifications(lastReading);

  const handleIpChange = (newIp) => {
    disconnect();
    setIp(newIp);
  };

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
            />
          )}
        </Tab.Screen>
      </Tab.Navigator>
    </NavigationContainer>
  );
}
