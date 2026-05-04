import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Notice how we import them from the src/screens folder based on your screenshot
import HomeScreen from './src/screens/HomeScreen';
import CallScreen from './src/screens/CallScreen';

const Stack = createNativeStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="HomeScreen">
        
        {/* Route 1: The Home Screen */}
        <Stack.Screen 
          name="HomeScreen" 
          component={HomeScreen} 
          options={{ headerShown: false }} 
        />
        
        {/* Route 2: The Call Screen (This fixes your error!) */}
        <Stack.Screen 
          name="CallScreen" 
          component={CallScreen} 
          options={{ headerShown: false }} 
        />

      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;