import React, { useState } from "react";
import WelcomeScreen from "./components/WelcomeScreen";
import Chat from "./components/Chat";
import Sidebar from "./components/Sidebar";
import "./App.css";

const App = () => {
  const [showWelcome, setShowWelcome] = useState(true); // Pantalla de bienvenida
  const [isSidebarVisible, setSidebarVisible] = useState(true); // Visibilidad de la barra lateral

  const handleSend = () => {
    setShowWelcome(false); // Oculta la bienvenida
  };

  const toggleSidebar = () => {
    setSidebarVisible(!isSidebarVisible); // Alterna la visibilidad de la barra lateral
  };

  return (
    <div className={`app ${showWelcome ? "welcome-active" : "chat-active"}`}>
      {showWelcome && <WelcomeScreen />}
      {isSidebarVisible && <Sidebar onToggleSidebar={toggleSidebar} />}
      <Chat onSend={handleSend} />
    </div>
  );
};

export default App;
