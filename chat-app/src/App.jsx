import React, { useState } from "react";
import WelcomeScreen from "./components/WelcomeScreen";
import Chat from "./components/Chat";
import "./App.css";

const App = () => {
  const [showWelcome, setShowWelcome] = useState(true); // Estado para controlar la pantalla de bienvenida

  const handleSend = () => {
    setShowWelcome(false); // Oculta la bienvenida al enviar un mensaje
  };

  return (
    <div className={`app ${showWelcome ? "welcome-active" : "chat-active"}`}>
      {showWelcome && <WelcomeScreen />}
      <Chat onSend={handleSend} />
    </div>
  );
};

export default App;
