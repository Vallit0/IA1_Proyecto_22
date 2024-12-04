import React from "react";
import "./WelcomeScreen.css";

const WelcomeScreen = () => {
  return (
    <div className="welcome-screen">
      <h1 className="welcome-title">Hola Estudiante ECYS</h1>
      <p className="welcome-subtitle">¿En qué puedo ayudarte hoy?</p>
      <div className="welcome-options">
        <div className="option-card">Ayuda con Compi</div>
        <div className="option-card">Explicación de EDD</div>
        <div className="option-card">Consejos del Pensum</div>
        <div className="option-card">IPC1</div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
