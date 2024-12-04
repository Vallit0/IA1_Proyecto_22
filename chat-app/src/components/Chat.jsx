import React, { useState } from "react";
import "./Chat.css";

const Chat = ({ onSend }) => {
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSend = () => {
    if (inputValue.trim() !== "") {
      setMessages([...messages, inputValue]); // Agrega el mensaje al estado
      setInputValue(""); // Limpia el campo de entrada
      onSend(); // Notifica al componente padre
    }
  };

  return (
    <div className="chat">
      {messages.length > 0 && (
        <div className="chat-window">
          <div className="messages">
            {messages.map((msg, index) => (
              <div key={index} className="message">
                {msg}
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="chat-input-container">
        <input
          type="text"
          placeholder="Escribe un mensaje..."
          value={inputValue}
          onChange={handleChange}
          className="chat-input"
        />
        <button onClick={handleSend} className="chat-send-button">
          Enviar
        </button>
      </div>
    </div>
  );
};

export default Chat;
