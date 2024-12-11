import React, { useState } from "react";
import "./Chat.css";
import { startChatbot } from '../../backend/chatbot.js';

const Chat = ({ onSend }) => {
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSend = async () => {
    if (inputValue.trim() !== "") {
      try {
        // Agregar el mensaje del usuario
        setMessages(prevMessages => [...prevMessages, inputValue]); 
        console.log('Usuario:', inputValue); 
  
        // Esperar respuesta del chatbot
        const response = await startChatbot(inputValue);
        if (!response) {
          throw new Error('No se recibió respuesta del chatbot');
        }
  
        // Agregar la respuesta del chatbot
        setMessages(prevMessages => [...prevMessages, response]); 
        console.log('Chatbot:', response); 
      } catch (error) {
        console.error('Error en la función startChatbot:', error);
        setMessages(prevMessages => [...prevMessages, 'Error en la respuesta del chatbot']);
      } finally {
        setInputValue(""); // Limpia el campo de entrada
        onSend(); // Notifica al componente padre
      }
    }
  };
  
  return (
    <div className="chat">
      {messages.length > 0 && (
        <div className="chat-window">
          <div className="messages">
          {messages.map((msg, index) => (
                <div 
                  key={`message-${index}`} 
                  className={index % 2 === 0 ? 'message' : 'message2'}
                >
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
