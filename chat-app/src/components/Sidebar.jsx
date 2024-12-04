import React from "react";
import "./Sidebar.css";
import Dropdown from "./Dropdown";

const Sidebar = ({ onToggleSidebar }) => {
  const chats = ["the first chat", "second chat", "let’s see", "test mutation"];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        
        <div className="sidebar-title">
          <img
            src="https://img.icons8.com/fluency/48/000000/lightning-bolt.png"
            alt="Logo"
            className="sidebar-logo"
          />
          <h2>Spike 1.0</h2>
          <button className="sidebar-toggle-button" onClick={onToggleSidebar}>
          ✖
        </button>
        </div>
      </div>
      
      <Dropdown />
      <h3>Chats</h3>
      <ul>
        {chats.map((chat, index) => (
          <li key={index}>{chat}</li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
