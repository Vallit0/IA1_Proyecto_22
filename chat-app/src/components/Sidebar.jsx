import React from 'react';

const Sidebar = () => {
  const chats = ['the first chat', 'second chat', 'let’s see', 'test mutation'];

  return (
    <div className="sidebar">
      <h2>Recent Chats</h2>
      <ul>
        {chats.map((chat, index) => (
          <li key={index}>{chat}</li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
