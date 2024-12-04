import React, { useState } from 'react';
import './Dropdown.css';

const Dropdown = () => {
  const [isOpen, setIsOpen] = useState(false);
  const courses = [
    'Compiladores 1',
    'Compiladores 2',
    'Teoría de Sistemas',
    'Algoritmos y Complejidad',
    'Inteligencia Artificial',
    'Historia de la Ciencia'
  ];

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="dropdown">
      <button className="dropdown-button" onClick={toggleDropdown}>
        Cursos Universitarios
      </button>
      {isOpen && (
        <ul className="dropdown-menu">
          {courses.map((course, index) => (
            <li key={index} className="dropdown-item">
              {course}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dropdown;
