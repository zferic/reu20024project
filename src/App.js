import React, { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() !== '') {
      setMessages([...messages, { text: inputValue, sender: 'user' }]);
      setTimeout(() => {
        setMessages(prevMessages => [...prevMessages, { text: `You said: ${inputValue}`, sender: 'bot' }]);
      }, 500);
      setInputValue('');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>PROTECT RAG</h1>
      </header>
      <div className="chat-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message here..."
          className="input-bar"
        />
        <button type="submit" className="send-button">Send</button>
      </form>
    </div>
  );
}

export default App;