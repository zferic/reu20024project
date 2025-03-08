import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaThumbsUp, FaThumbsDown, FaPaperPlane, FaSun, FaMoon, FaBars } from 'react-icons/fa';
import { BrowserRouter as Router, Route, Routes, useNavigate, Link } from 'react-router-dom';
import './App.css';
import ReactMarkdown from 'react-markdown';


function SplashScreen() {
  const [fadeOut, setFadeOut] = useState(false);
  const navigate = useNavigate();

  const handleSplashClick = () => {
    setFadeOut(true);
    setTimeout(() => {
      navigate('/dashboard/chat');
    }, 500);
  };

  return (
    <div className={`splash-screen ${fadeOut ? 'fade-out' : ''}`} onClick={handleSplashClick}>
      <img
        src="https://bpb-us-e1.wpmucdn.com/sites.northeastern.edu/dist/e/203/files/2020/05/protect_logo_2022_shortname.png"
        alt="Logo"
        className="splash-logo"
      />
    </div>
  );
}

function Dashboard({ theme, toggleTheme }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const handleDropdownToggle = () => {
    setDropdownOpen((prev) => !prev);
  };

  const dropdownRef = useRef(null);
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target)
      ) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className={`dashboard ${theme}`}>
      <nav className="navbar">
        <div className="nav-left">
          <h2 className="app-title">PROTECT RAG Chat</h2>
        </div>
        <div className="nav-center" ref={dropdownRef}>
          <button className="dropdown-toggle" onClick={handleDropdownToggle} aria-label="Menu">
            <FaBars />
          </button>
          {dropdownOpen && (
            <div className="dropdown-menu">
              <Link to="/dashboard/chat" className="dropdown-link" onClick={() => setDropdownOpen(false)}>
                Chat
              </Link>
              <Link to="/dashboard/previous-chats" className="dropdown-link" onClick={() => setDropdownOpen(false)}>
                Previous Chats
              </Link>
              <Link to="/dashboard/about" className="dropdown-link" onClick={() => setDropdownOpen(false)}>
                About
              </Link>
            </div>
          )}
        </div>
        <div className="nav-right">
          <button onClick={toggleTheme} className="theme-toggle" aria-label="Toggle Theme">
            {theme === 'light' ? <FaMoon /> : <FaSun />}
          </button>
        </div>
      </nav>
      <div className="dashboard-content">
        <Routes>
          <Route path="chat" element={<ChatApp />} />
          <Route path="previous-chats" element={<PreviousChatsPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<ChatApp />} />
        </Routes>
      </div>
    </div>
  );
}

function ChatApp() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim() !== '') {
      const userMessage = { text: inputValue, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setInputValue('');
      setIsLoading(true);

      try {
        const response = await axios.post('http://localhost:8000/query', {
          question: inputValue,
        });

        if (response.data && response.data.answer) {
          const botMessage = { text: response.data.answer, sender: 'bot', showFeedback: true };
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        } else {
          throw new Error('Response data is undefined or invalid');
        }
      } catch (error) {
        console.error('Error fetching response:', error);
        const errorMessage = {
          text: 'Sorry, something went wrong. Please try again later.',
          sender: 'bot',
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleFeedback = async (index, feedback) => {
    const message = messages[index];
    try {
      await axios.post('http://localhost:8000/feedback', {
        question: messages[index - 1]?.text || '',
        answer: message.text,
        feedback: feedback,
      });

      const updatedMessages = [...messages];
      updatedMessages[index] = {
        ...message,
        showFeedback: false,
        feedbackGiven: feedback,
      };
      setMessages(updatedMessages);
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  };

  return (
    <div className="chat-app">
      <header className="App-header">
      </header>
      <div className="chat-container" ref={chatContainerRef}>
      {messages.map((message, index) => (
  <div key={index} className={`message ${message.sender}`}>
    <div className="message-bubble">
      {message.sender === 'bot' ? (
        <ReactMarkdown className="message-text">
          {message.text}
        </ReactMarkdown>
      ) : (
        <div className="message-text">{message.text}</div>
      )}
      {message.sender === 'bot' && message.showFeedback && (
        <div className="feedback-buttons">
          <button
            onClick={() => handleFeedback(index, 'thumbs_up')}
            className="thumbs-up"
            aria-label="Thumbs Up"
          >
            <FaThumbsUp />
          </button>
          <button
            onClick={() => handleFeedback(index, 'thumbs_down')}
            className="thumbs-down"
            aria-label="Thumbs Down"
          >
            <FaThumbsDown />
          </button>
        </div>
      )}
    </div>
  </div>
))}

        {isLoading && (
          <div className="message bot">
            <div className="message-bubble">
              <div className="loading-indicator">
                <span className="dot dot1">.</span>
                <span className="dot dot2">.</span>
                <span className="dot dot3">.</span>
              </div>
            </div>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message here..."
          className="input-bar"
          disabled={isLoading}
        />
        <button type="submit" className="send-button" disabled={!inputValue.trim() || isLoading}>
          <FaPaperPlane />
        </button>
      </form>
    </div>
  );
}

function PreviousChatsPage() {
  return (
    <div className="previous-chats">
      <h2>Previous Chats</h2>
      <p>This is where your previous chats will be displayed.</p>
    </div>
  );
}

function AboutPage() {
  return (
    <div className="about-page">
      <h2>About PROTECT RAG</h2>
      <p>
        PROTECT RAG is a chatbot application designed to assist users with their inquiries surrounding Environmental
        Health Research conducted by the PROTECT (Puerto Rico Testsite for Exploring Contamination Threats). This website
        demonstrates the integration of distributed RAG modeling into a fullstack application and one of our many websites out
        for testing.
      </p>
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    const storedTheme = localStorage.getItem('app-theme');
    if (storedTheme) setTheme(storedTheme);
  }, []);

  const toggleTheme = () => {
    setTheme((prevTheme) => {
      const newTheme = prevTheme === 'light' ? 'dark' : 'light';
      localStorage.setItem('app-theme', newTheme);
      return newTheme;
    });
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={<SplashScreen />} />
        <Route path="/dashboard/*" element={<Dashboard theme={theme} toggleTheme={toggleTheme} />} />
      </Routes>
    </Router>
  );
}

export default App;
