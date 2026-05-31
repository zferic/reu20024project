import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaPaperPlane, FaSun, FaMoon, FaBars, FaComment } from 'react-icons/fa';
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
  const API_BASE_URL = process.env.REACT_APP_API_URL || "https://prollm.ece.neu.edu";
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [currentFeedbackIndex, setCurrentFeedbackIndex] = useState(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
  e.preventDefault();
  if (!inputValue.trim()) return;

  const userText = inputValue;
  setInputValue("");

  // Add user message
  setMessages((prev) => [...prev, { text: userText, sender: "user" }]);

  // Prepare bot message placeholder
  const botIndex = messages.length + 1;
  setMessages((prev) => [...prev, { text: "", sender: "bot", showFeedback: false }]);

  setIsLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/api/query_stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: userText })
    });

    if (!response.body) throw new Error("ReadableStream not supported.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let botText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      botText += chunk;

      // Update the bot message live
      setMessages((prev) => {
        const updated = [...prev];
        updated[botIndex] = { text: botText, sender: "bot", showFeedback: true };
        return updated;
      });
    }

  } catch (err) {
    console.error("Streaming error:", err);
    setMessages((prev) => [
      ...prev,
      { text: "Error receiving streamed response.", sender: "bot" }
    ]);
  }

  setIsLoading(false);
};
/*
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim() !== '') {
      const userMessage = { text: inputValue, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setInputValue('');
      setIsLoading(true);

      try {
        const response = await axios.post(`${API_BASE_URL}/api/query`, {
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
*/
  const submitFeedback = async (index, comment) => {
    const message = messages[index];
    try {
      await axios.post(`${API_BASE_URL}/api/feedback`, {
        question: messages[index - 1]?.text || '',
        answer: message.text,
        feedback: 'detailed_feedback',
        comment: comment
      });

      const updatedMessages = [...messages];
      updatedMessages[index] = {
        ...message,
        showFeedback: false,
        feedbackGiven: true,
      };
      setMessages(updatedMessages);
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  };

  const handleCloseModal = () => {
    setShowFeedbackModal(false);
    setFeedbackComment('');
    setCurrentFeedbackIndex(null);
  };

  const handleSubmitFeedback = async () => {
    if (currentFeedbackIndex !== null) {
      await submitFeedback(currentFeedbackIndex, feedbackComment);
      handleCloseModal();
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
                  onClick={() => {
                    setCurrentFeedbackIndex(index);
                    setShowFeedbackModal(true);
                  }}
                  className="feedback-comment"
                  aria-label="Add Feedback"
                >
                  <FaComment /> Feedback
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
      
      {showFeedbackModal && (
        <div className="feedback-modal-overlay">
          <div className="feedback-modal">
            <h3>Provide Feedback</h3>
            <textarea
              className="feedback-textarea"
              placeholder="What did you think of this response? Was it helpful? How could it be improved?"
              value={feedbackComment}
              onChange={(e) => setFeedbackComment(e.target.value)}
            ></textarea>
            <div className="feedback-modal-buttons">
              <button onClick={handleCloseModal} className="cancel-button">Cancel</button>
              <button onClick={handleSubmitFeedback} className="submit-button">Submit Feedback</button>
            </div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your query here..."
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
