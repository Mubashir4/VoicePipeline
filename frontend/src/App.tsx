import React from 'react';
import './App.css';
import SpeechRecognition from './components/SpeechRecognition';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🎤</div>
            <div className="header-text">
              <h1>Voice Pipeline</h1>
              <p>Real-time Speech Recognition & Speaker Diarization</p>
            </div>
          </div>
          <div className="tech-badge">
            <span className="badge primary">Google Cloud Speech v1</span>
            <span className="badge secondary">WebSocket Streaming</span>
            <span className="badge accent">25+ Languages</span>
          </div>
        </div>
      </header>
      <main className="main-content">
        <SpeechRecognition />
      </main>
    </div>
  );
}

export default App;