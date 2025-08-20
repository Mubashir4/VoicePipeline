import React, { useState, useEffect, useRef } from 'react';
import io, { Socket } from 'socket.io-client';
import './SpeechRecognition.css';

interface SpeakerSegment {
  speakerTag: number;
  text: string;
}

interface SpeechData {
  transcript: string;
  isFinal: boolean;
  confidence?: number;
  speakerInfo?: SpeakerSegment[];
  timestamp: number;
}

interface TranscriptEntry {
  id: string;
  transcript: string;
  isFinal: boolean;
  confidence?: number;
  speakerInfo?: SpeakerSegment[];
  timestamp: number;
}

// Language options with Google Cloud Speech-to-Text language codes
const LANGUAGES = [
  { code: 'en-US', name: 'English (US)', flag: '🇺🇸', rtl: false },
  { code: 'en-GB', name: 'English (UK)', flag: '🇬🇧', rtl: false },
  { code: 'nb-NO', name: 'Norwegian (Bokmål)', flag: '🇳🇴', rtl: false },
  { code: 'nn-NO', name: 'Norwegian (Nynorsk)', flag: '🇳🇴', rtl: false },
  { code: 'ur-PK', name: 'Urdu (Pakistan)', flag: '🇵🇰', rtl: true },
  { code: 'ur-IN', name: 'Urdu (India)', flag: '🇮🇳', rtl: true },
  { code: 'ar-SA', name: 'Arabic (Saudi Arabia)', flag: '🇸🇦', rtl: true },
  { code: 'ar-AE', name: 'Arabic (UAE)', flag: '🇦🇪', rtl: true },
  { code: 'es-ES', name: 'Spanish (Spain)', flag: '🇪🇸', rtl: false },
  { code: 'es-MX', name: 'Spanish (Mexico)', flag: '🇲🇽', rtl: false },
  { code: 'fr-FR', name: 'French (France)', flag: '🇫🇷', rtl: false },
  { code: 'de-DE', name: 'German (Germany)', flag: '🇩🇪', rtl: false },
  { code: 'it-IT', name: 'Italian (Italy)', flag: '🇮🇹', rtl: false },
  { code: 'pt-BR', name: 'Portuguese (Brazil)', flag: '🇧🇷', rtl: false },
  { code: 'pt-PT', name: 'Portuguese (Portugal)', flag: '🇵🇹', rtl: false },
  { code: 'ru-RU', name: 'Russian (Russia)', flag: '🇷🇺', rtl: false },
  { code: 'zh-CN', name: 'Chinese (Mandarin)', flag: '🇨🇳', rtl: false },
  { code: 'ja-JP', name: 'Japanese (Japan)', flag: '🇯🇵', rtl: false },
  { code: 'ko-KR', name: 'Korean (South Korea)', flag: '🇰🇷', rtl: false },
  { code: 'hi-IN', name: 'Hindi (India)', flag: '🇮🇳', rtl: false },
  { code: 'tr-TR', name: 'Turkish (Turkey)', flag: '🇹🇷', rtl: false },
  { code: 'nl-NL', name: 'Dutch (Netherlands)', flag: '🇳🇱', rtl: false },
  { code: 'sv-SE', name: 'Swedish (Sweden)', flag: '🇸🇪', rtl: false },
  { code: 'da-DK', name: 'Danish (Denmark)', flag: '🇩🇰', rtl: false },
  { code: 'fi-FI', name: 'Finnish (Finland)', flag: '🇫🇮', rtl: false },
];

const SpeechRecognition: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string>('');
  const [minSpeakers, setMinSpeakers] = useState(2);
  const [maxSpeakers, setMaxSpeakers] = useState(6);
  const [audioLevel, setAudioLevel] = useState(0);
  const [totalMessages, setTotalMessages] = useState(0);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState(() => {
    return localStorage.getItem('speechRecognitionLanguage') || 'en-US';
  });
  const [languageSearchTerm, setLanguageSearchTerm] = useState('');
  const [isLanguageDropdownOpen, setIsLanguageDropdownOpen] = useState(false);
  
  const socketRef = useRef<Socket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionStartRef = useRef<number>(0);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Speaker colors for UI
  const speakerColors = [
    '#667eea', '#764ba2', '#f093fb', '#f5576c', 
    '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
  ];

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Get current language info
  const currentLanguage = LANGUAGES.find(lang => lang.code === selectedLanguage) || LANGUAGES[0];
  
  // Get model info for current language
  const getModelInfo = (langCode: string) => {
    const enhancedLanguages = [
      'en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR', 'de-DE', 
      'it-IT', 'pt-BR', 'pt-PT', 'ru-RU', 'ja-JP', 'ko-KR'
    ];
    
    const basicModelLanguages = [
      'ur-PK', 'ur-IN', 'ar-SA', 'ar-AE', 'hi-IN', 'tr-TR',
      'nb-NO', 'nn-NO', 'sv-SE', 'da-DK', 'fi-FI', 'nl-NL', 'zh-CN'
    ];
    
    if (enhancedLanguages.includes(langCode)) {
      return { model: 'Enhanced', quality: 'high', color: '#48bb78' };
    } else if (basicModelLanguages.includes(langCode)) {
      return { model: 'Standard', quality: 'good', color: '#ed8936' };
    } else {
      return { model: 'Basic', quality: 'standard', color: '#a0aec0' };
    }
  };
  
  const modelInfo = getModelInfo(selectedLanguage);

  // Format time helper
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Handle language change
  const handleLanguageChange = (languageCode: string) => {
    setSelectedLanguage(languageCode);
    localStorage.setItem('speechRecognitionLanguage', languageCode);
    setIsLanguageDropdownOpen(false);
    setLanguageSearchTerm('');
  };

  // Filter languages based on search term
  const filteredLanguages = LANGUAGES.filter(lang => 
    lang.name.toLowerCase().includes(languageSearchTerm.toLowerCase()) ||
    lang.code.toLowerCase().includes(languageSearchTerm.toLowerCase())
  );

  // Format duration helper
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('http://localhost:5000');

    socketRef.current.on('speechData', (data: SpeechData) => {
      const entry: TranscriptEntry = {
        id: `${data.timestamp}-${Math.random()}`,
        ...data
      };

      setTranscripts(prev => {
        // If this is a final result, replace any interim result with same timestamp
        if (data.isFinal) {
          const filtered = prev.filter(t => t.isFinal || Math.abs(t.timestamp - data.timestamp) > 1000);
          setTotalMessages(filtered.length + 1);
          return [...filtered, entry];
        } else {
          // For interim results, replace the last interim result
          const finalResults = prev.filter(t => t.isFinal);
          return [...finalResults, entry];
        }
      });

      // Auto-scroll to bottom when new message arrives
      setTimeout(scrollToBottom, 100);
    });

    socketRef.current.on('error', (errorData: { error: string }) => {
      setError(errorData.error);
      setIsRecording(false);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Session duration timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      sessionStartRef.current = Date.now();
      interval = setInterval(() => {
        setSessionDuration(Math.floor((Date.now() - sessionStartRef.current) / 1000));
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  // Handle clicks outside dropdown to close it
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsLanguageDropdownOpen(false);
        setLanguageSearchTerm('');
      }
    };

    if (isLanguageDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isLanguageDropdownOpen]);

  const startRecording = async () => {
    try {
      setError('');
      setTranscripts([]);
      setTotalMessages(0);
      setSessionDuration(0);

      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 48000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      streamRef.current = stream;

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current = mediaRecorder;

      // Send audio data to backend
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socketRef.current) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const arrayBuffer = reader.result as ArrayBuffer;
            const uint8Array = new Uint8Array(arrayBuffer);
            socketRef.current?.emit('audioData', uint8Array);
          };
          reader.readAsArrayBuffer(event.data);
        }
      };

      // Start recognition on backend
      socketRef.current?.emit('startRecognition', {
        minSpeakerCount: minSpeakers,
        maxSpeakerCount: maxSpeakers,
        languageCode: selectedLanguage
      });

      // Start recording
      mediaRecorder.start(100); // Send data every 100ms
      setIsRecording(true);

    } catch (err) {
      setError(`Failed to start recording: ${err}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    socketRef.current?.emit('stopRecognition');
    setIsRecording(false);
    setAudioLevel(0);
  };

  const clearTranscripts = () => {
    setTranscripts([]);
    setTotalMessages(0);
    setSessionDuration(0);
  };

  const exportTranscript = () => {
    const finalTranscripts = transcripts.filter(t => t.isFinal);
    if (finalTranscripts.length === 0) {
      alert('No final transcripts to export');
      return;
    }

    const content = finalTranscripts
      .map(t => {
        const speakerText = t.speakerInfo && t.speakerInfo.length > 0 
          ? t.speakerInfo.map(s => `Speaker ${s.speakerTag + 1}: ${s.text}`).join(' ')
          : `Unknown Speaker: ${t.transcript}`;
        return `[${formatTime(t.timestamp)}] ${speakerText}`;
      })
      .join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };



  return (
    <div className="speech-recognition">
      {/* Control Panel */}
      <div className="control-panel">
        <h3>Controls</h3>
        
        {/* Language Selection */}
        <div className="language-config">
          <h4>Language Selection</h4>
          <div className="language-selector">
            <div className="custom-dropdown" ref={dropdownRef}>
              <button
                className={`language-dropdown-trigger ${isLanguageDropdownOpen ? 'open' : ''}`}
                onClick={() => setIsLanguageDropdownOpen(!isLanguageDropdownOpen)}
                disabled={isRecording}
              >
                <span className="selected-language">
                  <span className="language-flag">{currentLanguage.flag}</span>
                  <span className="language-name">{currentLanguage.name}</span>
                  <span className="model-indicator" style={{ backgroundColor: modelInfo.color }}>
                    {modelInfo.model}
                  </span>
                  {currentLanguage.rtl && <span className="rtl-indicator">RTL</span>}
                </span>
                <span className="dropdown-arrow">▼</span>
              </button>
              
              {isLanguageDropdownOpen && (
                <div className="language-dropdown-menu">
                  <div className="language-search">
                    <input
                      type="text"
                      placeholder="Search languages..."
                      value={languageSearchTerm}
                      onChange={(e) => setLanguageSearchTerm(e.target.value)}
                      className="language-search-input"
                      autoFocus
                    />
                  </div>
                  <div className="language-options">
                    {filteredLanguages.length > 0 ? (
                      filteredLanguages.map((lang) => {
                        const langModelInfo = getModelInfo(lang.code);
                        return (
                          <button
                            key={lang.code}
                            className={`language-option ${lang.code === selectedLanguage ? 'selected' : ''}`}
                            onClick={() => handleLanguageChange(lang.code)}
                          >
                            <span className="language-flag">{lang.flag}</span>
                            <span className="language-details">
                              <span className="language-name">{lang.name}</span>
                              <span className="language-code">{lang.code}</span>
                            </span>
                            <span className="model-indicator" style={{ backgroundColor: langModelInfo.color }}>
                              {langModelInfo.model}
                            </span>
                            {lang.rtl && <span className="rtl-indicator">RTL</span>}
                          </button>
                        );
                      })
                    ) : (
                      <div className="no-languages-found">
                        No languages found for "{languageSearchTerm}"
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Speaker Configuration */}
        <div className="speaker-config">
          <h4>Speaker Configuration</h4>
          <div className="config-row">
            <div className="config-group">
              <label>Min Speakers</label>
              <input
                type="number"
                min="1"
                max="10"
                value={minSpeakers}
                onChange={(e) => setMinSpeakers(parseInt(e.target.value))}
                disabled={isRecording}
              />
            </div>
            <div className="config-group">
              <label>Max Speakers</label>
              <input
                type="number"
                min="2"
                max="10"
                value={maxSpeakers}
                onChange={(e) => setMaxSpeakers(parseInt(e.target.value))}
                disabled={isRecording}
              />
            </div>
          </div>
        </div>

        {/* Recording Controls */}
        <div className="record-section">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`record-button ${isRecording ? 'recording' : ''}`}
          >
            {isRecording ? (
              <>
                <span>⏹️</span>
                Stop Recording
              </>
            ) : (
              <>
                <span>🎤</span>
                Start Recording
              </>
            )}
          </button>

          {isRecording && (
            <div className="recording-indicator">
              <div className="pulse-dot"></div>
              Recording with speaker diarization...
            </div>
          )}
        </div>

        {/* Audio Level Visualization */}
        {isRecording && (
          <div className="audio-level">
            <h4>Audio Level</h4>
            <div className="audio-bars">
              {Array.from({ length: 20 }, (_, i) => (
                <div
                  key={i}
                  className="audio-bar"
                  style={{
                    height: `${Math.max(2, (audioLevel / 100) * 30 * (1 - i * 0.05))}px`
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Session Statistics */}
        <div className="stats-section">
          <h4>Session Stats</h4>
          <div className="stat-item">
            <span className="stat-label">Duration</span>
            <span className="stat-value">{formatDuration(sessionDuration)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Messages</span>
            <span className="stat-value">{totalMessages}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Speakers</span>
            <span className="stat-value">
              {Math.max(0, ...transcripts.filter(t => t.isFinal && t.speakerInfo).flatMap(t => t.speakerInfo!.map(s => s.speakerTag))) + 1 || 0}
            </span>
          </div>
        </div>
      </div>

      {/* Conversation Panel */}
      <div className="conversation-panel">
        <div className="conversation-header">
          <h3>Live Conversation</h3>
          <div className="conversation-actions">
            <button className="action-button" onClick={clearTranscripts}>
              Clear
            </button>
            <button className="action-button" onClick={exportTranscript}>
              Export
            </button>
          </div>
        </div>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        <div className="messages-container">
          {transcripts.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🎙️</div>
              <h4>Ready to Start</h4>
              <p>Click "Start Recording" to begin real-time speech recognition with speaker diarization</p>
              <div className="empty-state-features">
                <div className="feature-item">
                  <span className="feature-icon">🌍</span>
                  <span>25+ Languages Supported</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">👥</span>
                  <span>Multi-Speaker Detection</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">⚡</span>
                  <span>Real-time Transcription</span>
                </div>
              </div>
            </div>
          ) : (
            <>
              {transcripts.map((entry) => {
                const speakerTag = entry.speakerInfo && entry.speakerInfo.length > 0 
                  ? entry.speakerInfo[0].speakerTag 
                  : 0;
                
                return (
                  <div key={entry.id} className={`message ${entry.isFinal ? 'final' : 'interim'} ${currentLanguage.rtl ? 'rtl' : 'ltr'}`}>
                    <div 
                      className="speaker-avatar"
                      style={{ backgroundColor: speakerColors[speakerTag % speakerColors.length] }}
                    >
                      {speakerTag + 1}
                    </div>
                    <div className="message-content">
                      <div className="message-header">
                        <span className="speaker-name">Speaker {speakerTag + 1}</span>
                        <span className="message-time">{formatTime(entry.timestamp)}</span>
                        <span className={`message-status ${entry.isFinal ? 'status-final' : 'status-interim'}`}>
                          {entry.isFinal ? 'Final' : 'Interim'}
                        </span>
                      </div>
                      <div className="message-text" dir={currentLanguage.rtl ? 'rtl' : 'ltr'}>
                        {entry.speakerInfo && entry.speakerInfo.length > 0 
                          ? entry.speakerInfo.map(s => s.text).join(' ')
                          : entry.transcript}
                      </div>
                      {entry.confidence && entry.isFinal && (
                        <div className="confidence-score">
                          Confidence: {Math.round(entry.confidence * 100)}%
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpeechRecognition;
