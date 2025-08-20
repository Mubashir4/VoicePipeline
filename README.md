# Voice Pipeline Test - Google Cloud Speech-to-Text v1 with Speaker Diarization

A small test project demonstrating real-time speech recognition with speaker diarization using Google Cloud Speech-to-Text v1 API.

## Features

- **Real-time streaming**: Live audio processing with interim and final results
- **Speaker diarization**: Identifies and separates different speakers (Speaker 1, 2, 3, etc.)
- **Google Cloud Speech-to-Text v1**: Uses the reliable v1 API with proven diarization capabilities
- **WebSocket communication**: Real-time data streaming between frontend and backend
- **Visual speaker separation**: Color-coded UI for different speakers
- **Configurable speaker count**: Set min/max speaker count for optimal diarization

## Architecture

```
Frontend (React + TypeScript)
    ↓ WebSocket + Audio Stream
Backend (Node.js + Express + Socket.io)
    ↓ gRPC Streaming
Google Cloud Speech-to-Text v1 API
```

## Prerequisites

1. **Node.js** (v16 or higher)
2. **Google Cloud Project** with Speech-to-Text API enabled
3. **Google Cloud credentials** configured

## Setup Instructions

### 1. Google Cloud Setup

1. Create a Google Cloud project or use an existing one
2. Enable the Speech-to-Text API:
   ```bash
   gcloud services enable speech.googleapis.com
   ```
3. Create a service account and download the JSON key file
4. Set up authentication (choose one method):
   
   **Method A: Environment Variable**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```
   
   **Method B: Default Application Credentials**
   ```bash
   gcloud auth application-default login
   ```

### 2. Install Dependencies

```bash
# Install root dependencies
npm run install-all

# Or install manually:
npm install
cd backend && npm install
cd ../frontend && npm install
```

### 3. Start the Application

```bash
# Start both frontend and backend
npm run dev
```

Or start them separately:

```bash
# Terminal 1 - Backend
cd backend
npm run dev

# Terminal 2 - Frontend  
cd frontend
npm start
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend: http://localhost:5000

## Usage

1. **Configure speakers**: Set the minimum and maximum number of expected speakers
2. **Start recording**: Click the microphone button to begin
3. **Speak**: The system will show:
   - Interim results (yellow, updating in real-time)
   - Final results (green, locked transcriptions)
   - Speaker tags (color-coded by speaker)
4. **Stop recording**: Click the stop button to end the session

## Configuration Options

### Backend Configuration (server.js)

```javascript
const request = {
  config: {
    encoding: 'WEBM_OPUS',
    sampleRateHertz: 48000,
    languageCode: 'en-US',
    enableAutomaticPunctuation: true,
    enableWordTimeOffsets: true,
    // Speaker diarization settings
    enableSpeakerDiarization: true,
    minSpeakerCount: 2,  // Configurable via frontend
    maxSpeakerCount: 6,  // Configurable via frontend
    model: 'latest_long',
    useEnhanced: true,
  },
  interimResults: true,
  singleUtterance: false,
};
```

### Frontend Configuration

- **Min/Max Speakers**: Adjust based on your use case
- **Audio Settings**: Currently optimized for web audio (48kHz, mono)
- **WebSocket URL**: Update if deploying to different hosts

## Project Structure

```
VoicePipline/
├── package.json                 # Root package with scripts
├── README.md                   # This file
├── backend/
│   ├── package.json           # Backend dependencies
│   └── server.js              # Express server with Socket.io and Google Cloud Speech
└── frontend/
    ├── package.json           # React app dependencies
    ├── src/
    │   ├── App.tsx           # Main app component
    │   ├── App.css           # App styles
    │   └── components/
    │       ├── SpeechRecognition.tsx  # Main speech component
    │       └── SpeechRecognition.css  # Component styles
    └── public/               # Static files
```

## Key Implementation Details

### Why Google Cloud Speech-to-Text v1?

- **Proven diarization**: v1 has stable, reliable speaker diarization
- **Streaming support**: Full support for real-time streaming with diarization
- **Avoid v2 issues**: v2 has reported diarization instability issues

### Audio Processing Flow

1. **Frontend**: Captures microphone audio using MediaRecorder (WebM/Opus)
2. **WebSocket**: Streams audio chunks to backend in real-time
3. **Backend**: Forwards audio to Google Cloud Speech-to-Text v1 streaming API
4. **Processing**: Google returns interim/final results with speaker tags
5. **UI Update**: Frontend displays results with speaker color coding

### Speaker Diarization Features

- **Word-level speaker tags**: Each word gets a speaker identification
- **Real-time processing**: Speaker separation happens during streaming
- **Visual separation**: Different colors for each speaker in the UI
- **Configurable**: Adjust min/max speaker count for better accuracy

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```
   Error: Could not load the default credentials
   ```
   - Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly
   - Or run `gcloud auth application-default login`

2. **Microphone Permission Denied**
   - Allow microphone access in your browser
   - Use HTTPS in production (required for microphone access)

3. **WebSocket Connection Failed**
   - Ensure backend is running on port 5000
   - Check firewall settings

4. **No Speaker Diarization**
   - Ensure you have multiple speakers in the audio
   - Try adjusting min/max speaker count
   - Speak clearly with pauses between speakers

### Audio Quality Tips

- Use a good quality microphone
- Minimize background noise
- Have speakers speak clearly and separately
- Allow small pauses between speakers for better diarization

## Development Notes

- **Real-time processing**: The system processes audio in 100ms chunks
- **Memory management**: Streams are properly cleaned up on disconnect
- **Error handling**: Comprehensive error handling for network and API issues
- **Responsive UI**: Works on desktop and mobile devices

## Next Steps for Production

1. **Security**: Add authentication and rate limiting
2. **Scaling**: Implement connection pooling and load balancing
3. **Storage**: Add transcript storage and retrieval
4. **Analytics**: Track usage and performance metrics
5. **HTTPS**: Deploy with SSL certificates for production use

## License

This is a test project for demonstration purposes.
