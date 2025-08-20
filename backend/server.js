const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const speech = require('@google-cloud/speech');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Initialize Google Cloud Speech client
const speechClient = new speech.SpeechClient();

// Store active recognition streams
const activeStreams = new Map();

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('startRecognition', (config) => {
    console.log('Starting recognition for client:', socket.id);
    console.log('Language:', config.languageCode || 'en-US');
    
    const languageCode = config.languageCode || 'en-US';
    
    // Define model compatibility based on language
    const getModelForLanguage = (langCode) => {
      // Languages that support latest_long model with enhanced features
      const enhancedLanguages = [
        'en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR', 'de-DE', 
        'it-IT', 'pt-BR', 'pt-PT', 'ru-RU', 'ja-JP', 'ko-KR'
      ];
      
      // Languages that need basic model
      const basicModelLanguages = [
        'ur-PK', 'ur-IN', 'ar-SA', 'ar-AE', 'hi-IN', 'tr-TR',
        'nb-NO', 'nn-NO', 'sv-SE', 'da-DK', 'fi-FI', 'nl-NL', 'zh-CN'
      ];
      
      if (enhancedLanguages.includes(langCode)) {
        return { model: 'latest_long', useEnhanced: true };
      } else if (basicModelLanguages.includes(langCode)) {
        return { model: 'latest_short', useEnhanced: false };
      } else {
        // Default fallback
        return { model: 'default', useEnhanced: false };
      }
    };
    
    const modelConfig = getModelForLanguage(languageCode);
    console.log('Using model:', modelConfig.model, 'Enhanced:', modelConfig.useEnhanced);
    
    // Google Cloud Speech-to-Text v1 streaming configuration with diarization
    const speechConfig = {
      encoding: 'WEBM_OPUS',
      sampleRateHertz: 48000,
      languageCode: languageCode,
      enableAutomaticPunctuation: true,
      enableWordTimeOffsets: true,
      // Enable speaker diarization (v1)
      enableSpeakerDiarization: true,
      minSpeakerCount: config.minSpeakerCount || 2,
      maxSpeakerCount: config.maxSpeakerCount || 6,
    };
    
    // Add model and useEnhanced only if not default
    if (modelConfig.model !== 'default') {
      speechConfig.model = modelConfig.model;
    }
    if (modelConfig.useEnhanced) {
      speechConfig.useEnhanced = true;
    }
    
    const request = {
      config: speechConfig,
      interimResults: true,
      singleUtterance: false,
    };

    // Create streaming recognition
    const recognizeStream = speechClient
      .streamingRecognize(request)
      .on('error', (error) => {
        console.error('Speech recognition error:', error);
        socket.emit('error', { error: error.message });
      })
      .on('data', (data) => {
        if (data.results[0]) {
          const result = data.results[0];
          const transcript = result.alternatives[0].transcript;
          const isFinal = result.isFinal;
          
          // Extract speaker information if available
          let speakerInfo = null;
          if (result.alternatives[0].words && result.alternatives[0].words.length > 0) {
            const words = result.alternatives[0].words;
            // Group words by speaker
            const speakerSegments = {};
            
            words.forEach(word => {
              const speakerTag = word.speakerTag || 0;
              if (!speakerSegments[speakerTag]) {
                speakerSegments[speakerTag] = [];
              }
              speakerSegments[speakerTag].push(word.word);
            });
            
            speakerInfo = Object.keys(speakerSegments).map(speakerTag => ({
              speakerTag: parseInt(speakerTag),
              text: speakerSegments[speakerTag].join(' ')
            }));
          }

          const response = {
            transcript,
            isFinal,
            confidence: result.alternatives[0].confidence,
            speakerInfo,
            timestamp: Date.now()
          };

          socket.emit('speechData', response);
          
          if (isFinal) {
            console.log('Final result:', transcript);
          }
        }
      });

    // Store the stream for this client
    activeStreams.set(socket.id, recognizeStream);
  });

  socket.on('audioData', (audioData) => {
    const recognizeStream = activeStreams.get(socket.id);
    if (recognizeStream && !recognizeStream.destroyed) {
      // Convert base64 to buffer if needed
      const audioBuffer = Buffer.isBuffer(audioData) ? audioData : Buffer.from(audioData, 'base64');
      recognizeStream.write(audioBuffer);
    }
  });

  socket.on('stopRecognition', () => {
    console.log('Stopping recognition for client:', socket.id);
    const recognizeStream = activeStreams.get(socket.id);
    if (recognizeStream && !recognizeStream.destroyed) {
      recognizeStream.end();
    }
    activeStreams.delete(socket.id);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
    const recognizeStream = activeStreams.get(socket.id);
    if (recognizeStream && !recognizeStream.destroyed) {
      recognizeStream.end();
    }
    activeStreams.delete(socket.id);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('Make sure you have set up Google Cloud credentials:');
  console.log('- Set GOOGLE_APPLICATION_CREDENTIALS environment variable');
  console.log('- Or run: gcloud auth application-default login');
});
