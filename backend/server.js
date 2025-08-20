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
    console.log('Configuration:', {
      language: config.languageCode || 'en-US',
      model: config.model || 'default',
      autoPunctuation: config.enableAutomaticPunctuation,
      diarization: config.enableSpeakerDiarization
    });
    
    const languageCode = config.languageCode || 'en-US';
    const requestedModel = config.model || 'default';
    
    // Automatic model selection based on language capabilities (matches frontend logic)
    const getOptimalModelForLanguage = (langCode) => {
      const premiumLanguages = [
        'en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 
        'pt-BR', 'ru-RU', 'ja-JP', 'ko-KR'
      ];
      
      const standardLanguages = [
        'es-MX', 'pt-PT', 'zh-CN', 'hi-IN', 'tr-TR', 'nl-NL', 
        'sv-SE', 'da-DK', 'fi-FI', 'nb-NO', 'nn-NO'
      ];
      
      const basicLanguages = [
        'ur-PK', 'ur-IN', 'ar-SA', 'ar-AE'
      ];
      
      if (premiumLanguages.includes(langCode)) {
        return { model: 'latest_long', useEnhanced: true };
      } else if (standardLanguages.includes(langCode)) {
        return { model: 'command_and_search', useEnhanced: false };
      } else if (basicLanguages.includes(langCode)) {
        return { model: 'default', useEnhanced: false };
      } else {
        return { model: 'default', useEnhanced: false };
      }
    };
    
    // Get optimal configuration for the language
    const optimalConfig = getOptimalModelForLanguage(languageCode);
    
    // Use the optimal model unless a specific model was requested and is valid
    let finalModel = requestedModel || optimalConfig.model;
    let useEnhanced = config.useEnhanced !== undefined ? config.useEnhanced : optimalConfig.useEnhanced;
    
    // Validate the requested model supports the language
    if (requestedModel && requestedModel !== optimalConfig.model) {
      console.log(`Requested model ${requestedModel} may not be optimal for ${languageCode}, using: ${finalModel}`);
    }
    
    console.log('Using model:', finalModel, 'Enhanced:', useEnhanced);
    
    // Google Cloud Speech-to-Text v1 streaming configuration
    const speechConfig = {
      encoding: 'WEBM_OPUS',
      sampleRateHertz: 48000,
      languageCode: languageCode,
      enableWordTimeOffsets: true,
    };
    
    // Add model if not default
    if (finalModel !== 'default') {
      speechConfig.model = finalModel;
    }
    
    // Add enhanced features if supported
    if (useEnhanced) {
      speechConfig.useEnhanced = true;
    }
    
    // Add automatic punctuation if requested and supported
    if (config.enableAutomaticPunctuation) {
      speechConfig.enableAutomaticPunctuation = true;
    }
    
    // Add speaker diarization if requested and supported
    if (config.enableSpeakerDiarization) {
      speechConfig.enableSpeakerDiarization = true;
      speechConfig.minSpeakerCount = config.minSpeakerCount || 2;
      speechConfig.maxSpeakerCount = config.maxSpeakerCount || 6;
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
        
        // Provide more specific error messages
        let errorMessage = error.message;
        if (error.message.includes('not supported for language')) {
          errorMessage = `The selected model "${finalModel}" is not supported for ${languageCode}. Please try selecting a different model or use the default model.`;
        } else if (error.message.includes('Invalid recognition')) {
          errorMessage = `Configuration error: ${error.message}. This may be due to unsupported features for the selected language.`;
        } else if (error.message.includes('UNAUTHENTICATED')) {
          errorMessage = 'Authentication failed. Please check your Google Cloud credentials.';
        } else if (error.message.includes('PERMISSION_DENIED')) {
          errorMessage = 'Permission denied. Please check your Google Cloud Speech API permissions.';
        }
        
        socket.emit('error', { error: errorMessage });
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
