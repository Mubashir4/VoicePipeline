# How to Run and Test WhisperLiveKit 🎤

## ✅ Server is Now Running!

Your WhisperLiveKit server is successfully running on **http://localhost:8000**

## Quick Test Steps

### 1. Start the Server (if not already running)
```bash
# Navigate to project directory
cd /Users/muhammadimran/Desktop/Mubashir/VoicePipeline

# Activate virtual environment
source venv/bin/activate

# Start the server
whisperlivekit-server --model base --language en --host 0.0.0.0 --port 8000
```

### 2. Access the Web Interface
1. Open your web browser
2. Navigate to: **http://localhost:8000**
3. You should see the WhisperLiveKit interface

### 3. Test Real-time Transcription
1. **Grant Microphone Permission**: Your browser will ask for microphone access
2. **Click "Start Recording"** or similar button in the interface
3. **Speak into your microphone**: You should see real-time transcription appear
4. **Test different scenarios**:
   - Normal speech
   - Different volumes
   - Pauses and silence
   - Multiple speakers (if diarization is enabled)

## Server Status ✅

- **HTTP Server**: Running on port 8000
- **Web Interface**: Available at http://localhost:8000
- **WebSocket**: Available at ws://localhost:8000/asr
- **Model**: Base model loaded
- **Language**: English (en)
- **VAD**: Voice Activity Detection enabled
- **Dependencies**: All installed correctly

## Testing Different Configurations

### Test with Different Models
```bash
# Tiny model (fastest)
whisperlivekit-server --model tiny --language en

# Small model (better accuracy)
whisperlivekit-server --model small --language en

# Large model (best accuracy, requires more resources)
whisperlivekit-server --model large-v3 --language en
```

### Test with Different Languages
```bash
# Spanish
whisperlivekit-server --model base --language es

# French
whisperlivekit-server --model base --language fr

# German
whisperlivekit-server --model base --language de

# Auto-detect language
whisperlivekit-server --model base --language auto
```

### Test with Speaker Diarization
```bash
# First install diarization support
pip install whisperlivekit[diarization]

# Run with speaker identification
whisperlivekit-server --model base --language en --diarization
```

## Advanced Testing

### 1. Test WebSocket Connection Directly
```bash
# Install wscat for testing (optional)
npm install -g wscat

# Test WebSocket connection
wscat -c ws://localhost:8000/asr
```

### 2. Test with Custom Settings
```bash
# Low latency setup
whisperlivekit-server --model base --language en --backend simulstreaming --frame-threshold 15

# High accuracy setup
whisperlivekit-server --model large-v3 --language en --confidence-validation

# Custom chunk size
whisperlivekit-server --model base --language en --min-chunk-size 0.5
```

### 3. Test API Endpoints
```bash
# Check server health
curl http://localhost:8000/

# View API documentation
curl http://localhost:8000/docs

# Check OpenAPI specification
curl http://localhost:8000/openapi.json
```

## Troubleshooting Tests

### If Web Interface Doesn't Load
1. Check server is running: `curl -I http://localhost:8000`
2. Check for port conflicts: `lsof -i :8000`
3. Try different port: `--port 8080`

### If Microphone Doesn't Work
1. **Browser Permissions**: Ensure microphone access is granted
2. **HTTPS Required**: For production, use HTTPS (browsers block mic on HTTP)
3. **Firewall**: Check if port 8000 is blocked

### If Transcription is Slow
1. **Try Smaller Model**: Use `--model tiny` for testing
2. **Disable VAD**: Use `--no-vad` flag
3. **Adjust Chunk Size**: Use `--min-chunk-size 0.5`

### If Getting Import Errors
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -e .
pip install torchaudio
```

## Performance Testing

### Test Latency
1. Start server with: `whisperlivekit-server --model base --language en --backend simulstreaming`
2. Speak a short phrase and measure time to first transcription
3. Expected latency: < 500ms for base model

### Test Accuracy
1. Read a known text passage
2. Compare transcription accuracy
3. Test with different accents and speaking speeds

### Test Resource Usage
```bash
# Monitor CPU and memory while running
top -p $(pgrep -f whisperlivekit-server)

# Or use htop for better visualization
htop
```

## Production Testing

### Test with SSL (for production)
```bash
# Generate self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
whisperlivekit-server --model base --language en --ssl-certfile cert.pem --ssl-keyfile key.pem

# Access via https://localhost:8000
```

### Test with Docker
```bash
# Build and test Docker image
docker build -t whisperlivekit .
docker run -p 8000:8000 whisperlivekit

# Test the containerized version
curl http://localhost:8000
```

## Expected Results

### ✅ Successful Test Indicators
- Web interface loads without errors
- Microphone permission granted
- Real-time transcription appears as you speak
- Transcription is reasonably accurate
- Server responds within 1-2 seconds
- No error messages in browser console

### ❌ Common Issues to Watch For
- "ModuleNotFoundError" - Missing dependencies
- "Connection refused" - Server not running
- "Microphone blocked" - Browser permissions
- Slow transcription - Model too large or system resources

## Next Steps After Testing

1. **Customize for Your Use Case**: Adjust model size and language
2. **Integrate with Your App**: Use the WebSocket API
3. **Deploy to Production**: Set up proper hosting with SSL
4. **Add Diarization**: For multi-speaker scenarios
5. **Monitor Performance**: Set up logging and metrics

---

🎉 **Your WhisperLiveKit setup is working perfectly!** 

The server is running and ready for real-time speech transcription. Open http://localhost:8000 in your browser and start speaking to see the magic happen!
