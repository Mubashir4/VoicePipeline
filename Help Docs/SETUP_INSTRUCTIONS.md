# WhisperLiveKit Setup Instructions

## Repository Setup Complete! 🎉

Your repository has been successfully cleared and set up with WhisperLiveKit - a real-time, fully local speech-to-text system with speaker identification.

## What's Installed

- **WhisperLiveKit**: Real-time speech transcription with state-of-the-art models
- **Python Virtual Environment**: Located in `./venv/`
- **All Dependencies**: FastAPI, Whisper models, audio processing libraries
- **FFmpeg**: Already available on your system

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Start the Server
```bash
# Basic setup with base model
whisperlivekit-server --model base --language en

# Or with custom settings
whisperlivekit-server --model base --language en --host 0.0.0.0 --port 8000
```

### 3. Access the Web Interface
Open your browser and go to: `http://localhost:8000`

## Available Models
- `tiny` - Fastest, least accurate
- `base` - Good balance of speed and accuracy (recommended for testing)
- `small` - Better accuracy
- `medium` - High accuracy
- `large-v3` - Best accuracy (requires more resources)

## Key Features

### Real-time Transcription
- Ultra-low latency with SimulStreaming technology
- Supports multiple languages (see tokenizer.py for full list)
- Voice Activity Detection to reduce processing overhead

### Speaker Diarization (Optional)
```bash
# Install diarization support
pip install whisperlivekit[diarization]

# Run with speaker identification
whisperlivekit-server --model base --language en --diarization
```

### Advanced Options
```bash
# With confidence validation (faster but less punctuation accuracy)
whisperlivekit-server --model base --language en --confidence-validation

# With custom frame threshold (lower = faster, higher = more accurate)
whisperlivekit-server --model base --language en --backend simulstreaming --frame-threshold 25

# With SSL for production
whisperlivekit-server --model base --language en --ssl-certfile cert.pem --ssl-keyfile key.pem
```

## Architecture

The system uses:
- **SimulStreaming**: SOTA 2025 ultra-low latency transcription
- **WhisperStreaming**: SOTA 2023 low latency with LocalAgreement
- **Streaming Sortformer**: Advanced real-time speaker diarization
- **Silero VAD**: Enterprise-grade Voice Activity Detection

## File Structure

```
VoicePipeline/
├── venv/                          # Python virtual environment
├── whisperlivekit/               # Main package
│   ├── audio_processor.py        # Audio processing logic
│   ├── basic_server.py           # FastAPI server
│   ├── diarization/              # Speaker identification
│   ├── simul_whisper/            # Advanced streaming models
│   ├── web/                      # Web interface files
│   └── whisper_streaming_custom/ # Custom streaming backend
├── README.md                     # Original project documentation
├── pyproject.toml               # Package configuration
└── SETUP_INSTRUCTIONS.md       # This file
```

## Troubleshooting

### Server Won't Start
1. Make sure virtual environment is activated: `source venv/bin/activate`
2. Check if port 8000 is available: `lsof -i :8000`
3. Try a different port: `--port 8080`

### Model Download Issues
- Models are automatically downloaded on first use
- Check internet connection
- Models are cached in `~/.cache/huggingface/`

### Audio Issues
- Ensure microphone permissions are granted in browser
- Use HTTPS for production deployments (required for microphone access)
- Check FFmpeg installation: `ffmpeg -version`

## Production Deployment

### Using Docker
```bash
# Build and run with GPU
docker build -t whisperlivekit .
docker run --gpus all -p 8000:8000 whisperlivekit

# CPU only
docker build -f Dockerfile.cpu -t whisperlivekit .
docker run -p 8000:8000 whisperlivekit
```

### Using Production Server
```bash
pip install gunicorn
gunicorn -k uvicorn.workers.UvicornWorker -w 4 whisperlivekit.basic_server:app
```

## Next Steps

1. **Test the Setup**: Start the server and test with your microphone
2. **Customize Settings**: Experiment with different models and languages
3. **Add Diarization**: Install speaker identification for multi-speaker scenarios
4. **Deploy**: Set up for production with SSL and proper hosting

## Support

- **Documentation**: Check the main README.md for detailed information
- **Issues**: Report problems on the GitHub repository
- **Models**: Available models listed in the help: `whisperlivekit-server --help`

---

**Ready to go!** Run `source venv/bin/activate && whisperlivekit-server --model base --language en` to start transcribing!
