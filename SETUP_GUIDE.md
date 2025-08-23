# Quick Setup Guide

## 1. Google Cloud Setup (5 minutes)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Speech-to-Text API:
   - Go to APIs & Services > Library
   - Search for "Speech-to-Text API"
   - Click Enable

4. Create service account:
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Name: "speech-to-text-test"
   - Role: "Speech-to-Text Admin" or "Editor"
   - Create and download JSON key file

5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/downloaded-key.json"
   ```

## 2. Install and Run (2 minutes)

```bash
# Clone/navigate to project directory
cd VoicePipline

# Install all dependencies
npm run install-all

# Start both frontend and backend
npm run dev
```

## 3. Test (1 minute)

1. Open http://localhost:3000
2. Allow microphone permission
3. Click "Start Recording"
4. Have 2+ people speak (or speak in different voices)
5. Watch real-time transcription with speaker separation

## Troubleshooting

**Can't authenticate?**
```bash
gcloud auth application-default login
```

**No microphone?**
- Check browser permissions
- Try different browser
- Use HTTPS for production

**No speakers detected?**
- Ensure 2+ different voices
- Adjust min/max speaker settings
- Speak with clear pauses between speakers

That's it! The system should now show real-time speech-to-text with speaker diarization.
