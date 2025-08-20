# Voice Pipeline

Real-time speech recognition with speaker diarization using Google Cloud Speech-to-Text API.

## Prerequisites

- Node.js (v16+)
- Google Cloud project with Speech-to-Text API enabled
- Google Cloud credentials configured

## Setup

### 1. Google Cloud Authentication

Choose one method:

**Option A: Service Account Key**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option B: Default Credentials**
```bash
gcloud auth application-default login
```

### 2. Install Dependencies

```bash
npm install
cd backend && npm install
cd ../frontend && npm install
```

## Development

Start both frontend and backend:
```bash
npm run dev
```

Or start separately:
```bash
# Backend (Terminal 1)
cd backend && npm run dev

# Frontend (Terminal 2)
cd frontend && npm start
```

Access at:
- Frontend: http://localhost:3000
- Backend: http://localhost:5000

## Build

```bash
cd frontend && npm run build
```

## Environment

Copy `backend/env.example` to `backend/.env` and configure as needed.