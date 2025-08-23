# Server Control Guide 🔧

## How to Stop the Server

### Method 1: If Running in Foreground
```bash
# Press Ctrl+C in the terminal where the server is running
# This is the cleanest way to stop the server
```

### Method 2: Kill Background Process
```bash
# Kill the server process
pkill -f whisperlivekit-server

# Or find and kill by process ID
ps aux | grep whisperlivekit-server
kill <process_id>
```

### Method 3: Kill by Port (if needed)
```bash
# Find what's using port 8000
lsof -ti:8000

# Kill the process using port 8000
kill $(lsof -ti:8000)
```

### Verify Server is Stopped
```bash
curl -I http://localhost:8000 2>/dev/null && echo "Still running" || echo "Stopped ✅"
```

## How to Run the Server

### Basic Setup Commands
```bash
# 1. Navigate to project directory
cd /Users/muhammadimran/Desktop/Mubashir/VoicePipeline

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start server (choose one method below)
```

### Method 1: Foreground Mode (Recommended)
```bash
# Basic setup
whisperlivekit-server --model base --language en

# With custom host and port
whisperlivekit-server --model base --language en --host 0.0.0.0 --port 8000

# To stop: Press Ctrl+C
```

**✅ Advantages:**
- See real-time logs
- Easy to stop with Ctrl+C
- Good for development and testing

### Method 2: Background Mode
```bash
# Run in background
nohup whisperlivekit-server --model base --language en > server.log 2>&1 &

# Check if running
ps aux | grep whisperlivekit-server

# View logs
tail -f server.log
```

**✅ Advantages:**
- Keeps running after closing terminal
- Good for production-like testing

### Method 3: Using Screen/Tmux (Advanced)
```bash
# Using screen
screen -S whisper
whisperlivekit-server --model base --language en
# Press Ctrl+A, then D to detach
# screen -r whisper to reattach

# Using tmux
tmux new-session -d -s whisper 'whisperlivekit-server --model base --language en'
# tmux attach-session -t whisper to attach
```

## Quick Start Commands

### Default Setup
```bash
source venv/bin/activate && whisperlivekit-server --model base --language en
```

### Fast Model (for testing)
```bash
source venv/bin/activate && whisperlivekit-server --model tiny --language en
```

### High Accuracy Model
```bash
source venv/bin/activate && whisperlivekit-server --model large-v3 --language en
```

### With Speaker Diarization
```bash
source venv/bin/activate && whisperlivekit-server --model base --language en --diarization
```

### Auto Language Detection
```bash
source venv/bin/activate && whisperlivekit-server --model base --language auto
```

## Server Status Commands

### Check if Server is Running
```bash
# Method 1: HTTP check
curl -I http://localhost:8000

# Method 2: Process check
ps aux | grep whisperlivekit-server

# Method 3: Port check
lsof -i :8000
```

### Access the Interface
```bash
# Open in browser
open http://localhost:8000

# Or manually navigate to: http://localhost:8000
```

## Troubleshooting

### Server Won't Start
```bash
# Check if port is already in use
lsof -i :8000

# Try different port
whisperlivekit-server --model base --language en --port 8080

# Check virtual environment
which python
pip list | grep whisperlivekit
```

### Server Won't Stop
```bash
# Force kill all WhisperLiveKit processes
pkill -9 -f whisperlivekit

# Kill by port
kill -9 $(lsof -ti:8000)

# Restart terminal if needed
```

### Permission Issues
```bash
# Make sure you're in the right directory
pwd
# Should show: /Users/muhammadimran/Desktop/Mubashir/VoicePipeline

# Activate virtual environment
source venv/bin/activate
```

## Production Deployment

### Using systemd (Linux)
```bash
# Create service file
sudo nano /etc/systemd/system/whisperlivekit.service

# Start/stop service
sudo systemctl start whisperlivekit
sudo systemctl stop whisperlivekit
sudo systemctl status whisperlivekit
```

### Using PM2 (Node.js process manager)
```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
pm2 ecosystem

# Start with PM2
pm2 start ecosystem.config.js
pm2 stop whisperlivekit
pm2 restart whisperlivekit
```

### Using Docker
```bash
# Build image
docker build -t whisperlivekit .

# Run container
docker run -d -p 8000:8000 --name whisper-server whisperlivekit

# Stop container
docker stop whisper-server

# Remove container
docker rm whisper-server
```

## Current Server Status ✅

**Your server is currently RUNNING on:**
- **URL:** http://localhost:8000
- **Model:** base
- **Language:** English (en)
- **Host:** 0.0.0.0 (accessible from other devices)
- **Port:** 8000

### Quick Actions:
```bash
# Stop current server
pkill -f whisperlivekit-server

# Start server again
source venv/bin/activate && whisperlivekit-server --model base --language en

# Check status
curl -I http://localhost:8000
```

---

🎯 **Pro Tips:**
- Use **foreground mode** for development (easy to see logs and stop)
- Use **background mode** for longer testing sessions
- Always **activate the virtual environment** before starting
- **Check the port** if you get connection errors
- **Use Ctrl+C** to gracefully stop foreground processes
