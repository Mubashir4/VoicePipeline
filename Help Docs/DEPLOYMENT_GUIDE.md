# Deployment Guide - Voice Pipeline

## 🚀 Quick Deployment Setup

### Option 1: Copy Credentials File (Easiest)

1. **Copy your credentials file** to the new PC:
   ```
   backend/credentials/google-cloud-key.json
   ```

2. **Create `.env` file** in `backend/` folder:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud-key.json
   PORT=5000
   ```

3. **Install and run**:
   ```bash
   npm run install-all
   npm run dev
   ```

### Option 2: Environment Variable (For Deployment/CI)

1. **Get your JSON credentials** (the file you downloaded)

2. **Create `.env` file** with JSON content:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account","project_id":"YOUR_PROJECT_ID","private_key_id":"YOUR_PRIVATE_KEY_ID","private_key":"-----BEGIN PRIVATE KEY-----\nYOUR_FULL_PRIVATE_KEY_CONTENT_HERE\n-----END PRIVATE KEY-----\n","client_email":"your-service-account@your-project.iam.gserviceaccount.com","client_id":"YOUR_CLIENT_ID","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com","universe_domain":"googleapis.com"}
   PORT=5000
   ```

### Option 3: Individual Fields (Alternative)

Create `.env` file with individual credential fields:
```bash
GOOGLE_CLIENT_EMAIL=cloud-speech-to-text-admin@t-osprey-436607-m5.iam.gserviceaccount.com
GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
GOOGLE_PROJECT_ID=t-osprey-436607-m5
PORT=5000
```

### Option 4: Google Cloud CLI (For Development)

1. **Install Google Cloud CLI** on the new PC
2. **Authenticate**:
   ```bash
   gcloud auth application-default login
   ```
3. **Create simple `.env`**:
   ```bash
   PORT=5000
   ```

## 📁 File Structure for Deployment

```
VoicePipline/
├── backend/
│   ├── credentials/
│   │   └── google-cloud-key.json    # Your credentials file
│   ├── .env                         # Environment configuration
│   ├── server.js
│   └── package.json
├── frontend/
│   ├── src/
│   └── package.json
└── package.json
```

## 🔧 Troubleshooting

### Error: "Could not load the default credentials"

**Solution 1: Check file path**
```bash
# Make sure the file exists
ls backend/credentials/google-cloud-key.json

# Check .env file
cat backend/.env
```

**Solution 2: Use absolute path**
```bash
# In .env file, use absolute path:
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/your/VoicePipeline/backend/credentials/google-cloud-key.json
```

**Solution 3: Use JSON content method**
```bash
# Copy your entire JSON file content into .env:
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
```

### Error: "Permission denied"

**Windows:**
```bash
# Run as administrator or check file permissions
icacls backend/credentials/google-cloud-key.json
```

**Linux/Mac:**
```bash
chmod 600 backend/credentials/google-cloud-key.json
```

### Error: "Module not found"

```bash
# Reinstall dependencies
cd backend && npm install
cd ../frontend && npm install
```

## 🌐 Production Deployment

### Environment Variables for Production

```bash
# Production .env
NODE_ENV=production
PORT=5000
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}

# Optional: Enable detailed logging
DEBUG_SPEECH=true
```

### Docker Deployment (Optional)

Create `Dockerfile` in backend:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5000
CMD ["npm", "start"]
```

### Vercel/Netlify Deployment

1. **Backend**: Deploy to Railway, Render, or Heroku
2. **Frontend**: Deploy to Vercel or Netlify
3. **Environment Variables**: Set in your hosting platform

## 📋 Pre-deployment Checklist

- [ ] Google Cloud credentials file copied
- [ ] `.env` file created with correct paths
- [ ] Dependencies installed (`npm run install-all`)
- [ ] Credentials tested (`npm run dev`)
- [ ] Firewall allows port 5000
- [ ] CORS settings updated for production domain

## 🆘 Quick Fix Commands

```bash
# Test credentials
cd backend && node -e "require('dotenv').config(); const speech = require('@google-cloud/speech'); new speech.SpeechClient(); console.log('✅ Credentials working!');"

# Reset and reinstall
rm -rf node_modules package-lock.json
rm -rf backend/node_modules backend/package-lock.json  
rm -rf frontend/node_modules frontend/package-lock.json
npm run install-all

# Check environment
cd backend && node -e "require('dotenv').config(); console.log('GOOGLE_APPLICATION_CREDENTIALS:', process.env.GOOGLE_APPLICATION_CREDENTIALS);"
```

## 💡 Tips

1. **Keep credentials secure** - Never commit `.env` files to git
2. **Use environment variables** in production
3. **Test locally first** before deploying
4. **Check Google Cloud quotas** if you get API errors
5. **Monitor usage** in Google Cloud Console
