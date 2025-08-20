# 🚨 QUICK FIX - Credentials Error

## For Your Specific Case

You have this error:
```
Could not load the default credentials
```

**SOLUTION 1: Copy the credentials file (Easiest)**

1. Copy your credentials file `t-osprey-436607-m5-761bdefeaac7.json` to the new PC

2. Create `backend/.env` file with:
```
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud-key.json
PORT=5000
```

3. Create `backend/credentials/` folder and put your JSON file there renamed as `google-cloud-key.json`

**SOLUTION 2: Use JSON content in .env (No file needed)**

Create `backend/.env` file with your full JSON:
```
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account","project_id":"YOUR_PROJECT_ID","private_key_id":"YOUR_PRIVATE_KEY_ID","private_key":"-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_CONTENT_HERE\n-----END PRIVATE KEY-----\n","client_email":"your-service-account@your-project.iam.gserviceaccount.com","client_id":"YOUR_CLIENT_ID","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com","universe_domain":"googleapis.com"}
PORT=5000
```

## Test Your Fix

Run this to test:
```bash
npm run test-credentials
```

If it works, you'll see: ✅ Credentials working!

Then start the app:
```bash
npm run dev
```

## Automated Setup

Or use the setup tool:
```bash
npm run setup-credentials
```

## Still Having Issues?

1. **Check file paths** - Make sure the JSON file is in the right place
2. **Check .env file** - Make sure it's in the `backend/` folder, not the root
3. **Try absolute paths** - Use full Windows path like `C:\Users\...\credentials\google-cloud-key.json`
4. **Permissions** - Make sure the file is readable

The updated server now gives you detailed error messages and multiple authentication options!
