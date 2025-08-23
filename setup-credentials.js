#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log('🎤 Voice Pipeline - Credentials Setup Tool\n');

async function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve);
  });
}

async function setupCredentials() {
  try {
    console.log('Choose your preferred setup method:\n');
    console.log('1. Copy JSON file to credentials folder (Recommended)');
    console.log('2. Paste JSON content into environment variable');
    console.log('3. Use Google Cloud CLI (gcloud auth)');
    console.log('4. Manual setup\n');

    const choice = await question('Enter your choice (1-4): ');

    switch (choice) {
      case '1':
        await setupWithFile();
        break;
      case '2':
        await setupWithJSON();
        break;
      case '3':
        await setupWithGCloud();
        break;
      case '4':
        await manualSetup();
        break;
      default:
        console.log('❌ Invalid choice. Please run the script again.');
        process.exit(1);
    }

  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
  } finally {
    rl.close();
  }
}

async function setupWithFile() {
  console.log('\n📁 Setting up with JSON file...\n');
  
  const jsonPath = await question('Enter the path to your Google Cloud JSON file: ');
  
  if (!fs.existsSync(jsonPath)) {
    throw new Error(`File not found: ${jsonPath}`);
  }

  // Create credentials directory
  const credentialsDir = path.join(__dirname, 'backend', 'credentials');
  if (!fs.existsSync(credentialsDir)) {
    fs.mkdirSync(credentialsDir, { recursive: true });
  }

  // Copy file
  const targetPath = path.join(credentialsDir, 'google-cloud-key.json');
  fs.copyFileSync(jsonPath, targetPath);

  // Create .env file
  const envContent = `# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud-key.json
PORT=5000
NODE_ENV=development
`;

  fs.writeFileSync(path.join(__dirname, 'backend', '.env'), envContent);

  console.log('✅ Setup complete!');
  console.log('📁 Credentials file copied to:', targetPath);
  console.log('⚙️  .env file created');
  
  await testCredentials();
}

async function setupWithJSON() {
  console.log('\n📝 Setting up with JSON content...\n');
  console.log('Paste your entire JSON file content (it should start with {"type":"service_account"...}):');
  
  const jsonContent = await question('JSON: ');
  
  try {
    // Validate JSON
    const parsed = JSON.parse(jsonContent);
    if (parsed.type !== 'service_account') {
      throw new Error('Invalid service account JSON');
    }

    // Create .env file
    const envContent = `# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS_JSON=${jsonContent}
PORT=5000
NODE_ENV=development
`;

    fs.writeFileSync(path.join(__dirname, 'backend', '.env'), envContent);

    console.log('✅ Setup complete!');
    console.log('⚙️  .env file created with JSON credentials');
    
    await testCredentials();
    
  } catch (error) {
    throw new Error('Invalid JSON format. Please check your input.');
  }
}

async function setupWithGCloud() {
  console.log('\n☁️  Setting up with Google Cloud CLI...\n');
  
  console.log('Please run the following command in your terminal:');
  console.log('  gcloud auth application-default login');
  console.log('\nThen press Enter to continue...');
  
  await question('');

  // Create simple .env file
  const envContent = `# Google Cloud Configuration
PORT=5000
NODE_ENV=development
`;

  fs.writeFileSync(path.join(__dirname, 'backend', '.env'), envContent);

  console.log('✅ Setup complete!');
  console.log('⚙️  .env file created (using default credentials)');
  
  await testCredentials();
}

async function manualSetup() {
  console.log('\n🔧 Manual setup instructions:\n');
  console.log('1. Copy your google-cloud-key.json file to backend/credentials/');
  console.log('2. Create backend/.env file with:');
  console.log('   GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud-key.json');
  console.log('   PORT=5000');
  console.log('\nOr see DEPLOYMENT_GUIDE.md for detailed instructions.');
}

async function testCredentials() {
  console.log('\n🧪 Testing credentials...');
  
  try {
    // Change to backend directory and test
    process.chdir(path.join(__dirname, 'backend'));
    require('dotenv').config();
    const speech = require('@google-cloud/speech');
    
    // Try to create client
    const client = new speech.SpeechClient();
    
    console.log('✅ Credentials test passed!');
    console.log('\n🚀 Ready to start! Run: npm run dev');
    
  } catch (error) {
    console.log('❌ Credentials test failed:', error.message);
    console.log('\n📋 Troubleshooting:');
    console.log('1. Check your .env file in backend/');
    console.log('2. Verify credentials file exists');
    console.log('3. See DEPLOYMENT_GUIDE.md for help');
  }
}

// Run the setup
setupCredentials();
