const bcrypt = require('bcryptjs');
const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { setupTestDatabase, teardownTestDatabase } = require('./test-utils');
process.env.NODE_ENV = 'test';

const BASE_URL = 'http://localhost:3005';

async function testMediaLibrary() {
  try {
        // 0. Initialize database
    console.log('Initializing test database...');
    await setupTestDatabase();
    console.log('Test database initialized');
    
    // Set the test URI globally so it's available to the Next.js API routes
    if (!global.TEST_MONGODB_URI) {
      global.TEST_MONGODB_URI = testDbUri;
    }
    console.log('Test database initialized with URI:', global.TEST_MONGODB_URI);
    
    // Create admin user
    const adminPassword = 'admin123';
    const setupResponse = await fetch('http://localhost:3005/api/auth/setup-admin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'admin@example.com',
        password: adminPassword,
        name: 'Admin User'
      })
    });
    const setupResult = await setupResponse.text();
    if (!setupResponse.ok) {
      console.error('Admin setup failed:', setupResult);
      throw new Error(`Admin setup failed: ${setupResult}`);
    } else {
      console.log('Admin setup successful:', setupResult);
    }
    
    // 1. Create test files
    console.log('\nCreating test files...');
    const testDir = path.join(process.cwd(), 'test-files');
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }

    // Create test image
    const imageBuffer = Buffer.from('R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=', 'base64');
    const imagePath = path.join(testDir, 'test.gif');
    fs.writeFileSync(imagePath, imageBuffer);

    // Create test text file
    const textPath = path.join(testDir, 'test.txt');
    fs.writeFileSync(textPath, 'This is a test document');

    // 2. Sign in
    console.log('\nAuthenticating...');
    const authResponse = await fetch(`${BASE_URL}/api/auth/signin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'admin@example.com',
        password: adminPassword
      })
    });

    const authResult = await authResponse.text();
    if (!authResponse.ok) {
      console.error('Authentication failed:', authResult);
      throw new Error(`Authentication failed: ${authResult}`);
    }

    const { token } = JSON.parse(authResult);
    console.log('Authentication successful');

    // 3. Create a folder
    console.log('\nCreating test folder...');
    const folderResponse = await fetch(`${BASE_URL}/api/media/folders`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: 'Test Folder',
        description: 'Folder for test uploads',
        color: '#ff0000'
      })
    });

    if (!folderResponse.ok) {
      throw new Error(`Folder creation failed: ${await folderResponse.text()}`);
    }

    const { data: folder } = await folderResponse.json();
    console.log('Folder created:', folder.name);

    // 4. Upload files
    console.log('\nUploading test image...');
    const imageForm = new FormData();
    imageForm.append('file', fs.createReadStream(imagePath));
    imageForm.append('type', 'image');
    imageForm.append('metadata', JSON.stringify({
      title: 'Test Image',
      description: 'A test image upload',
      tags: ['test']
    }));
    imageForm.append('folder', folder._id);

    const imageUploadResponse = await fetch(`${BASE_URL}/api/media`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        ...imageForm.getHeaders()
      },
      body: imageForm
    });

    if (!imageUploadResponse.ok) {
      throw new Error(`Image upload failed: ${await imageUploadResponse.text()}`);
    }

    const imageResult = await imageUploadResponse.json();
    console.log('Image uploaded successfully');

    // 5. Upload document
    console.log('\nUploading test document...');
    const docForm = new FormData();
    docForm.append('file', fs.createReadStream(textPath));
    docForm.append('type', 'document');
    docForm.append('metadata', JSON.stringify({
      title: 'Test Document',
      description: 'A test document upload',
      tags: ['test', 'document']
    }));
    docForm.append('folder', folder._id);

    const docUploadResponse = await fetch(`${BASE_URL}/api/media`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        ...docForm.getHeaders()
      },
      body: docForm
    });

    if (!docUploadResponse.ok) {
      throw new Error(`Document upload failed: ${await docUploadResponse.text()}`);
    }

    const docResult = await docUploadResponse.json();
    console.log('Document uploaded successfully');

    // 6. List files
    console.log('\nListing files in test folder...');
    const listResponse = await fetch(
      `${BASE_URL}/api/media?folder=${folder._id}&limit=10`,
      {
        headers: { 'Authorization': `Bearer ${token}` }
      }
    );

    if (!listResponse.ok) {
      throw new Error(`Failed to list files: ${await listResponse.text()}`);
    }

    const { data: { mediaFiles } } = await listResponse.json();
    console.log(`Found ${mediaFiles.length} files in folder:`);
    mediaFiles.forEach(file => {
      console.log(`- ${file.title} (${file.type})`);
    });

    // Clean up
    console.log('\nCleaning up test files...');
    fs.rmSync(testDir, { recursive: true, force: true });
    console.log('Test completed successfully');

  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  } finally {
    // Clean up
    try {
      await teardownTestDatabase();
    } catch (cleanupError) {
      console.error('Error during cleanup:', cleanupError);
    }
  }
}

module.exports = testMediaLibrary;

if (require.main === module) {
  testMediaLibrary();
}
