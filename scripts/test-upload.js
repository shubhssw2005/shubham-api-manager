const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const fetch = require('node-fetch');

async function testUploads() {
  try {
    // Create test directory
    const testDir = path.join(__dirname, 'test-files');
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }

    // Create test files
    const imageData = Buffer.from('R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=', 'base64');
    const imagePath = path.join(testDir, 'test.gif');
    fs.writeFileSync(imagePath, imageData);

    const videoData = Buffer.from('MOCK VIDEO CONTENT');
    const videoPath = path.join(testDir, 'test.mp4');
    fs.writeFileSync(videoPath, videoData);

    // 1. Authenticate
    console.log('1. Authenticating...');
    const authResponse = await fetch('http://localhost:3005/api/auth/signin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'admin@example.com',
        password: 'admin123'
      })
    });

    if (!authResponse.ok) {
      throw new Error(`Auth failed: ${await authResponse.text()}`);
    }

    const authData = await authResponse.json();
    const token = authData.token;
    console.log('Authentication successful');

    // 2. Upload image
    console.log('\n2. Uploading image...');
    const imageForm = new FormData();
    imageForm.append('file', fs.createReadStream(imagePath));
    imageForm.append('type', 'image');
    imageForm.append('metadata', JSON.stringify({
      title: 'Test Image',
      description: 'A test image upload',
      tags: ['test']
    }));

    const imageResponse = await fetch('http://localhost:3005/api/media', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        ...imageForm.getHeaders()
      },
      body: imageForm
    });

    if (!imageResponse.ok) {
      throw new Error(`Image upload failed: ${await imageResponse.text()}`);
    }

    const imageResult = await imageResponse.json();
    console.log('Image upload successful:', imageResult);

    // 3. Upload video
    console.log('\n3. Uploading video...');
    const videoForm = new FormData();
    videoForm.append('file', fs.createReadStream(videoPath));
    videoForm.append('type', 'video');
    videoForm.append('metadata', JSON.stringify({
      title: 'Test Video',
      description: 'A test video upload',
      tags: ['test']
    }));

    const videoResponse = await fetch('http://localhost:3005/api/media', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        ...videoForm.getHeaders()
      },
      body: videoForm
    });

    if (!videoResponse.ok) {
      throw new Error(`Video upload failed: ${await videoResponse.text()}`);
    }

    const videoResult = await videoResponse.json();
    console.log('Video upload successful:', videoResult);

    // 4. Create blog post with media relations
    console.log('\n4. Creating blog post...');
    const blogResponse = await fetch('http://localhost:3005/api/content', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        title: 'Test Blog Post',
        content: 'This is a test blog post with media attachments',
        metadata: {
          author: 'Test User',
          tags: ['test', 'sample']
        },
        mediaIds: [imageResult.data.id, videoResult.data.id]
      })
    });

    if (!blogResponse.ok) {
      throw new Error(`Blog post creation failed: ${await blogResponse.text()}`);
    }

    const blogResult = await blogResponse.json();
    console.log('Blog post created successfully:', blogResult);

    // Clean up test files
    fs.rmSync(testDir, { recursive: true, force: true });
    console.log('\nTest completed successfully');

  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

// Run tests
testUploads();
