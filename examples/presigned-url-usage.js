/**
 * Example usage of the Presigned URL Service for Direct S3 Uploads
 * 
 * This example demonstrates how to use the presigned URL service for:
 * 1. Single file uploads (small files)
 * 2. Multipart uploads (large files)
 * 3. Resumable uploads
 */

import fetch from 'node-fetch';

// Configuration
const API_BASE_URL = 'http://localhost:3005';
const JWT_TOKEN = 'your-jwt-token-here';

/**
 * Example 1: Single File Upload (Small Files < 100MB)
 */
async function singleFileUploadExample() {
  console.log('🔄 Single File Upload Example');
  
  try {
    // Step 1: Request presigned URL
    const presignedResponse = await fetch(`${API_BASE_URL}/api/media/presigned-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        originalName: 'profile-picture.jpg',
        contentType: 'image/jpeg',
        size: 2 * 1024 * 1024, // 2MB
        metadata: {
          description: 'User profile picture',
          category: 'profile'
        }
      })
    });

    const presignedData = await presignedResponse.json();
    console.log('✅ Presigned URL generated:', presignedData.data.uploadUrl);

    // Step 2: Upload file directly to S3
    const fileBuffer = Buffer.from('fake-image-data'); // In real usage, this would be your file data
    
    const uploadResponse = await fetch(presignedData.data.uploadUrl, {
      method: 'PUT',
      headers: {
        'Content-Type': 'image/jpeg'
      },
      body: fileBuffer
    });

    if (uploadResponse.ok) {
      console.log('✅ File uploaded successfully to S3');
      console.log('📍 S3 Key:', presignedData.data.s3Key);
    } else {
      console.error('❌ Upload failed:', uploadResponse.statusText);
    }

  } catch (error) {
    console.error('❌ Single file upload error:', error);
  }
}

/**
 * Example 2: Multipart Upload (Large Files > 100MB)
 */
async function multipartUploadExample() {
  console.log('🔄 Multipart Upload Example');
  
  try {
    // Step 1: Request presigned URLs for multipart upload
    const presignedResponse = await fetch(`${API_BASE_URL}/api/media/presigned-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        originalName: 'large-video.mp4',
        contentType: 'video/mp4',
        size: 200 * 1024 * 1024, // 200MB
        metadata: {
          description: 'Marketing video',
          category: 'media'
        }
      })
    });

    const presignedData = await presignedResponse.json();
    console.log('✅ Multipart upload URLs generated');
    console.log('📊 Parts count:', presignedData.data.partCount);

    // Step 2: Upload each part
    const uploadedParts = [];
    const fakeVideoData = Buffer.alloc(presignedData.data.partSize, 'fake-video-data');

    for (const part of presignedData.data.partUrls) {
      console.log(`📤 Uploading part ${part.partNumber}...`);
      
      const partResponse = await fetch(part.uploadUrl, {
        method: 'PUT',
        headers: {
          'Content-Type': 'video/mp4'
        },
        body: fakeVideoData.slice(0, part.maxSize)
      });

      if (partResponse.ok) {
        const etag = partResponse.headers.get('etag');
        uploadedParts.push({
          PartNumber: part.partNumber,
          ETag: etag
        });
        console.log(`✅ Part ${part.partNumber} uploaded, ETag: ${etag}`);
      } else {
        console.error(`❌ Part ${part.partNumber} upload failed:`, partResponse.statusText);
        return;
      }
    }

    // Step 3: Complete multipart upload
    const completeResponse = await fetch(`${API_BASE_URL}/api/media/complete-multipart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        s3Key: presignedData.data.s3Key,
        uploadId: presignedData.data.uploadId,
        parts: uploadedParts
      })
    });

    const completeData = await completeResponse.json();
    if (completeData.success) {
      console.log('✅ Multipart upload completed successfully');
      console.log('📍 Final location:', completeData.data.location);
    } else {
      console.error('❌ Failed to complete multipart upload:', completeData);
    }

  } catch (error) {
    console.error('❌ Multipart upload error:', error);
  }
}

/**
 * Example 3: Resumable Upload Session
 */
async function resumableUploadExample() {
  console.log('🔄 Resumable Upload Example');
  
  try {
    // Step 1: Create resumable session
    const sessionResponse = await fetch(`${API_BASE_URL}/api/media/resumable-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        originalName: 'presentation.pptx',
        contentType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        size: 50 * 1024 * 1024, // 50MB
        metadata: {
          description: 'Company presentation',
          category: 'documents'
        }
      })
    });

    const sessionData = await sessionResponse.json();
    console.log('✅ Resumable session created');
    console.log('🆔 Session ID:', sessionData.data.sessionId);
    console.log('📊 Total chunks:', sessionData.data.totalChunks);
    console.log('📏 Chunk size:', sessionData.data.chunkSize);

    // In a real implementation, you would:
    // 1. Split the file into chunks
    // 2. Upload chunks one by one
    // 3. Handle failures and resume from the last successful chunk
    // 4. Track progress and provide user feedback

    console.log('📤 Upload URL:', sessionData.data.uploadUrl);
    console.log('📊 Status URL:', sessionData.data.statusUrl);

  } catch (error) {
    console.error('❌ Resumable upload error:', error);
  }
}

/**
 * Example 4: Error Handling
 */
async function errorHandlingExample() {
  console.log('🔄 Error Handling Example');
  
  try {
    // Attempt to upload an unsupported file type
    const response = await fetch(`${API_BASE_URL}/api/media/presigned-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        originalName: 'malware.exe',
        contentType: 'application/x-executable',
        size: 1024 * 1024
      })
    });

    const data = await response.json();
    
    if (!response.ok) {
      console.log('❌ Expected error received:', data.message);
      console.log('🔍 Error type:', data.error);
    }

  } catch (error) {
    console.error('❌ Unexpected error:', error);
  }
}

/**
 * Example 5: Abort Multipart Upload
 */
async function abortMultipartExample() {
  console.log('🔄 Abort Multipart Upload Example');
  
  try {
    // Start a multipart upload
    const presignedResponse = await fetch(`${API_BASE_URL}/api/media/presigned-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        originalName: 'cancelled-upload.mp4',
        contentType: 'video/mp4',
        size: 150 * 1024 * 1024 // 150MB
      })
    });

    const presignedData = await presignedResponse.json();
    console.log('✅ Multipart upload started');

    // Simulate cancellation - abort the upload
    const abortResponse = await fetch(`${API_BASE_URL}/api/media/abort-multipart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${JWT_TOKEN}`
      },
      body: JSON.stringify({
        s3Key: presignedData.data.s3Key,
        uploadId: presignedData.data.uploadId
      })
    });

    const abortData = await abortResponse.json();
    if (abortData.success) {
      console.log('✅ Multipart upload aborted successfully');
    }

  } catch (error) {
    console.error('❌ Abort multipart error:', error);
  }
}

/**
 * Run all examples
 */
async function runExamples() {
  console.log('🚀 Starting Presigned URL Service Examples\n');
  
  await singleFileUploadExample();
  console.log('\n' + '='.repeat(50) + '\n');
  
  await multipartUploadExample();
  console.log('\n' + '='.repeat(50) + '\n');
  
  await resumableUploadExample();
  console.log('\n' + '='.repeat(50) + '\n');
  
  await errorHandlingExample();
  console.log('\n' + '='.repeat(50) + '\n');
  
  await abortMultipartExample();
  
  console.log('\n✅ All examples completed');
}

// Run examples if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExamples().catch(console.error);
}

export {
  singleFileUploadExample,
  multipartUploadExample,
  resumableUploadExample,
  errorHandlingExample,
  abortMultipartExample
};