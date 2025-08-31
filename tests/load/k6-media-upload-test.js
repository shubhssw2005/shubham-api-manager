import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics for media operations
const uploadErrorRate = new Rate('upload_errors');
const uploadLatency = new Trend('upload_latency');
const presignedUrlLatency = new Trend('presigned_url_latency');
const processingLatency = new Trend('processing_latency');
const uploadThroughput = new Counter('uploads_per_second');

export let options = {
  scenarios: {
    // Media upload stress test
    media_upload_stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 10 },   // Warm up
        { duration: '3m', target: 50 },   // Moderate load
        { duration: '5m', target: 100 },  // High load
        { duration: '10m', target: 200 }, // Peak load
        { duration: '2m', target: 0 },    // Cool down
      ],
      tags: { test_type: 'media_stress' }
    },
    
    // Concurrent upload test
    concurrent_uploads: {
      executor: 'constant-vus',
      vus: 50,
      duration: '10m',
      tags: { test_type: 'concurrent' }
    },
    
    // Large file upload test
    large_file_test: {
      executor: 'constant-vus',
      vus: 5,
      duration: '15m',
      tags: { test_type: 'large_files' }
    }
  },
  
  thresholds: {
    'http_req_duration': ['p(95)<2000', 'p(99)<5000'], // More lenient for uploads
    'http_req_failed': ['rate<0.02'], // 98% success rate for uploads
    'upload_errors': ['rate<0.02'],
    'upload_latency': ['p(95)<2000', 'p(99)<5000'],
    'presigned_url_latency': ['p(95)<500', 'p(99)<1000'],
    'checks': ['rate>0.98']
  }
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';
const TENANT_ID = __ENV.TENANT_ID || 'test-tenant';

// File type configurations
const FILE_TYPES = {
  image: {
    mimeTypes: ['image/jpeg', 'image/png', 'image/gif'],
    sizes: [50000, 100000, 500000, 1000000], // 50KB to 1MB
    extensions: ['jpg', 'png', 'gif']
  },
  video: {
    mimeTypes: ['video/mp4', 'video/avi', 'video/mov'],
    sizes: [5000000, 10000000, 50000000, 100000000], // 5MB to 100MB
    extensions: ['mp4', 'avi', 'mov']
  },
  document: {
    mimeTypes: ['application/pdf', 'application/msword', 'text/plain'],
    sizes: [100000, 500000, 2000000, 5000000], // 100KB to 5MB
    extensions: ['pdf', 'doc', 'txt']
  }
};

function generateTestFile(type = 'image') {
  const config = FILE_TYPES[type];
  const mimeType = config.mimeTypes[randomIntBetween(0, config.mimeTypes.length - 1)];
  const size = config.sizes[randomIntBetween(0, config.sizes.length - 1)];
  const extension = config.extensions[randomIntBetween(0, config.extensions.length - 1)];
  
  return {
    name: `test-file-${randomString(8)}.${extension}`,
    type: mimeType,
    size: size,
    content: 'x'.repeat(size) // Simple content for testing
  };
}

function generateLargeTestFile() {
  // Generate files between 50MB and 200MB for large file testing
  const size = randomIntBetween(50000000, 200000000);
  return {
    name: `large-test-file-${randomString(8)}.mp4`,
    type: 'video/mp4',
    size: size,
    content: 'x'.repeat(Math.min(size, 1000000)) // Limit actual content for memory
  };
}

export function setup() {
  // Authenticate if needed
  if (!JWT_TOKEN) {
    const authResponse = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
      email: 'admin@example.com',
      password: 'admin123'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (authResponse.status === 200) {
      const token = JSON.parse(authResponse.body).token;
      return { token };
    }
  }
  
  return { token: JWT_TOKEN };
}

export default function(data) {
  const token = data?.token || JWT_TOKEN;
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
    'X-Tenant-ID': TENANT_ID
  };

  // Determine test scenario based on tags
  const testType = __ENV.K6_SCENARIO_NAME || 'media_upload_stress';
  
  if (testType === 'large_file_test') {
    testLargeFileUpload(headers);
  } else {
    testStandardMediaUpload(headers);
  }
  
  // Think time between uploads
  sleep(randomIntBetween(2, 5));
}

function testStandardMediaUpload(headers) {
  // Generate test file
  const fileTypes = ['image', 'video', 'document'];
  const fileType = fileTypes[randomIntBetween(0, fileTypes.length - 1)];
  const testFile = generateTestFile(fileType);
  
  // Step 1: Request presigned URL
  const presignedStartTime = Date.now();
  const presignedResponse = http.post(`${BASE_URL}/api/media/upload/presigned`, JSON.stringify({
    filename: testFile.name,
    contentType: testFile.type,
    size: testFile.size
  }), { headers });
  
  const presignedSuccess = check(presignedResponse, {
    'Presigned URL status is 200': (r) => r.status === 200,
    'Presigned URL response time < 500ms': (r) => r.timings.duration < 500,
    'Presigned URL contains upload URL': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.uploadUrl !== undefined;
      } catch (e) {
        return false;
      }
    }
  });
  
  if (!presignedSuccess) {
    uploadErrorRate.add(1);
    return;
  }
  
  presignedUrlLatency.add(presignedResponse.timings.duration);
  
  // Step 2: Upload to S3 using presigned URL
  try {
    const presignedData = JSON.parse(presignedResponse.body);
    const uploadStartTime = Date.now();
    
    const uploadResponse = http.put(presignedData.uploadUrl, testFile.content, {
      headers: {
        'Content-Type': testFile.type,
        'Content-Length': testFile.size.toString()
      }
    });
    
    const uploadSuccess = check(uploadResponse, {
      'S3 upload status is 200': (r) => r.status === 200,
      'S3 upload response time < 2000ms': (r) => r.timings.duration < 2000
    });
    
    if (!uploadSuccess) {
      uploadErrorRate.add(1);
      return;
    }
    
    uploadLatency.add(uploadResponse.timings.duration);
    uploadThroughput.add(1);
    
    // Step 3: Confirm upload and check processing status
    const confirmResponse = http.post(`${BASE_URL}/api/media/upload/confirm`, JSON.stringify({
      key: presignedData.key,
      filename: testFile.name
    }), { headers });
    
    check(confirmResponse, {
      'Upload confirmation status is 200': (r) => r.status === 200,
      'Upload confirmation response time < 300ms': (r) => r.timings.duration < 300
    }) || uploadErrorRate.add(1);
    
    // Step 4: Monitor processing status (for images and videos)
    if (fileType === 'image' || fileType === 'video') {
      monitorProcessingStatus(presignedData.key, headers);
    }
    
  } catch (e) {
    console.error('Upload error:', e.message);
    uploadErrorRate.add(1);
  }
}

function testLargeFileUpload(headers) {
  const testFile = generateLargeTestFile();
  
  // Request multipart upload
  const multipartResponse = http.post(`${BASE_URL}/api/media/upload/multipart/initiate`, JSON.stringify({
    filename: testFile.name,
    contentType: testFile.type,
    size: testFile.size
  }), { headers });
  
  const multipartSuccess = check(multipartResponse, {
    'Multipart initiate status is 200': (r) => r.status === 200,
    'Multipart initiate response time < 1000ms': (r) => r.timings.duration < 1000
  });
  
  if (!multipartSuccess) {
    uploadErrorRate.add(1);
    return;
  }
  
  try {
    const multipartData = JSON.parse(multipartResponse.body);
    const chunkSize = 5 * 1024 * 1024; // 5MB chunks
    const totalChunks = Math.ceil(testFile.size / chunkSize);
    const uploadedParts = [];
    
    // Upload chunks
    for (let i = 0; i < Math.min(totalChunks, 5); i++) { // Limit to 5 chunks for testing
      const chunkStart = i * chunkSize;
      const chunkEnd = Math.min(chunkStart + chunkSize, testFile.content.length);
      const chunk = testFile.content.slice(chunkStart, chunkEnd);
      
      const partResponse = http.put(multipartData.uploadUrls[i], chunk, {
        headers: {
          'Content-Type': testFile.type
        }
      });
      
      if (partResponse.status === 200) {
        uploadedParts.push({
          PartNumber: i + 1,
          ETag: partResponse.headers.ETag
        });
      }
    }
    
    // Complete multipart upload
    const completeResponse = http.post(`${BASE_URL}/api/media/upload/multipart/complete`, JSON.stringify({
      uploadId: multipartData.uploadId,
      key: multipartData.key,
      parts: uploadedParts
    }), { headers });
    
    check(completeResponse, {
      'Multipart complete status is 200': (r) => r.status === 200,
      'Multipart complete response time < 2000ms': (r) => r.timings.duration < 2000
    }) || uploadErrorRate.add(1);
    
    uploadThroughput.add(1);
    
  } catch (e) {
    console.error('Multipart upload error:', e.message);
    uploadErrorRate.add(1);
  }
}

function monitorProcessingStatus(key, headers) {
  const maxAttempts = 10;
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    sleep(2); // Wait 2 seconds between checks
    
    const statusResponse = http.get(`${BASE_URL}/api/media/processing/status/${encodeURIComponent(key)}`, { headers });
    
    if (statusResponse.status === 200) {
      try {
        const status = JSON.parse(statusResponse.body);
        
        if (status.processingStatus === 'completed') {
          processingLatency.add((Date.now() - status.uploadTime));
          
          check(statusResponse, {
            'Processing completed successfully': () => true,
            'Thumbnails generated': () => status.thumbnails && status.thumbnails.length > 0
          });
          break;
        } else if (status.processingStatus === 'failed') {
          uploadErrorRate.add(1);
          break;
        }
      } catch (e) {
        // Continue monitoring
      }
    }
    
    attempts++;
  }
}

export function teardown(data) {
  console.log('Media upload load test completed');
}