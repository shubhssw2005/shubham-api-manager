import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('upload_errors');
const uploadTime = new Trend('upload_duration');
const presignedUrlTime = new Trend('presigned_url_duration');

export let options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 25 },   // Stay at 25 users
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '3m', target: 50 },   // Stay at 50 users
    { duration: '1m', target: 0 },    // Ramp down
  ],
  
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.05'],
    upload_errors: ['rate<0.05'],
    upload_duration: ['p(95)<2000'],
    presigned_url_duration: ['p(95)<500'],
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const JWT_TOKEN = __ENV.JWT_TOKEN;

// Sample file data for different file types
const sampleFiles = {
  image: {
    name: 'test-image.jpg',
    type: 'image/jpeg',
    size: 1024 * 100, // 100KB
    data: generateRandomData(1024 * 100)
  },
  document: {
    name: 'test-document.pdf',
    type: 'application/pdf',
    size: 1024 * 500, // 500KB
    data: generateRandomData(1024 * 500)
  },
  video: {
    name: 'test-video.mp4',
    type: 'video/mp4',
    size: 1024 * 1024 * 5, // 5MB
    data: generateRandomData(1024 * 1024 * 5)
  }
};

function generateRandomData(size) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < size; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export function setup() {
  if (!JWT_TOKEN) {
    const loginResponse = http.post(`${BASE_URL}/api/auth/login`, {
      email: 'test@example.com',
      password: 'testpassword123'
    });
    
    if (loginResponse.status === 200) {
      const token = JSON.parse(loginResponse.body).accessToken;
      return { token };
    }
  }
  
  return { token: JWT_TOKEN };
}

export default function(data) {
  const headers = {
    'Authorization': `Bearer ${data.token}`,
    'Content-Type': 'application/json',
  };

  // Select random file type
  const fileTypes = Object.keys(sampleFiles);
  const fileType = fileTypes[Math.floor(Math.random() * fileTypes.length)];
  const file = sampleFiles[fileType];

  // Test the complete upload flow
  testCompleteUploadFlow(headers, file);

  sleep(2);
}

function testCompleteUploadFlow(headers, file) {
  // Step 1: Request presigned URL
  const presignedUrlStart = Date.now();
  const presignedResponse = http.post(
    `${BASE_URL}/api/media/presigned-url`,
    JSON.stringify({
      filename: file.name,
      contentType: file.type,
      size: file.size
    }),
    { headers }
  );

  const presignedUrlDuration = Date.now() - presignedUrlStart;
  presignedUrlTime.add(presignedUrlDuration);

  const presignedSuccess = check(presignedResponse, {
    'Presigned URL request status is 200': (r) => r.status === 200,
    'Presigned URL response time < 500ms': (r) => r.timings.duration < 500,
    'Presigned URL response has uploadUrl': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.uploadUrl && body.key;
      } catch {
        return false;
      }
    },
  });

  if (!presignedSuccess) {
    errorRate.add(1);
    return;
  }

  const presignedData = JSON.parse(presignedResponse.body);

  // Step 2: Upload file to S3 using presigned URL
  const uploadStart = Date.now();
  const uploadResponse = http.put(
    presignedData.uploadUrl,
    file.data,
    {
      headers: {
        'Content-Type': file.type,
        'Content-Length': file.size.toString(),
      }
    }
  );

  const uploadDuration = Date.now() - uploadStart;
  uploadTime.add(uploadDuration);

  const uploadSuccess = check(uploadResponse, {
    'S3 upload status is 200': (r) => r.status === 200,
    'S3 upload response time < 2000ms': (r) => r.timings.duration < 2000,
  });

  if (!uploadSuccess) {
    errorRate.add(1);
    return;
  }

  // Step 3: Confirm upload completion (optional)
  const confirmResponse = http.post(
    `${BASE_URL}/api/media/confirm-upload`,
    JSON.stringify({
      key: presignedData.key,
      etag: uploadResponse.headers.ETag
    }),
    { headers }
  );

  const confirmSuccess = check(confirmResponse, {
    'Upload confirmation status is 200': (r) => r.status === 200,
    'Upload confirmation response time < 300ms': (r) => r.timings.duration < 300,
  });

  if (!confirmSuccess) {
    errorRate.add(1);
  }

  // Step 4: Test multipart upload for large files (>5MB)
  if (file.size > 5 * 1024 * 1024) {
    testMultipartUpload(headers, file);
  }
}

function testMultipartUpload(headers, file) {
  // Initiate multipart upload
  const initiateResponse = http.post(
    `${BASE_URL}/api/media/multipart/initiate`,
    JSON.stringify({
      filename: file.name,
      contentType: file.type,
      size: file.size
    }),
    { headers }
  );

  const initiateSuccess = check(initiateResponse, {
    'Multipart initiate status is 200': (r) => r.status === 200,
    'Multipart initiate has uploadId': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.uploadId && body.key;
      } catch {
        return false;
      }
    },
  });

  if (!initiateSuccess) {
    errorRate.add(1);
    return;
  }

  const multipartData = JSON.parse(initiateResponse.body);
  const chunkSize = 5 * 1024 * 1024; // 5MB chunks
  const totalChunks = Math.ceil(file.size / chunkSize);
  const parts = [];

  // Upload parts
  for (let i = 0; i < Math.min(totalChunks, 3); i++) { // Limit to 3 parts for testing
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunkData = file.data.slice(start, end);

    const partResponse = http.put(
      `${BASE_URL}/api/media/multipart/part`,
      chunkData,
      {
        headers: {
          'Content-Type': file.type,
          'x-upload-id': multipartData.uploadId,
          'x-part-number': (i + 1).toString(),
        }
      }
    );

    const partSuccess = check(partResponse, {
      [`Part ${i + 1} upload status is 200`]: (r) => r.status === 200,
    });

    if (partSuccess && partResponse.headers.ETag) {
      parts.push({
        PartNumber: i + 1,
        ETag: partResponse.headers.ETag
      });
    }
  }

  // Complete multipart upload
  if (parts.length > 0) {
    const completeResponse = http.post(
      `${BASE_URL}/api/media/multipart/complete`,
      JSON.stringify({
        uploadId: multipartData.uploadId,
        key: multipartData.key,
        parts: parts
      }),
      { headers }
    );

    check(completeResponse, {
      'Multipart complete status is 200': (r) => r.status === 200,
    });
  }
}

export function teardown(data) {
  console.log('Media upload load test completed');
}