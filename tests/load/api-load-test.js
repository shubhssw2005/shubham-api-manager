import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 500 },   // Stay at 500 users
    { duration: '2m', target: 1000 },  // Ramp up to 1000 users
    { duration: '5m', target: 1000 },  // Stay at 1000 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  
  thresholds: {
    http_req_duration: ['p(95)<200', 'p(99)<500'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
    response_time: ['p(95)<200'],
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const JWT_TOKEN = __ENV.JWT_TOKEN;

export function setup() {
  // Authenticate and get token if not provided
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

  // Test scenarios with different weights
  const scenarios = [
    { weight: 40, test: testGetPosts },
    { weight: 20, test: testGetPost },
    { weight: 15, test: testCreatePost },
    { weight: 10, test: testUpdatePost },
    { weight: 10, test: testGetMedia },
    { weight: 5, test: testDeletePost },
  ];

  // Select scenario based on weight
  const random = Math.random() * 100;
  let cumulative = 0;
  
  for (const scenario of scenarios) {
    cumulative += scenario.weight;
    if (random <= cumulative) {
      scenario.test(headers);
      break;
    }
  }

  sleep(1);
}

function testGetPosts(headers) {
  const response = http.get(`${BASE_URL}/api/posts?limit=20&page=1`, { headers });
  
  const success = check(response, {
    'GET /api/posts status is 200': (r) => r.status === 200,
    'GET /api/posts response time < 200ms': (r) => r.timings.duration < 200,
    'GET /api/posts has posts array': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.data);
      } catch {
        return false;
      }
    },
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

function testGetPost(headers) {
  // Use a known post ID or create one first
  const postId = '507f1f77bcf86cd799439011'; // Example ObjectId
  const response = http.get(`${BASE_URL}/api/posts/${postId}`, { headers });
  
  const success = check(response, {
    'GET /api/posts/:id status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'GET /api/posts/:id response time < 100ms': (r) => r.timings.duration < 100,
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

function testCreatePost(headers) {
  const payload = {
    title: `Load Test Post ${Date.now()}`,
    content: 'This is a test post created during load testing.',
    status: 'published',
    tags: ['load-test', 'performance'],
  };

  const response = http.post(`${BASE_URL}/api/posts`, JSON.stringify(payload), { headers });
  
  const success = check(response, {
    'POST /api/posts status is 201': (r) => r.status === 201,
    'POST /api/posts response time < 300ms': (r) => r.timings.duration < 300,
    'POST /api/posts returns created post': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.title === payload.title;
      } catch {
        return false;
      }
    },
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

function testUpdatePost(headers) {
  const postId = '507f1f77bcf86cd799439011'; // Example ObjectId
  const payload = {
    title: `Updated Post ${Date.now()}`,
    content: 'This post has been updated during load testing.',
  };

  const response = http.put(`${BASE_URL}/api/posts/${postId}`, JSON.stringify(payload), { headers });
  
  const success = check(response, {
    'PUT /api/posts/:id status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'PUT /api/posts/:id response time < 250ms': (r) => r.timings.duration < 250,
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

function testGetMedia(headers) {
  const response = http.get(`${BASE_URL}/api/media?limit=10&page=1`, { headers });
  
  const success = check(response, {
    'GET /api/media status is 200': (r) => r.status === 200,
    'GET /api/media response time < 200ms': (r) => r.timings.duration < 200,
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

function testDeletePost(headers) {
  const postId = '507f1f77bcf86cd799439011'; // Example ObjectId
  const response = http.del(`${BASE_URL}/api/posts/${postId}`, null, { headers });
  
  const success = check(response, {
    'DELETE /api/posts/:id status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'DELETE /api/posts/:id response time < 200ms': (r) => r.timings.duration < 200,
  });

  if (!success) {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

export function teardown(data) {
  // Cleanup if needed
  console.log('Load test completed');
}