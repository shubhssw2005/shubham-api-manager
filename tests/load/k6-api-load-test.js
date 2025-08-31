import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const throughput = new Counter('requests_per_second');

export let options = {
  scenarios: {
    // Baseline load test
    baseline_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Stay at baseline
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'baseline' }
    },
    
    // Stress test - gradually increase load
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 500 },
        { duration: '10m', target: 1000 },
        { duration: '15m', target: 2000 },
        { duration: '10m', target: 5000 }, // Peak load
        { duration: '5m', target: 0 },
      ],
      tags: { test_type: 'stress' }
    },
    
    // Spike test - sudden load increase
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 100,
      stages: [
        { duration: '1m', target: 100 },
        { duration: '30s', target: 2000 }, // Sudden spike
        { duration: '2m', target: 2000 },
        { duration: '30s', target: 100 },
        { duration: '1m', target: 100 },
      ],
      tags: { test_type: 'spike' }
    },
    
    // Soak test - sustained load
    soak_test: {
      executor: 'constant-vus',
      vus: 200,
      duration: '1h',
      tags: { test_type: 'soak' }
    }
  },
  
  thresholds: {
    // SLO requirements from design
    'http_req_duration': ['p(95)<200', 'p(99)<500'],
    'http_req_failed': ['rate<0.01'], // 99% success rate
    'errors': ['rate<0.01'],
    'api_latency': ['p(95)<200', 'p(99)<500'],
    'checks': ['rate>0.99']
  }
};

// Test configuration
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';
const TENANT_ID = __ENV.TENANT_ID || 'test-tenant';

// Test data generators
function generateTestPost() {
  return {
    title: `Load Test Post ${randomString(10)}`,
    content: `This is a test post content generated during load testing. ${randomString(100)}`,
    status: 'published',
    tags: ['load-test', 'performance'],
    metadata: {
      testRun: __ENV.TEST_RUN_ID || 'default',
      timestamp: Date.now()
    }
  };
}

function generateTestUser() {
  return {
    email: `loadtest-${randomString(8)}@example.com`,
    name: `Load Test User ${randomString(5)}`,
    role: 'user'
  };
}

export function setup() {
  // Authenticate and get JWT token if not provided
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

  // Test scenario weights
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    // 40% - Read operations (GET requests)
    testReadOperations(headers);
  } else if (scenario < 0.7) {
    // 30% - Write operations (POST/PUT requests)
    testWriteOperations(headers);
  } else if (scenario < 0.9) {
    // 20% - Mixed operations
    testMixedOperations(headers);
  } else {
    // 10% - Heavy operations
    testHeavyOperations(headers);
  }
  
  // Random think time between 1-3 seconds
  sleep(randomIntBetween(1, 3));
}

function testReadOperations(headers) {
  const startTime = Date.now();
  
  // Test posts listing
  const postsResponse = http.get(`${BASE_URL}/api/posts?limit=20&page=1`, { headers });
  
  const success = check(postsResponse, {
    'Posts list status is 200': (r) => r.status === 200,
    'Posts list response time < 200ms': (r) => r.timings.duration < 200,
    'Posts list has data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.data);
      } catch (e) {
        return false;
      }
    }
  });
  
  if (!success) errorRate.add(1);
  apiLatency.add(postsResponse.timings.duration);
  throughput.add(1);
  
  // Test single post retrieval if posts exist
  if (postsResponse.status === 200) {
    try {
      const posts = JSON.parse(postsResponse.body).data;
      if (posts.length > 0) {
        const randomPost = posts[randomIntBetween(0, posts.length - 1)];
        const postResponse = http.get(`${BASE_URL}/api/posts/${randomPost.id}`, { headers });
        
        check(postResponse, {
          'Single post status is 200': (r) => r.status === 200,
          'Single post response time < 100ms': (r) => r.timings.duration < 100
        }) || errorRate.add(1);
        
        apiLatency.add(postResponse.timings.duration);
        throughput.add(1);
      }
    } catch (e) {
      errorRate.add(1);
    }
  }
  
  // Test users listing
  const usersResponse = http.get(`${BASE_URL}/api/users?limit=10`, { headers });
  
  check(usersResponse, {
    'Users list status is 200': (r) => r.status === 200,
    'Users list response time < 150ms': (r) => r.timings.duration < 150
  }) || errorRate.add(1);
  
  apiLatency.add(usersResponse.timings.duration);
  throughput.add(1);
}

function testWriteOperations(headers) {
  // Create new post
  const postData = generateTestPost();
  const createResponse = http.post(`${BASE_URL}/api/posts`, JSON.stringify(postData), { headers });
  
  const createSuccess = check(createResponse, {
    'Post creation status is 201': (r) => r.status === 201,
    'Post creation response time < 300ms': (r) => r.timings.duration < 300,
    'Post creation returns ID': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id !== undefined;
      } catch (e) {
        return false;
      }
    }
  });
  
  if (!createSuccess) errorRate.add(1);
  apiLatency.add(createResponse.timings.duration);
  throughput.add(1);
  
  // Update the created post
  if (createResponse.status === 201) {
    try {
      const createdPost = JSON.parse(createResponse.body);
      const updateData = {
        ...postData,
        title: `Updated ${postData.title}`,
        content: `Updated content: ${postData.content}`
      };
      
      const updateResponse = http.put(`${BASE_URL}/api/posts/${createdPost.id}`, JSON.stringify(updateData), { headers });
      
      check(updateResponse, {
        'Post update status is 200': (r) => r.status === 200,
        'Post update response time < 250ms': (r) => r.timings.duration < 250
      }) || errorRate.add(1);
      
      apiLatency.add(updateResponse.timings.duration);
      throughput.add(1);
    } catch (e) {
      errorRate.add(1);
    }
  }
}

function testMixedOperations(headers) {
  // Combination of read and write operations
  testReadOperations(headers);
  sleep(0.5);
  testWriteOperations(headers);
}

function testHeavyOperations(headers) {
  // Test bulk operations and complex queries
  const bulkData = {
    posts: Array.from({ length: 5 }, () => generateTestPost())
  };
  
  const bulkResponse = http.post(`${BASE_URL}/api/posts/bulk`, JSON.stringify(bulkData), { headers });
  
  check(bulkResponse, {
    'Bulk operation status is 200': (r) => r.status === 200,
    'Bulk operation response time < 1000ms': (r) => r.timings.duration < 1000
  }) || errorRate.add(1);
  
  apiLatency.add(bulkResponse.timings.duration);
  throughput.add(1);
  
  // Test complex search
  const searchResponse = http.get(`${BASE_URL}/api/posts/search?q=load test&sort=created_at&order=desc&limit=50`, { headers });
  
  check(searchResponse, {
    'Search operation status is 200': (r) => r.status === 200,
    'Search operation response time < 500ms': (r) => r.timings.duration < 500
  }) || errorRate.add(1);
  
  apiLatency.add(searchResponse.timings.duration);
  throughput.add(1);
}

export function teardown(data) {
  // Cleanup test data if needed
  console.log('Load test completed');
}