import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Chaos engineering metrics
const chaosErrorRate = new Rate('chaos_errors');
const recoveryTime = new Trend('recovery_time');
const resilienceScore = new Rate('resilience_score');
const failureInjections = new Counter('failure_injections');

export let options = {
  scenarios: {
    // Database failure simulation
    database_chaos: {
      executor: 'constant-vus',
      vus: 20,
      duration: '10m',
      tags: { chaos_type: 'database' }
    },
    
    // Redis cache failure simulation
    cache_chaos: {
      executor: 'constant-vus',
      vus: 15,
      duration: '8m',
      tags: { chaos_type: 'cache' }
    },
    
    // S3 service failure simulation
    storage_chaos: {
      executor: 'constant-vus',
      vus: 10,
      duration: '12m',
      tags: { chaos_type: 'storage' }
    },
    
    // Network partition simulation
    network_chaos: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '2m', target: 25 },
        { duration: '5m', target: 25 },
        { duration: '2m', target: 5 }
      ],
      tags: { chaos_type: 'network' }
    },
    
    // Pod failure simulation
    pod_chaos: {
      executor: 'constant-vus',
      vus: 30,
      duration: '15m',
      tags: { chaos_type: 'pod' }
    }
  },
  
  thresholds: {
    'http_req_duration': ['p(95)<1000', 'p(99)<2000'], // More lenient during chaos
    'chaos_errors': ['rate<0.1'], // Allow 10% error rate during chaos
    'resilience_score': ['rate>0.8'], // 80% resilience target
    'recovery_time': ['p(95)<30000'], // 30 second recovery time
    'checks': ['rate>0.8']
  }
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const CHAOS_API_URL = __ENV.CHAOS_API_URL || 'http://localhost:8080';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';
const TENANT_ID = __ENV.TENANT_ID || 'test-tenant';

// Chaos experiment configurations
const CHAOS_EXPERIMENTS = {
  database: {
    name: 'database-failure',
    duration: 120, // 2 minutes
    config: {
      action: 'pod-failure',
      selector: 'app=postgres',
      mode: 'one'
    }
  },
  cache: {
    name: 'redis-failure',
    duration: 90, // 1.5 minutes
    config: {
      action: 'network-partition',
      selector: 'app=redis',
      mode: 'all'
    }
  },
  storage: {
    name: 's3-latency',
    duration: 180, // 3 minutes
    config: {
      action: 'network-delay',
      selector: 'service=s3-proxy',
      delay: '2000ms'
    }
  },
  network: {
    name: 'network-partition',
    duration: 150, // 2.5 minutes
    config: {
      action: 'network-partition',
      selector: 'tier=api',
      mode: 'random-one'
    }
  },
  pod: {
    name: 'api-pod-kill',
    duration: 100, // 1.67 minutes
    config: {
      action: 'pod-kill',
      selector: 'app=api-service',
      mode: 'random-max-percent',
      value: '50'
    }
  }
};

export function setup() {
  // Authenticate
  let token = JWT_TOKEN;
  if (!token) {
    const authResponse = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
      email: 'admin@example.com',
      password: 'admin123'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (authResponse.status === 200) {
      token = JSON.parse(authResponse.body).token;
    }
  }
  
  return { 
    token,
    baselineMetrics: measureBaselinePerformance(token)
  };
}

export default function(data) {
  const token = data?.token || JWT_TOKEN;
  const chaosType = __ENV.K6_SCENARIO_NAME || 'database_chaos';
  const experimentType = chaosType.replace('_chaos', '');
  
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
    'X-Tenant-ID': TENANT_ID
  };

  // Execute chaos experiment based on scenario
  switch (experimentType) {
    case 'database':
      testDatabaseChaos(headers);
      break;
    case 'cache':
      testCacheChaos(headers);
      break;
    case 'storage':
      testStorageChaos(headers);
      break;
    case 'network':
      testNetworkChaos(headers);
      break;
    case 'pod':
      testPodChaos(headers);
      break;
    default:
      testGeneralResilience(headers);
  }
  
  sleep(randomIntBetween(1, 3));
}

function measureBaselinePerformance(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
    'X-Tenant-ID': TENANT_ID
  };
  
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts?limit=10`, { headers });
  const endTime = Date.now();
  
  return {
    responseTime: endTime - startTime,
    statusCode: response.status,
    timestamp: startTime
  };
}

function testDatabaseChaos(headers) {
  // Inject database failure
  const chaosStart = Date.now();
  injectChaos('database');
  
  // Test database-dependent operations during failure
  const operations = [
    () => testPostOperations(headers),
    () => testUserOperations(headers),
    () => testSearchOperations(headers)
  ];
  
  let successfulOperations = 0;
  let totalOperations = 0;
  
  // Run operations during chaos
  for (let i = 0; i < 5; i++) {
    const operation = operations[i % operations.length];
    const result = operation();
    
    totalOperations++;
    if (result.success) {
      successfulOperations++;
    }
    
    sleep(2);
  }
  
  // Stop chaos and measure recovery
  stopChaos('database');
  const recoveryStart = Date.now();
  
  // Test recovery
  let recovered = false;
  let recoveryAttempts = 0;
  const maxRecoveryAttempts = 15; // 30 seconds max
  
  while (!recovered && recoveryAttempts < maxRecoveryAttempts) {
    const testResult = testPostOperations(headers);
    if (testResult.success && testResult.responseTime < 500) {
      recovered = true;
      recoveryTime.add(Date.now() - recoveryStart);
    }
    
    recoveryAttempts++;
    sleep(2);
  }
  
  // Calculate resilience score
  const resilienceRatio = successfulOperations / totalOperations;
  resilienceScore.add(resilienceRatio);
  
  if (!recovered) {
    chaosErrorRate.add(1);
  }
}

function testCacheChaos(headers) {
  // Inject cache failure
  injectChaos('cache');
  
  // Test cache-dependent operations
  const cacheOperations = [
    () => testCachedEndpoints(headers),
    () => testSessionOperations(headers),
    () => testRateLimitedOperations(headers)
  ];
  
  let degradationDetected = false;
  let operationCount = 0;
  
  for (let i = 0; i < 8; i++) {
    const operation = cacheOperations[i % cacheOperations.length];
    const result = operation();
    
    operationCount++;
    
    // Expect some degradation but not complete failure
    if (result.responseTime > 1000) {
      degradationDetected = true;
    }
    
    // System should still function without cache
    check(result, {
      'Cache failure - system still responds': (r) => r.statusCode < 500,
      'Cache failure - acceptable degradation': (r) => r.responseTime < 2000
    }) || chaosErrorRate.add(1);
    
    sleep(1);
  }
  
  stopChaos('cache');
  
  // Verify graceful degradation
  if (degradationDetected) {
    resilienceScore.add(1); // System degraded gracefully
  }
}

function testStorageChaos(headers) {
  // Inject storage latency/failure
  injectChaos('storage');
  
  // Test storage-dependent operations
  const storageTests = [
    () => testMediaUpload(headers),
    () => testMediaRetrieval(headers),
    () => testPresignedUrls(headers)
  ];
  
  let storageOperationsSuccessful = 0;
  let totalStorageOperations = 0;
  
  for (let i = 0; i < 6; i++) {
    const test = storageTests[i % storageTests.length];
    const result = test();
    
    totalStorageOperations++;
    
    // Check if operation succeeded or failed gracefully
    const gracefulHandling = check(result, {
      'Storage chaos - proper error handling': (r) => r.statusCode !== 500,
      'Storage chaos - timeout handling': (r) => r.responseTime < 10000
    });
    
    if (gracefulHandling) {
      storageOperationsSuccessful++;
    } else {
      chaosErrorRate.add(1);
    }
    
    sleep(2);
  }
  
  stopChaos('storage');
  
  const storageResilience = storageOperationsSuccessful / totalStorageOperations;
  resilienceScore.add(storageResilience);
}

function testNetworkChaos(headers) {
  // Inject network partition
  injectChaos('network');
  
  // Test network-dependent operations
  let networkFailures = 0;
  let networkTests = 0;
  
  for (let i = 0; i < 10; i++) {
    const testResult = testNetworkResilience(headers);
    networkTests++;
    
    if (!testResult.success) {
      networkFailures++;
    }
    
    // Check circuit breaker behavior
    check(testResult, {
      'Network chaos - circuit breaker active': (r) => r.statusCode === 503 || r.statusCode < 400,
      'Network chaos - fast failure': (r) => r.responseTime < 5000
    }) || chaosErrorRate.add(1);
    
    sleep(1.5);
  }
  
  stopChaos('network');
  
  // Network resilience should show circuit breaker behavior
  const networkResilience = (networkTests - networkFailures) / networkTests;
  resilienceScore.add(networkResilience);
}

function testPodChaos(headers) {
  // Inject pod failures
  injectChaos('pod');
  
  // Test load balancing and failover
  let podFailureHandled = true;
  
  for (let i = 0; i < 12; i++) {
    const result = testLoadBalancing(headers);
    
    // Even with pod failures, load balancer should route to healthy pods
    const handledGracefully = check(result, {
      'Pod chaos - load balancer working': (r) => r.statusCode < 500,
      'Pod chaos - reasonable response time': (r) => r.responseTime < 3000
    });
    
    if (!handledGracefully) {
      podFailureHandled = false;
      chaosErrorRate.add(1);
    }
    
    sleep(1);
  }
  
  stopChaos('pod');
  
  if (podFailureHandled) {
    resilienceScore.add(1);
  }
}

// Helper functions for specific operations
function testPostOperations(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts?limit=5`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testUserOperations(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/users/me`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testSearchOperations(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts/search?q=test`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testCachedEndpoints(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts/popular`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testSessionOperations(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/auth/me`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testRateLimitedOperations(headers) {
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/posts`, JSON.stringify({
    title: 'Chaos Test Post',
    content: 'Testing during cache failure'
  }), { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testMediaUpload(headers) {
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/media/upload/presigned`, JSON.stringify({
    filename: 'chaos-test.jpg',
    contentType: 'image/jpeg',
    size: 100000
  }), { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testMediaRetrieval(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/media?limit=5`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testPresignedUrls(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/media/signed-url/test-file.jpg`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testNetworkResilience(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/health`, { 
    headers,
    timeout: '5s'
  });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testLoadBalancing(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts?limit=1`, { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    statusCode: response.status,
    responseTime: endTime - startTime
  };
}

function testGeneralResilience(headers) {
  // General resilience test without specific chaos injection
  const operations = [
    () => testPostOperations(headers),
    () => testUserOperations(headers),
    () => testMediaRetrieval(headers)
  ];
  
  const operation = operations[randomIntBetween(0, operations.length - 1)];
  const result = operation();
  
  check(result, {
    'General resilience - operation successful': (r) => r.success,
    'General resilience - acceptable response time': (r) => r.responseTime < 1000
  }) || chaosErrorRate.add(1);
}

// Chaos injection functions (these would integrate with actual chaos engineering tools)
function injectChaos(type) {
  const experiment = CHAOS_EXPERIMENTS[type];
  if (!experiment) return;
  
  console.log(`Injecting ${type} chaos: ${experiment.name}`);
  
  // In a real implementation, this would call Chaos Mesh, Litmus, or similar tools
  const chaosResponse = http.post(`${CHAOS_API_URL}/chaos/inject`, JSON.stringify({
    experiment: experiment.name,
    config: experiment.config,
    duration: experiment.duration
  }), {
    headers: { 'Content-Type': 'application/json' }
  });
  
  if (chaosResponse.status === 200) {
    failureInjections.add(1);
  }
  
  // Wait for chaos to take effect
  sleep(5);
}

function stopChaos(type) {
  const experiment = CHAOS_EXPERIMENTS[type];
  if (!experiment) return;
  
  console.log(`Stopping ${type} chaos: ${experiment.name}`);
  
  http.post(`${CHAOS_API_URL}/chaos/stop`, JSON.stringify({
    experiment: experiment.name
  }), {
    headers: { 'Content-Type': 'application/json' }
  });
  
  // Wait for system to stabilize
  sleep(3);
}

export function teardown(data) {
  // Ensure all chaos experiments are stopped
  Object.keys(CHAOS_EXPERIMENTS).forEach(type => {
    stopChaos(type);
  });
  
  console.log('Chaos engineering tests completed');
}