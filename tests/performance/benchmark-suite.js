import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Performance metrics
const performanceScore = new Gauge('performance_score');
const capacityUtilization = new Gauge('capacity_utilization');
const resourceEfficiency = new Gauge('resource_efficiency');
const scalabilityIndex = new Gauge('scalability_index');
const throughputPerCore = new Trend('throughput_per_core');
const memoryEfficiency = new Trend('memory_efficiency');
const networkUtilization = new Trend('network_utilization');

// Benchmark-specific metrics
const apiThroughput = new Counter('api_requests_per_second');
const databaseThroughput = new Counter('database_ops_per_second');
const cacheHitRate = new Rate('cache_hit_rate');
const errorBudgetConsumption = new Rate('error_budget_consumption');

export let options = {
  scenarios: {
    // Baseline performance benchmark
    baseline_benchmark: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
      tags: { benchmark_type: 'baseline' }
    },
    
    // Capacity planning - gradual load increase
    capacity_planning: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '2m', target: 500 },
        { duration: '2m', target: 1000 },
        { duration: '2m', target: 2000 },
        { duration: '2m', target: 5000 },
        { duration: '2m', target: 10000 },
        { duration: '5m', target: 10000 }, // Sustained peak
        { duration: '2m', target: 0 }
      ],
      tags: { benchmark_type: 'capacity' }
    },
    
    // Scalability test - exponential growth
    scalability_test: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '1m', target: 10 },
        { duration: '1m', target: 100 },
        { duration: '1m', target: 1000 },
        { duration: '1m', target: 10000 },
        { duration: '2m', target: 10000 },
        { duration: '1m', target: 0 }
      ],
      tags: { benchmark_type: 'scalability' }
    },
    
    // Resource efficiency test
    efficiency_test: {
      executor: 'constant-vus',
      vus: 1000,
      duration: '10m',
      tags: { benchmark_type: 'efficiency' }
    },
    
    // Endurance test - 24 hour soak
    endurance_test: {
      executor: 'constant-vus',
      vus: 500,
      duration: '24h',
      tags: { benchmark_type: 'endurance' }
    }
  },
  
  thresholds: {
    // SLO-based thresholds
    'http_req_duration': ['p(95)<200', 'p(99)<500'],
    'http_req_failed': ['rate<0.01'],
    'performance_score': ['value>0.8'],
    'capacity_utilization': ['value<0.8'], // Don't exceed 80% capacity
    'resource_efficiency': ['value>0.7'],
    'scalability_index': ['value>0.8'],
    'cache_hit_rate': ['rate>0.8'],
    'error_budget_consumption': ['rate<0.1']
  }
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const METRICS_URL = __ENV.METRICS_URL || 'http://localhost:9090';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';
const TENANT_ID = __ENV.TENANT_ID || 'benchmark-tenant';

// Benchmark configuration
const BENCHMARK_CONFIG = {
  baseline: {
    operations: ['read_heavy', 'write_light', 'cache_test'],
    weights: [0.7, 0.2, 0.1],
    target_rps: 1000
  },
  capacity: {
    operations: ['mixed_workload', 'database_intensive', 'media_operations'],
    weights: [0.5, 0.3, 0.2],
    max_rps: 50000
  },
  scalability: {
    operations: ['concurrent_users', 'data_growth', 'feature_complexity'],
    weights: [0.4, 0.3, 0.3],
    scaling_factor: 10
  },
  efficiency: {
    operations: ['resource_optimization', 'cache_efficiency', 'query_optimization'],
    weights: [0.4, 0.3, 0.3],
    efficiency_target: 0.8
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
  
  // Collect baseline system metrics
  const systemMetrics = collectSystemMetrics();
  
  return { 
    token,
    baselineMetrics: systemMetrics,
    startTime: Date.now()
  };
}

export default function(data) {
  const token = data?.token || JWT_TOKEN;
  const benchmarkType = __ENV.K6_SCENARIO_NAME || 'baseline_benchmark';
  const testType = benchmarkType.replace('_benchmark', '').replace('_test', '');
  
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
    'X-Tenant-ID': TENANT_ID
  };

  // Execute benchmark based on scenario
  switch (testType) {
    case 'baseline':
      runBaselineBenchmark(headers);
      break;
    case 'capacity_planning':
      runCapacityPlanningTest(headers);
      break;
    case 'scalability':
      runScalabilityTest(headers);
      break;
    case 'efficiency':
      runEfficiencyTest(headers);
      break;
    case 'endurance':
      runEnduranceTest(headers);
      break;
    default:
      runGeneralBenchmark(headers);
  }
  
  // Collect real-time metrics
  collectRealTimeMetrics();
  
  sleep(randomIntBetween(1, 2));
}

function runBaselineBenchmark(headers) {
  const config = BENCHMARK_CONFIG.baseline;
  const operation = selectWeightedOperation(config.operations, config.weights);
  
  const startTime = Date.now();
  let result;
  
  switch (operation) {
    case 'read_heavy':
      result = executeReadHeavyWorkload(headers);
      break;
    case 'write_light':
      result = executeWriteLightWorkload(headers);
      break;
    case 'cache_test':
      result = executeCacheTestWorkload(headers);
      break;
  }
  
  const endTime = Date.now();
  const responseTime = endTime - startTime;
  
  // Calculate performance score
  const targetResponseTime = 200; // ms
  const performanceRatio = Math.max(0, (targetResponseTime - responseTime) / targetResponseTime);
  performanceScore.add(performanceRatio);
  
  apiThroughput.add(1);
  
  if (result && result.cacheHit !== undefined) {
    cacheHitRate.add(result.cacheHit ? 1 : 0);
  }
}

function runCapacityPlanningTest(headers) {
  const currentVUs = __VU;
  const config = BENCHMARK_CONFIG.capacity;
  
  // Simulate different load patterns based on VU count
  let workloadIntensity;
  if (currentVUs < 100) {
    workloadIntensity = 'light';
  } else if (currentVUs < 1000) {
    workloadIntensity = 'medium';
  } else if (currentVUs < 5000) {
    workloadIntensity = 'heavy';
  } else {
    workloadIntensity = 'extreme';
  }
  
  const result = executeCapacityWorkload(headers, workloadIntensity);
  
  // Calculate capacity utilization
  const maxCapacity = config.max_rps;
  const currentThroughput = result.throughput || 0;
  const utilization = currentThroughput / maxCapacity;
  capacityUtilization.add(utilization);
  
  // Track resource efficiency
  const efficiency = calculateResourceEfficiency(result);
  resourceEfficiency.add(efficiency);
}

function runScalabilityTest(headers) {
  const config = BENCHMARK_CONFIG.scalability;
  const currentVUs = __VU;
  
  // Test how performance scales with load
  const baselinePerformance = 200; // ms baseline response time
  const result = executeScalabilityWorkload(headers, currentVUs);
  
  if (result && result.responseTime) {
    const scalingFactor = baselinePerformance / result.responseTime;
    const scalabilityScore = Math.min(1, scalingFactor);
    scalabilityIndex.add(scalabilityScore);
  }
  
  // Test different scaling dimensions
  const scalingDimension = selectWeightedOperation(config.operations, config.weights);
  executeScalingDimensionTest(headers, scalingDimension, currentVUs);
}

function runEfficiencyTest(headers) {
  const config = BENCHMARK_CONFIG.efficiency;
  
  // Measure resource efficiency across different operations
  const operations = [
    () => measureCPUEfficiency(headers),
    () => measureMemoryEfficiency(headers),
    () => measureNetworkEfficiency(headers),
    () => measureDatabaseEfficiency(headers)
  ];
  
  const operation = operations[randomIntBetween(0, operations.length - 1)];
  const efficiencyResult = operation();
  
  if (efficiencyResult) {
    resourceEfficiency.add(efficiencyResult.efficiency);
    
    if (efficiencyResult.throughputPerCore) {
      throughputPerCore.add(efficiencyResult.throughputPerCore);
    }
    
    if (efficiencyResult.memoryEfficiency) {
      memoryEfficiency.add(efficiencyResult.memoryEfficiency);
    }
    
    if (efficiencyResult.networkUtilization) {
      networkUtilization.add(efficiencyResult.networkUtilization);
    }
  }
}

function runEnduranceTest(headers) {
  // Long-running test to detect memory leaks and performance degradation
  const testDuration = Date.now() - (__ENV.TEST_START_TIME || Date.now());
  const hoursRunning = testDuration / (1000 * 60 * 60);
  
  // Vary workload over time to simulate real usage patterns
  const timeOfDay = new Date().getHours();
  let workloadMultiplier;
  
  if (timeOfDay >= 9 && timeOfDay <= 17) {
    workloadMultiplier = 1.5; // Business hours
  } else if (timeOfDay >= 18 && timeOfDay <= 22) {
    workloadMultiplier = 1.2; // Evening peak
  } else {
    workloadMultiplier = 0.5; // Off hours
  }
  
  const result = executeEnduranceWorkload(headers, workloadMultiplier, hoursRunning);
  
  // Monitor for performance degradation over time
  if (result && result.responseTime) {
    const expectedDegradation = 1 + (hoursRunning * 0.01); // 1% per hour acceptable
    const actualDegradation = result.responseTime / 200; // vs 200ms baseline
    
    if (actualDegradation > expectedDegradation * 1.5) {
      errorBudgetConsumption.add(1);
    }
  }
}

// Workload execution functions
function executeReadHeavyWorkload(headers) {
  const operations = [
    () => http.get(`${BASE_URL}/api/posts?limit=20&sort=created_at`, { headers }),
    () => http.get(`${BASE_URL}/api/users?limit=10&active=true`, { headers }),
    () => http.get(`${BASE_URL}/api/categories`, { headers }),
    () => http.get(`${BASE_URL}/api/posts/popular?limit=5`, { headers }),
    () => http.get(`${BASE_URL}/api/media?type=image&limit=15`, { headers })
  ];
  
  const operation = operations[randomIntBetween(0, operations.length - 1)];
  const response = operation();
  
  const success = check(response, {
    'Read operation successful': (r) => r.status < 400,
    'Read operation fast': (r) => r.timings.duration < 200
  });
  
  // Check if response came from cache
  const cacheHit = response.headers['X-Cache-Status'] === 'HIT';
  
  return {
    success,
    responseTime: response.timings.duration,
    cacheHit,
    throughput: success ? 1 : 0
  };
}

function executeWriteLightWorkload(headers) {
  const writeOperations = [
    () => createTestPost(headers),
    () => updateTestPost(headers),
    () => createTestUser(headers)
  ];
  
  const operation = writeOperations[randomIntBetween(0, writeOperations.length - 1)];
  const result = operation();
  
  databaseThroughput.add(result.success ? 1 : 0);
  
  return result;
}

function executeCacheTestWorkload(headers) {
  // Test cache performance with repeated requests
  const cacheKey = `cache-test-${randomIntBetween(1, 100)}`;
  
  // First request (cache miss expected)
  const firstResponse = http.get(`${BASE_URL}/api/posts/cached/${cacheKey}`, { headers });
  
  sleep(0.1);
  
  // Second request (cache hit expected)
  const secondResponse = http.get(`${BASE_URL}/api/posts/cached/${cacheKey}`, { headers });
  
  const cacheHit = secondResponse.timings.duration < firstResponse.timings.duration * 0.5;
  
  return {
    success: secondResponse.status < 400,
    responseTime: secondResponse.timings.duration,
    cacheHit,
    cacheEfficiency: cacheHit ? 1 : 0
  };
}

function executeCapacityWorkload(headers, intensity) {
  const workloads = {
    light: () => executeReadHeavyWorkload(headers),
    medium: () => executeMixedWorkload(headers),
    heavy: () => executeWriteHeavyWorkload(headers),
    extreme: () => executeComplexWorkload(headers)
  };
  
  const workload = workloads[intensity] || workloads.medium;
  return workload();
}

function executeScalabilityWorkload(headers, vuCount) {
  // Adjust workload complexity based on VU count
  const complexity = Math.min(10, Math.floor(vuCount / 100));
  
  const operations = [];
  for (let i = 0; i < complexity; i++) {
    operations.push(() => http.get(`${BASE_URL}/api/posts?page=${i + 1}&limit=10`, { headers }));
  }
  
  const startTime = Date.now();
  const results = operations.map(op => op());
  const endTime = Date.now();
  
  const avgResponseTime = results.reduce((sum, r) => sum + r.timings.duration, 0) / results.length;
  const successRate = results.filter(r => r.status < 400).length / results.length;
  
  return {
    responseTime: avgResponseTime,
    successRate,
    throughput: results.length / ((endTime - startTime) / 1000)
  };
}

function executeScalingDimensionTest(headers, dimension, vuCount) {
  switch (dimension) {
    case 'concurrent_users':
      return testConcurrentUserScaling(headers, vuCount);
    case 'data_growth':
      return testDataGrowthScaling(headers, vuCount);
    case 'feature_complexity':
      return testFeatureComplexityScaling(headers, vuCount);
  }
}

// Efficiency measurement functions
function measureCPUEfficiency(headers) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/posts/compute-intensive`, { headers });
  const endTime = Date.now();
  
  const processingTime = endTime - startTime;
  const efficiency = Math.max(0, (1000 - processingTime) / 1000); // Target 1 second
  
  return {
    efficiency,
    throughputPerCore: 1000 / processingTime, // ops per second per core
    responseTime: processingTime
  };
}

function measureMemoryEfficiency(headers) {
  // Test memory-intensive operations
  const response = http.post(`${BASE_URL}/api/data/bulk-process`, JSON.stringify({
    size: 1000,
    operation: 'memory-test'
  }), { headers });
  
  const memoryScore = response.headers['X-Memory-Usage'] || '0';
  const memoryUsage = parseInt(memoryScore);
  const efficiency = Math.max(0, (100 - memoryUsage) / 100); // Target <100MB
  
  return {
    efficiency,
    memoryEfficiency: efficiency,
    memoryUsage
  };
}

function measureNetworkEfficiency(headers) {
  const payloadSizes = [1000, 10000, 100000]; // Different payload sizes
  const results = [];
  
  for (const size of payloadSizes) {
    const payload = 'x'.repeat(size);
    const startTime = Date.now();
    const response = http.post(`${BASE_URL}/api/echo`, payload, { headers });
    const endTime = Date.now();
    
    const throughput = size / (endTime - startTime); // bytes per ms
    results.push(throughput);
  }
  
  const avgThroughput = results.reduce((sum, t) => sum + t, 0) / results.length;
  const efficiency = Math.min(1, avgThroughput / 1000); // Target 1MB/s
  
  return {
    efficiency,
    networkUtilization: efficiency,
    throughput: avgThroughput
  };
}

function measureDatabaseEfficiency(headers) {
  const queries = [
    'simple-select',
    'complex-join',
    'aggregation',
    'full-text-search'
  ];
  
  const results = [];
  
  for (const query of queries) {
    const startTime = Date.now();
    const response = http.get(`${BASE_URL}/api/benchmark/db/${query}`, { headers });
    const endTime = Date.now();
    
    if (response.status < 400) {
      results.push(endTime - startTime);
    }
  }
  
  if (results.length === 0) return { efficiency: 0 };
  
  const avgQueryTime = results.reduce((sum, t) => sum + t, 0) / results.length;
  const efficiency = Math.max(0, (100 - avgQueryTime) / 100); // Target <100ms
  
  return {
    efficiency,
    avgQueryTime,
    queryCount: results.length
  };
}

// Helper functions
function selectWeightedOperation(operations, weights) {
  const random = Math.random();
  let cumulativeWeight = 0;
  
  for (let i = 0; i < operations.length; i++) {
    cumulativeWeight += weights[i];
    if (random <= cumulativeWeight) {
      return operations[i];
    }
  }
  
  return operations[operations.length - 1];
}

function calculateResourceEfficiency(result) {
  if (!result) return 0;
  
  const factors = [
    result.responseTime ? Math.max(0, (500 - result.responseTime) / 500) : 0,
    result.throughput ? Math.min(1, result.throughput / 1000) : 0,
    result.successRate || 0,
    result.cacheEfficiency || 0
  ];
  
  return factors.reduce((sum, f) => sum + f, 0) / factors.length;
}

function collectSystemMetrics() {
  // Collect system metrics from monitoring endpoints
  try {
    const metricsResponse = http.get(`${METRICS_URL}/api/v1/query?query=up`);
    if (metricsResponse.status === 200) {
      return JSON.parse(metricsResponse.body);
    }
  } catch (e) {
    console.log('Could not collect system metrics:', e.message);
  }
  
  return {};
}

function collectRealTimeMetrics() {
  // Collect real-time performance metrics
  const currentTime = Date.now();
  const vuCount = __VU;
  
  // This would integrate with actual monitoring systems
  // For now, we'll simulate metric collection
  
  if (vuCount % 100 === 0) { // Sample every 100th VU
    try {
      const healthResponse = http.get(`${BASE_URL}/health`);
      if (healthResponse.status === 200) {
        const healthData = JSON.parse(healthResponse.body);
        
        if (healthData.metrics) {
          capacityUtilization.add(healthData.metrics.cpuUsage || 0);
          resourceEfficiency.add(healthData.metrics.efficiency || 0);
        }
      }
    } catch (e) {
      // Ignore metric collection errors
    }
  }
}

// Additional workload functions
function executeMixedWorkload(headers) {
  const readResult = executeReadHeavyWorkload(headers);
  sleep(0.1);
  const writeResult = executeWriteLightWorkload(headers);
  
  return {
    success: readResult.success && writeResult.success,
    responseTime: (readResult.responseTime + writeResult.responseTime) / 2,
    throughput: (readResult.throughput + writeResult.throughput)
  };
}

function executeWriteHeavyWorkload(headers) {
  const operations = [];
  for (let i = 0; i < 5; i++) {
    operations.push(() => createTestPost(headers));
  }
  
  const results = operations.map(op => op());
  const successCount = results.filter(r => r.success).length;
  const avgResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
  
  return {
    success: successCount === results.length,
    responseTime: avgResponseTime,
    throughput: successCount
  };
}

function executeComplexWorkload(headers) {
  // Complex multi-step workflow
  const steps = [
    () => createTestPost(headers),
    () => http.get(`${BASE_URL}/api/posts/search?q=complex`, { headers }),
    () => http.post(`${BASE_URL}/api/posts/bulk-update`, JSON.stringify({
      ids: [1, 2, 3],
      updates: { status: 'published' }
    }), { headers }),
    () => http.get(`${BASE_URL}/api/analytics/posts`, { headers })
  ];
  
  const startTime = Date.now();
  const results = steps.map(step => step());
  const endTime = Date.now();
  
  const successCount = results.filter(r => r.status < 400).length;
  
  return {
    success: successCount === results.length,
    responseTime: endTime - startTime,
    throughput: successCount / ((endTime - startTime) / 1000)
  };
}

function executeEnduranceWorkload(headers, multiplier, hoursRunning) {
  // Simulate realistic usage patterns over time
  const baseWorkload = executeReadHeavyWorkload(headers);
  
  // Add some writes based on multiplier
  if (Math.random() < 0.3 * multiplier) {
    const writeResult = executeWriteLightWorkload(headers);
    return {
      success: baseWorkload.success && writeResult.success,
      responseTime: (baseWorkload.responseTime + writeResult.responseTime) / 2,
      throughput: baseWorkload.throughput + writeResult.throughput
    };
  }
  
  return baseWorkload;
}

function createTestPost(headers) {
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/posts`, JSON.stringify({
    title: `Benchmark Post ${randomString(8)}`,
    content: `Benchmark content ${randomString(50)}`,
    status: 'published'
  }), { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    responseTime: endTime - startTime,
    throughput: response.status < 400 ? 1 : 0
  };
}

function updateTestPost(headers) {
  const startTime = Date.now();
  const response = http.put(`${BASE_URL}/api/posts/1`, JSON.stringify({
    title: `Updated Benchmark Post ${randomString(8)}`
  }), { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    responseTime: endTime - startTime,
    throughput: response.status < 400 ? 1 : 0
  };
}

function createTestUser(headers) {
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/users`, JSON.stringify({
    email: `benchmark-${randomString(8)}@example.com`,
    name: `Benchmark User ${randomString(5)}`,
    role: 'user'
  }), { headers });
  const endTime = Date.now();
  
  return {
    success: response.status < 400,
    responseTime: endTime - startTime,
    throughput: response.status < 400 ? 1 : 0
  };
}

function testConcurrentUserScaling(headers, vuCount) {
  // Test how system handles concurrent user sessions
  const sessionId = `session-${__VU}-${Date.now()}`;
  
  const response = http.get(`${BASE_URL}/api/auth/me`, {
    headers: {
      ...headers,
      'X-Session-ID': sessionId
    }
  });
  
  return {
    success: response.status < 400,
    responseTime: response.timings.duration,
    concurrentUsers: vuCount
  };
}

function testDataGrowthScaling(headers, vuCount) {
  // Test how system handles growing data sets
  const dataSize = Math.floor(vuCount / 10); // Scale data with VUs
  
  const response = http.get(`${BASE_URL}/api/posts?limit=${dataSize}&include=all`, { headers });
  
  return {
    success: response.status < 400,
    responseTime: response.timings.duration,
    dataSize
  };
}

function testFeatureComplexityScaling(headers, vuCount) {
  // Test complex features under load
  const complexity = Math.min(5, Math.floor(vuCount / 1000));
  
  const response = http.post(`${BASE_URL}/api/complex-operation`, JSON.stringify({
    complexity,
    iterations: complexity * 10
  }), { headers });
  
  return {
    success: response.status < 400,
    responseTime: response.timings.duration,
    complexity
  };
}

export function teardown(data) {
  // Generate performance report
  const endTime = Date.now();
  const testDuration = endTime - (data.startTime || endTime);
  
  console.log(`Performance benchmark completed in ${testDuration}ms`);
  
  // This would typically generate a detailed report
  // and send results to a performance monitoring system
}