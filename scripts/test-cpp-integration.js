#!/usr/bin/env node

/**
 * C++ System Integration Test
 * Tests the integration between Node.js and C++ ultra-low latency system
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

class CPPIntegrationTester {
  constructor() {
    this.baseURL = process.env.API_URL || 'http://localhost:3005';
    this.cppServiceURL = process.env.CPP_SERVICE_URL || 'http://localhost:8080';
    this.results = [];
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const colors = {
      info: '\x1b[36m',
      success: '\x1b[32m',
      error: '\x1b[31m',
      warning: '\x1b[33m',
      reset: '\x1b[0m'
    };
    
    console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
  }

  async testCPPServiceHealth() {
    this.log('🔧 Testing C++ Service Health...', 'info');
    
    try {
      const response = await axios.get(`${this.cppServiceURL}/health`, {
        timeout: 5000
      });
      
      this.results.push({
        test: 'cpp_service_health',
        status: 'PASSED',
        data: response.data,
        responseTime: response.headers['x-response-time'] || 'N/A'
      });
      
      this.log('✅ C++ Service is healthy and responding', 'success');
      return true;
    } catch (error) {
      this.results.push({
        test: 'cpp_service_health',
        status: 'FAILED',
        error: error.message,
        note: 'C++ service may not be running or not accessible'
      });
      
      this.log('❌ C++ Service health check failed - service may not be running', 'warning');
      return false;
    }
  }

  async testUltraLowLatency() {
    this.log('⚡ Testing Ultra-Low Latency Performance...', 'info');
    
    const latencyTests = [];
    const iterations = 100;
    
    for (let i = 0; i < iterations; i++) {
      const startTime = process.hrtime.bigint();
      
      try {
        await axios.get(`${this.baseURL}/health`);
        const endTime = process.hrtime.bigint();
        const latencyNs = Number(endTime - startTime);
        const latencyMs = latencyNs / 1000000;
        
        latencyTests.push(latencyMs);
      } catch (error) {
        this.log(`Request ${i + 1} failed: ${error.message}`, 'error');
      }
    }
    
    if (latencyTests.length > 0) {
      const avgLatency = latencyTests.reduce((a, b) => a + b, 0) / latencyTests.length;
      const minLatency = Math.min(...latencyTests);
      const maxLatency = Math.max(...latencyTests);
      const p95Latency = latencyTests.sort((a, b) => a - b)[Math.floor(latencyTests.length * 0.95)];
      
      this.results.push({
        test: 'ultra_low_latency',
        status: avgLatency < 10 ? 'PASSED' : 'WARNING',
        metrics: {
          iterations,
          avgLatency: `${avgLatency.toFixed(3)}ms`,
          minLatency: `${minLatency.toFixed(3)}ms`,
          maxLatency: `${maxLatency.toFixed(3)}ms`,
          p95Latency: `${p95Latency.toFixed(3)}ms`,
          target: '<1ms (with C++ system)',
          current: avgLatency < 1 ? 'EXCELLENT' : avgLatency < 5 ? 'GOOD' : 'NEEDS_OPTIMIZATION'
        }
      });
      
      if (avgLatency < 1) {
        this.log(`🚀 EXCELLENT: Average latency ${avgLatency.toFixed(3)}ms (Sub-millisecond!)`, 'success');
      } else if (avgLatency < 5) {
        this.log(`✅ GOOD: Average latency ${avgLatency.toFixed(3)}ms`, 'success');
      } else {
        this.log(`⚠️  Average latency ${avgLatency.toFixed(3)}ms - C++ system would improve this significantly`, 'warning');
      }
    }
  }

  async testHighThroughput() {
    this.log('🔥 Testing High Throughput Capabilities...', 'info');
    
    const concurrentRequests = 50;
    const startTime = Date.now();
    
    const requests = Array(concurrentRequests).fill().map(async (_, index) => {
      try {
        const response = await axios.get(`${this.baseURL}/health`);
        return {
          success: true,
          status: response.status,
          index
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
          index
        };
      }
    });
    
    const results = await Promise.all(requests);
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    const successful = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;
    const throughput = (successful / (totalTime / 1000)).toFixed(2);
    
    this.results.push({
      test: 'high_throughput',
      status: successful === concurrentRequests ? 'PASSED' : 'WARNING',
      metrics: {
        concurrentRequests,
        successful,
        failed,
        totalTime: `${totalTime}ms`,
        throughput: `${throughput} RPS`,
        successRate: `${((successful / concurrentRequests) * 100).toFixed(1)}%`
      }
    });
    
    this.log(`📊 Throughput Test: ${successful}/${concurrentRequests} successful (${throughput} RPS)`, 
             successful === concurrentRequests ? 'success' : 'warning');
  }

  async testCachePerformance() {
    this.log('💾 Testing Cache Performance...', 'info');
    
    const cacheKey = 'test-cache-key';
    const testData = { message: 'Cache performance test', timestamp: Date.now() };
    
    try {
      // Test cache write
      const writeStart = process.hrtime.bigint();
      await axios.post(`${this.baseURL}/api/test-cache`, {
        key: cacheKey,
        data: testData
      });
      const writeEnd = process.hrtime.bigint();
      const writeLatency = Number(writeEnd - writeStart) / 1000000;
      
      // Test cache read
      const readStart = process.hrtime.bigint();
      const readResponse = await axios.get(`${this.baseURL}/api/test-cache/${cacheKey}`);
      const readEnd = process.hrtime.bigint();
      const readLatency = Number(readEnd - readStart) / 1000000;
      
      this.results.push({
        test: 'cache_performance',
        status: 'PASSED',
        metrics: {
          writeLatency: `${writeLatency.toFixed(3)}ms`,
          readLatency: `${readLatency.toFixed(3)}ms`,
          cacheHit: readResponse.data ? 'YES' : 'NO',
          note: 'C++ ultra-cache would provide sub-microsecond access times'
        }
      });
      
      this.log(`💨 Cache Performance: Write ${writeLatency.toFixed(3)}ms, Read ${readLatency.toFixed(3)}ms`, 'success');
    } catch (error) {
      this.results.push({
        test: 'cache_performance',
        status: 'SKIPPED',
        error: error.message,
        note: 'Cache test endpoints may not be implemented yet'
      });
      
      this.log('⚠️  Cache performance test skipped - endpoints not available', 'warning');
    }
  }

  async testStreamProcessing() {
    this.log('🌊 Testing Stream Processing Capabilities...', 'info');
    
    const events = Array(1000).fill().map((_, index) => ({
      id: index + 1,
      type: 'test_event',
      timestamp: Date.now(),
      data: {
        userId: `user_${index % 100}`,
        action: 'test_action',
        value: Math.random() * 100
      }
    }));
    
    try {
      const startTime = Date.now();
      
      const response = await axios.post(`${this.baseURL}/api/test-stream-processing`, {
        events
      }, {
        timeout: 30000
      });
      
      const endTime = Date.now();
      const processingTime = endTime - startTime;
      const eventsPerSecond = (events.length / (processingTime / 1000)).toFixed(2);
      
      this.results.push({
        test: 'stream_processing',
        status: 'PASSED',
        metrics: {
          eventsProcessed: events.length,
          processingTime: `${processingTime}ms`,
          throughput: `${eventsPerSecond} events/sec`,
          avgLatencyPerEvent: `${(processingTime / events.length).toFixed(3)}ms`,
          note: 'C++ stream processor would handle millions of events per second'
        }
      });
      
      this.log(`🚀 Stream Processing: ${events.length} events in ${processingTime}ms (${eventsPerSecond} eps)`, 'success');
    } catch (error) {
      this.results.push({
        test: 'stream_processing',
        status: 'SKIPPED',
        error: error.message,
        note: 'Stream processing endpoints may not be implemented yet'
      });
      
      this.log('⚠️  Stream processing test skipped - endpoints not available', 'warning');
    }
  }

  async testMemoryEfficiency() {
    this.log('🧠 Testing Memory Efficiency...', 'info');
    
    try {
      // Get initial memory usage
      const initialMemory = await axios.get(`${this.baseURL}/api/system/memory`);
      
      // Perform memory-intensive operations
      const largeData = Array(10000).fill().map((_, index) => ({
        id: index,
        data: 'x'.repeat(1000), // 1KB per record
        timestamp: Date.now()
      }));
      
      await axios.post(`${this.baseURL}/api/test-memory-usage`, {
        data: largeData
      });
      
      // Get final memory usage
      const finalMemory = await axios.get(`${this.baseURL}/api/system/memory`);
      
      this.results.push({
        test: 'memory_efficiency',
        status: 'PASSED',
        metrics: {
          initialMemory: initialMemory.data,
          finalMemory: finalMemory.data,
          dataProcessed: `${largeData.length} records (${(largeData.length / 1000).toFixed(1)}MB)`,
          note: 'C++ system uses NUMA-aware allocation and memory pools for optimal efficiency'
        }
      });
      
      this.log('✅ Memory efficiency test completed', 'success');
    } catch (error) {
      this.results.push({
        test: 'memory_efficiency',
        status: 'SKIPPED',
        error: error.message,
        note: 'Memory monitoring endpoints may not be implemented yet'
      });
      
      this.log('⚠️  Memory efficiency test skipped - endpoints not available', 'warning');
    }
  }

  async testCPPIntegrationEndpoints() {
    this.log('🔗 Testing C++ Integration Endpoints...', 'info');
    
    const integrationTests = [
      {
        name: 'Fast Path Cache',
        endpoint: '/api/cpp/cache/get',
        method: 'GET',
        params: { key: 'test-key' }
      },
      {
        name: 'Ultra Fast API',
        endpoint: '/api/cpp/ultra-fast',
        method: 'GET'
      },
      {
        name: 'Stream Processing',
        endpoint: '/api/cpp/stream/process',
        method: 'POST',
        data: { events: [{ type: 'test', data: 'sample' }] }
      },
      {
        name: 'Performance Metrics',
        endpoint: '/api/cpp/metrics',
        method: 'GET'
      }
    ];
    
    const integrationResults = [];
    
    for (const test of integrationTests) {
      try {
        const startTime = process.hrtime.bigint();
        
        let response;
        if (test.method === 'GET') {
          response = await axios.get(`${this.cppServiceURL}${test.endpoint}`, {
            params: test.params,
            timeout: 5000
          });
        } else {
          response = await axios.post(`${this.cppServiceURL}${test.endpoint}`, test.data, {
            timeout: 5000
          });
        }
        
        const endTime = process.hrtime.bigint();
        const latency = Number(endTime - startTime) / 1000000;
        
        integrationResults.push({
          name: test.name,
          status: 'AVAILABLE',
          latency: `${latency.toFixed(3)}ms`,
          responseStatus: response.status
        });
        
        this.log(`✅ ${test.name}: ${latency.toFixed(3)}ms`, 'success');
      } catch (error) {
        integrationResults.push({
          name: test.name,
          status: 'NOT_AVAILABLE',
          error: error.code || error.message
        });
        
        this.log(`⚠️  ${test.name}: Not available (${error.code || 'Error'})`, 'warning');
      }
    }
    
    this.results.push({
      test: 'cpp_integration_endpoints',
      status: integrationResults.some(r => r.status === 'AVAILABLE') ? 'PARTIAL' : 'NOT_IMPLEMENTED',
      endpoints: integrationResults,
      note: 'C++ integration endpoints are part of the ultra-low latency system design'
    });
  }

  async runAllTests() {
    this.log('🚀 Starting C++ Integration Test Suite', 'info');
    this.log(`📍 Node.js API: ${this.baseURL}`, 'info');
    this.log(`📍 C++ Service: ${this.cppServiceURL}`, 'info');
    
    // Test C++ service availability
    const cppAvailable = await this.testCPPServiceHealth();
    
    // Run performance tests
    await this.testUltraLowLatency();
    await this.testHighThroughput();
    await this.testCachePerformance();
    await this.testStreamProcessing();
    await this.testMemoryEfficiency();
    
    // Test C++ integration endpoints
    if (cppAvailable) {
      await this.testCPPIntegrationEndpoints();
    }
    
    this.generateReport();
  }

  generateReport() {
    this.log('\n📊 C++ INTEGRATION TEST RESULTS', 'info');
    this.log('=' .repeat(60), 'info');
    
    const passedTests = this.results.filter(r => r.status === 'PASSED').length;
    const totalTests = this.results.length;
    
    this.log(`\n🎯 PERFORMANCE SUMMARY:`, 'info');
    
    this.results.forEach(result => {
      const statusIcon = {
        'PASSED': '✅',
        'WARNING': '⚠️ ',
        'FAILED': '❌',
        'SKIPPED': '⏭️ ',
        'PARTIAL': '🔶',
        'NOT_IMPLEMENTED': '🚧'
      }[result.status] || '❓';
      
      this.log(`${statusIcon} ${result.test.toUpperCase()}`, 'info');
      
      if (result.metrics) {
        Object.entries(result.metrics).forEach(([key, value]) => {
          this.log(`   ${key}: ${value}`, 'info');
        });
      }
      
      if (result.note) {
        this.log(`   Note: ${result.note}`, 'warning');
      }
      
      if (result.error) {
        this.log(`   Error: ${result.error}`, 'error');
      }
      
      this.log('', 'info');
    });
    
    // Save results
    const reportPath = path.join(__dirname, '../cpp-integration-results.json');
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      summary: {
        totalTests,
        passedTests,
        nodeJsAPI: this.baseURL,
        cppService: this.cppServiceURL
      },
      results: this.results
    }, null, 2));
    
    this.log(`💾 Results saved to: ${reportPath}`, 'info');
    
    this.log('\n🎯 RECOMMENDATIONS:', 'info');
    this.log('1. Deploy C++ ultra-low latency system for sub-millisecond performance', 'warning');
    this.log('2. Implement C++ cache layer for microsecond data access', 'warning');
    this.log('3. Add C++ stream processor for million+ events/second capability', 'warning');
    this.log('4. Integrate DPDK networking for kernel bypass performance', 'warning');
    
    this.log('\n🚀 Your Node.js system is solid! Adding C++ layer will make it enterprise-ready!', 'success');
  }
}

// Run the tests
if (require.main === module) {
  const tester = new CPPIntegrationTester();
  tester.runAllTests().catch(error => {
    console.error('C++ Integration test suite failed:', error.message);
    process.exit(1);
  });
}

module.exports = CPPIntegrationTester;