#!/usr/bin/env node

/**
 * Test Runner - Runs all system tests and generates comprehensive reports
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class TestRunner {
  constructor() {
    this.testSuites = [
      {
        name: 'Comprehensive System Test',
        script: 'test-system-comprehensive.js',
        description: 'Tests all Node.js API endpoints and functionality'
      },
      {
        name: 'C++ Integration Test',
        script: 'test-cpp-integration.js',
        description: 'Tests C++ system integration and performance'
      }
    ];
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

  async runTestSuite(testSuite) {
    return new Promise((resolve, reject) => {
      this.log(`🚀 Starting: ${testSuite.name}`, 'info');
      this.log(`📝 ${testSuite.description}`, 'info');
      
      const scriptPath = path.join(__dirname, testSuite.script);
      const child = spawn('node', [scriptPath], {
        stdio: 'inherit',
        cwd: __dirname
      });
      
      child.on('close', (code) => {
        if (code === 0) {
          this.log(`✅ ${testSuite.name} completed successfully`, 'success');
          resolve({ name: testSuite.name, status: 'success', code });
        } else {
          this.log(`❌ ${testSuite.name} failed with code ${code}`, 'error');
          resolve({ name: testSuite.name, status: 'failed', code });
        }
      });
      
      child.on('error', (error) => {
        this.log(`💥 ${testSuite.name} error: ${error.message}`, 'error');
        reject(error);
      });
    });
  }

  async runAllTests() {
    this.log('🎯 GROOT API SYSTEM TEST SUITE', 'info');
    this.log('=' .repeat(50), 'info');
    this.log('Testing your ultra-low latency system with real data...', 'info');
    this.log('', 'info');
    
    const results = [];
    
    for (const testSuite of this.testSuites) {
      try {
        const result = await this.runTestSuite(testSuite);
        results.push(result);
        this.log('', 'info');
      } catch (error) {
        results.push({ 
          name: testSuite.name, 
          status: 'error', 
          error: error.message 
        });
        this.log('', 'info');
      }
    }
    
    this.generateFinalReport(results);
  }

  generateFinalReport(results) {
    this.log('📊 FINAL TEST REPORT', 'info');
    this.log('=' .repeat(50), 'info');
    
    const successful = results.filter(r => r.status === 'success').length;
    const failed = results.filter(r => r.status === 'failed').length;
    const errors = results.filter(r => r.status === 'error').length;
    
    this.log(`Total Test Suites: ${results.length}`, 'info');
    this.log(`Successful: ${successful}`, 'success');
    this.log(`Failed: ${failed}`, failed > 0 ? 'error' : 'info');
    this.log(`Errors: ${errors}`, errors > 0 ? 'error' : 'info');
    
    this.log('\n📋 DETAILED RESULTS:', 'info');
    results.forEach(result => {
      const status = result.status === 'success' ? '✅' : 
                    result.status === 'failed' ? '❌' : '💥';
      this.log(`${status} ${result.name}`, result.status === 'success' ? 'success' : 'error');
    });
    
    // Check for result files
    const resultFiles = [
      '../test-results.json',
      '../cpp-integration-results.json'
    ];
    
    this.log('\n📁 GENERATED REPORTS:', 'info');
    resultFiles.forEach(file => {
      const filePath = path.join(__dirname, file);
      if (fs.existsSync(filePath)) {
        this.log(`📄 ${path.basename(file)} - Available`, 'success');
      } else {
        this.log(`📄 ${path.basename(file)} - Not generated`, 'warning');
      }
    });
    
    this.log('\n🎯 SYSTEM STATUS:', 'info');
    if (successful === results.length) {
      this.log('🎉 ALL SYSTEMS OPERATIONAL! Your Groot API is working perfectly!', 'success');
      this.log('🚀 Ready for production deployment!', 'success');
    } else if (successful > 0) {
      this.log('⚠️  System partially operational. Some components may need attention.', 'warning');
      this.log('🔧 Check the detailed reports for specific issues.', 'warning');
    } else {
      this.log('🚨 System needs attention. Check server status and configuration.', 'error');
      this.log('💡 Make sure your server is running: npm run dev', 'info');
    }
    
    this.log('\n📖 NEXT STEPS:', 'info');
    this.log('1. Review detailed test results in the generated JSON files', 'info');
    this.log('2. Check server logs for any errors or warnings', 'info');
    this.log('3. Verify all external services are configured correctly', 'info');
    this.log('4. Run performance benchmarks with larger datasets', 'info');
    
    if (successful > 0) {
      this.log('\n🏆 Your system is processing real data successfully!', 'success');
    }
  }
}

// Check if server is running
async function checkServerHealth() {
  const axios = require('axios');
  const baseURL = process.env.API_URL || 'http://localhost:3005';
  
  try {
    await axios.get(`${baseURL}/health`, { timeout: 5000 });
    return true;
  } catch (error) {
    console.log('\n🚨 Server Health Check Failed!');
    console.log('❌ Your server is not responding at:', baseURL);
    console.log('\n💡 To start your server, run:');
    console.log('   npm run dev');
    console.log('\n⏳ Then run this test again:');
    console.log('   npm run test:system');
    console.log('');
    return false;
  }
}

// Main execution
if (require.main === module) {
  checkServerHealth().then(isHealthy => {
    if (isHealthy) {
      const runner = new TestRunner();
      runner.runAllTests().catch(error => {
        console.error('Test runner failed:', error.message);
        process.exit(1);
      });
    } else {
      process.exit(1);
    }
  });
}

module.exports = TestRunner;