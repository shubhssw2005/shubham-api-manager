#!/usr/bin/env node

/**
 * Load Testing Setup Validation
 * 
 * This script validates that the load testing infrastructure is properly configured
 * and all dependencies are available.
 */

import { execSync } from 'child_process';
import fs from 'fs';
import http from 'http';
import https from 'https';
import { URL } from 'url';

class SetupValidator {
  constructor() {
    this.results = {
      dependencies: {},
      configuration: {},
      connectivity: {},
      overall: 'unknown'
    };
  }

  async validate() {
    console.log('🔍 Validating Load Testing Setup...\n');
    
    try {
      await this.validateDependencies();
      await this.validateConfiguration();
      await this.validateConnectivity();
      
      this.generateReport();
      
    } catch (error) {
      console.error('❌ Validation failed:', error.message);
      process.exit(1);
    }
  }

  async validateDependencies() {
    console.log('📦 Checking Dependencies...');
    
    // Check k6
    try {
      const k6Version = execSync('k6 version', { encoding: 'utf8' });
      this.results.dependencies.k6 = {
        status: 'ok',
        version: k6Version.trim(),
        message: 'k6 is installed and accessible'
      };
      console.log('  ✅ k6:', k6Version.trim());
    } catch (error) {
      this.results.dependencies.k6 = {
        status: 'error',
        message: 'k6 is not installed or not in PATH'
      };
      console.log('  ❌ k6: Not found');
    }
    
    // Check Node.js
    try {
      const nodeVersion = process.version;
      const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
      
      if (majorVersion >= 16) {
        this.results.dependencies.nodejs = {
          status: 'ok',
          version: nodeVersion,
          message: 'Node.js version is compatible'
        };
        console.log('  ✅ Node.js:', nodeVersion);
      } else {
        this.results.dependencies.nodejs = {
          status: 'warning',
          version: nodeVersion,
          message: 'Node.js version should be 16 or higher'
        };
        console.log('  ⚠️  Node.js:', nodeVersion, '(recommend 16+)');
      }
    } catch (error) {
      this.results.dependencies.nodejs = {
        status: 'error',
        message: 'Node.js version check failed'
      };
      console.log('  ❌ Node.js: Version check failed');
    }
    
    // Check jq
    try {
      const jqVersion = execSync('jq --version', { encoding: 'utf8' });
      this.results.dependencies.jq = {
        status: 'ok',
        version: jqVersion.trim(),
        message: 'jq is available for JSON processing'
      };
      console.log('  ✅ jq:', jqVersion.trim());
    } catch (error) {
      this.results.dependencies.jq = {
        status: 'warning',
        message: 'jq is not installed (optional but recommended)'
      };
      console.log('  ⚠️  jq: Not found (optional)');
    }
    
    // Check curl
    try {
      const curlVersion = execSync('curl --version | head -1', { encoding: 'utf8' });
      this.results.dependencies.curl = {
        status: 'ok',
        version: curlVersion.trim(),
        message: 'curl is available for connectivity tests'
      };
      console.log('  ✅ curl:', curlVersion.trim());
    } catch (error) {
      this.results.dependencies.curl = {
        status: 'warning',
        message: 'curl is not available'
      };
      console.log('  ⚠️  curl: Not found');
    }
    
    console.log();
  }

  async validateConfiguration() {
    console.log('⚙️  Checking Configuration...');
    
    // Check test scripts exist
    const testFiles = [
      'tests/load/k6-api-load-test.js',
      'tests/load/k6-media-upload-test.js',
      'tests/chaos/chaos-engineering-tests.js',
      'tests/performance/benchmark-suite.js',
      'scripts/capacity-planning-tool.js',
      'scripts/run-load-tests.sh'
    ];
    
    let allFilesExist = true;
    
    for (const file of testFiles) {
      if (fs.existsSync(file)) {
        console.log(`  ✅ ${file}`);
      } else {
        console.log(`  ❌ ${file}: Missing`);
        allFilesExist = false;
      }
    }
    
    this.results.configuration.testFiles = {
      status: allFilesExist ? 'ok' : 'error',
      message: allFilesExist ? 'All test files present' : 'Some test files are missing'
    };
    
    // Check configuration file
    const configFile = 'tests/performance/load-test-config.json';
    if (fs.existsSync(configFile)) {
      try {
        const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
        this.results.configuration.configFile = {
          status: 'ok',
          message: 'Configuration file is valid JSON'
        };
        console.log(`  ✅ ${configFile}: Valid`);
      } catch (error) {
        this.results.configuration.configFile = {
          status: 'error',
          message: 'Configuration file has invalid JSON'
        };
        console.log(`  ❌ ${configFile}: Invalid JSON`);
      }
    } else {
      this.results.configuration.configFile = {
        status: 'error',
        message: 'Configuration file is missing'
      };
      console.log(`  ❌ ${configFile}: Missing`);
    }
    
    // Check output directory
    const outputDir = 'tests/performance/results';
    if (!fs.existsSync(outputDir)) {
      try {
        fs.mkdirSync(outputDir, { recursive: true });
        this.results.configuration.outputDir = {
          status: 'ok',
          message: 'Output directory created'
        };
        console.log(`  ✅ ${outputDir}: Created`);
      } catch (error) {
        this.results.configuration.outputDir = {
          status: 'error',
          message: 'Cannot create output directory'
        };
        console.log(`  ❌ ${outputDir}: Cannot create`);
      }
    } else {
      this.results.configuration.outputDir = {
        status: 'ok',
        message: 'Output directory exists'
      };
      console.log(`  ✅ ${outputDir}: Exists`);
    }
    
    // Check script permissions
    const scriptFile = 'scripts/run-load-tests.sh';
    if (fs.existsSync(scriptFile)) {
      try {
        const stats = fs.statSync(scriptFile);
        const isExecutable = !!(stats.mode & parseInt('111', 8));
        
        if (isExecutable) {
          this.results.configuration.scriptPermissions = {
            status: 'ok',
            message: 'Run script is executable'
          };
          console.log(`  ✅ ${scriptFile}: Executable`);
        } else {
          this.results.configuration.scriptPermissions = {
            status: 'warning',
            message: 'Run script is not executable (run chmod +x)'
          };
          console.log(`  ⚠️  ${scriptFile}: Not executable`);
        }
      } catch (error) {
        this.results.configuration.scriptPermissions = {
          status: 'error',
          message: 'Cannot check script permissions'
        };
        console.log(`  ❌ ${scriptFile}: Permission check failed`);
      }
    }
    
    console.log();
  }

  async validateConnectivity() {
    console.log('🌐 Checking Connectivity...');
    
    // Get URLs from environment or defaults
    const apiUrl = process.env.API_BASE_URL || 'http://localhost:3000';
    const metricsUrl = process.env.METRICS_URL || 'http://localhost:9090';
    const chaosUrl = process.env.CHAOS_API_URL || 'http://localhost:8080';
    
    // Test API connectivity
    await this.testUrl(apiUrl + '/health', 'API Health Endpoint');
    
    // Test metrics connectivity (optional)
    await this.testUrl(metricsUrl + '/api/v1/query?query=up', 'Prometheus Metrics', true);
    
    // Test chaos API connectivity (optional)
    await this.testUrl(chaosUrl + '/health', 'Chaos Engineering API', true);
    
    console.log();
  }

  async testUrl(url, name, optional = false) {
    try {
      const response = await this.makeRequest(url);
      
      if (response.statusCode >= 200 && response.statusCode < 400) {
        this.results.connectivity[name] = {
          status: 'ok',
          statusCode: response.statusCode,
          message: `${name} is accessible`
        };
        console.log(`  ✅ ${name}: ${response.statusCode}`);
      } else {
        this.results.connectivity[name] = {
          status: optional ? 'warning' : 'error',
          statusCode: response.statusCode,
          message: `${name} returned ${response.statusCode}`
        };
        console.log(`  ${optional ? '⚠️' : '❌'} ${name}: ${response.statusCode}`);
      }
    } catch (error) {
      this.results.connectivity[name] = {
        status: optional ? 'warning' : 'error',
        message: `${name} is not accessible: ${error.message}`
      };
      console.log(`  ${optional ? '⚠️' : '❌'} ${name}: ${error.message}`);
    }
  }

  makeRequest(url) {
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const client = urlObj.protocol === 'https:' ? https : http;
      
      const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: 'GET',
        timeout: 5000,
        headers: {
          'User-Agent': 'LoadTest-Setup-Validator/1.0'
        }
      };
      
      const req = client.request(options, (res) => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers
        });
      });
      
      req.on('error', (error) => {
        reject(error);
      });
      
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });
      
      req.end();
    });
  }

  generateReport() {
    console.log('📋 Validation Summary\n');
    
    // Count results
    let totalChecks = 0;
    let passedChecks = 0;
    let warningChecks = 0;
    let failedChecks = 0;
    
    const countResults = (category) => {
      Object.values(category).forEach(result => {
        totalChecks++;
        switch (result.status) {
          case 'ok':
            passedChecks++;
            break;
          case 'warning':
            warningChecks++;
            break;
          case 'error':
            failedChecks++;
            break;
        }
      });
    };
    
    countResults(this.results.dependencies);
    countResults(this.results.configuration);
    countResults(this.results.connectivity);
    
    // Determine overall status
    if (failedChecks === 0) {
      if (warningChecks === 0) {
        this.results.overall = 'excellent';
        console.log('🎉 Overall Status: EXCELLENT');
        console.log('   All checks passed! Ready for load testing.');
      } else {
        this.results.overall = 'good';
        console.log('✅ Overall Status: GOOD');
        console.log('   Ready for load testing with minor warnings.');
      }
    } else {
      this.results.overall = 'needs_attention';
      console.log('⚠️  Overall Status: NEEDS ATTENTION');
      console.log('   Some critical issues need to be resolved.');
    }
    
    console.log(`\n📊 Results: ${passedChecks} passed, ${warningChecks} warnings, ${failedChecks} failed (${totalChecks} total)\n`);
    
    // Show specific issues
    if (failedChecks > 0 || warningChecks > 0) {
      console.log('🔧 Issues to Address:\n');
      
      const showIssues = (category, categoryName) => {
        Object.entries(category).forEach(([name, result]) => {
          if (result.status === 'error') {
            console.log(`   ❌ ${categoryName} - ${name}: ${result.message}`);
          } else if (result.status === 'warning') {
            console.log(`   ⚠️  ${categoryName} - ${name}: ${result.message}`);
          }
        });
      };
      
      showIssues(this.results.dependencies, 'Dependencies');
      showIssues(this.results.configuration, 'Configuration');
      showIssues(this.results.connectivity, 'Connectivity');
    }
    
    // Save detailed results
    const resultsFile = 'tests/performance/setup-validation-results.json';
    fs.writeFileSync(resultsFile, JSON.stringify(this.results, null, 2));
    console.log(`\n💾 Detailed results saved to: ${resultsFile}`);
    
    // Provide next steps
    console.log('\n🚀 Next Steps:');
    
    if (this.results.overall === 'excellent' || this.results.overall === 'good') {
      console.log('   1. Run basic load test: ./scripts/run-load-tests.sh --api-only');
      console.log('   2. Check results in: tests/performance/results/');
      console.log('   3. Review performance report');
    } else {
      console.log('   1. Resolve the issues listed above');
      console.log('   2. Re-run this validation: node tests/performance/validate-setup.js');
      console.log('   3. Once all issues are resolved, start load testing');
    }
    
    console.log('\n📚 Documentation: tests/performance/README.md');
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new SetupValidator();
  validator.validate().catch(console.error);
}

export default SetupValidator;