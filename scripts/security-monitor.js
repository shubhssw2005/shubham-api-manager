#!/usr/bin/env node

const AWS = require('aws-sdk');
const fs = require('fs');
const path = require('path');

// Configure AWS SDK
const ecr = new AWS.ECR({ region: process.env.AWS_REGION || 'us-east-1' });
const inspector = new AWS.Inspector2({ region: process.env.AWS_REGION || 'us-east-1' });
const secretsManager = new AWS.SecretsManager({ region: process.env.AWS_REGION || 'us-east-1' });

class SecurityMonitor {
  constructor() {
    this.repositories = [
      'strapi-platform/production/api-service',
      'strapi-platform/production/media-service',
      'strapi-platform/production/worker-service'
    ];
  }

  async run() {
    console.log('🔒 Starting security monitoring...');
    
    try {
      // Check ECR scan results
      await this.checkECRScans();
      
      // Check Inspector findings
      await this.checkInspectorFindings();
      
      // Check secrets rotation status
      await this.checkSecretsRotation();
      
      // Generate security report
      await this.generateSecurityReport();
      
      console.log('✅ Security monitoring completed successfully');
    } catch (error) {
      console.error('❌ Security monitoring failed:', error);
      process.exit(1);
    }
  }

  async checkECRScans() {
    console.log('📊 Checking ECR scan results...');
    
    const scanResults = [];
    
    for (const repo of this.repositories) {
      try {
        const images = await ecr.describeImages({
          repositoryName: repo,
          maxResults: 5
        }).promise();
        
        for (const image of images.imageDetails) {
          if (image.imageTags && image.imageTags.length > 0) {
            const findings = await this.getImageScanFindings(repo, image.imageTags[0]);
            scanResults.push({
              repository: repo,
              tag: image.imageTags[0],
              findings: findings
            });
          }
        }
      } catch (error) {
        console.warn(`⚠️  Could not scan repository ${repo}:`, error.message);
      }
    }
    
    this.ecrResults = scanResults;
    this.logScanSummary(scanResults);
  }

  async getImageScanFindings(repositoryName, imageTag) {
    try {
      const response = await ecr.describeImageScanFindings({
        repositoryName: repositoryName,
        imageId: { imageTag: imageTag }
      }).promise();
      
      const findings = response.imageScanFindings.findings || [];
      
      return {
        total: findings.length,
        critical: findings.filter(f => f.severity === 'CRITICAL').length,
        high: findings.filter(f => f.severity === 'HIGH').length,
        medium: findings.filter(f => f.severity === 'MEDIUM').length,
        low: findings.filter(f => f.severity === 'LOW').length
      };
    } catch (error) {
      if (error.code === 'ScanNotFoundException') {
        return { total: 0, critical: 0, high: 0, medium: 0, low: 0 };
      }
      throw error;
    }
  }

  async checkInspectorFindings() {
    console.log('🔍 Checking Inspector findings...');
    
    try {
      const findings = await inspector.listFindings({
        maxResults: 100,
        filterCriteria: {
          severity: [
            { comparison: 'EQUALS', value: 'CRITICAL' },
            { comparison: 'EQUALS', value: 'HIGH' }
          ]
        }
      }).promise();
      
      this.inspectorResults = {
        total: findings.findings.length,
        critical: findings.findings.filter(f => f.severity === 'CRITICAL').length,
        high: findings.findings.filter(f => f.severity === 'HIGH').length
      };
      
      console.log(`📋 Inspector findings: ${this.inspectorResults.total} total (${this.inspectorResults.critical} critical, ${this.inspectorResults.high} high)`);
    } catch (error) {
      console.warn('⚠️  Could not retrieve Inspector findings:', error.message);
      this.inspectorResults = { total: 0, critical: 0, high: 0 };
    }
  }

  async checkSecretsRotation() {
    console.log('🔑 Checking secrets rotation status...');
    
    const secretNames = [
      `${process.env.PROJECT_NAME || 'strapi-platform'}-${process.env.ENVIRONMENT || 'production'}-jwt-keys`,
      `${process.env.PROJECT_NAME || 'strapi-platform'}-${process.env.ENVIRONMENT || 'production'}-api-keys`
    ];
    
    const rotationStatus = [];
    
    for (const secretName of secretNames) {
      try {
        const secret = await secretsManager.describeSecret({
          SecretId: secretName
        }).promise();
        
        const rotationEnabled = secret.RotationEnabled || false;
        const lastRotated = secret.LastRotatedDate;
        const daysSinceRotation = lastRotated ? 
          Math.floor((Date.now() - lastRotated.getTime()) / (1000 * 60 * 60 * 24)) : 
          null;
        
        rotationStatus.push({
          name: secretName,
          rotationEnabled,
          lastRotated,
          daysSinceRotation
        });
        
        // Alert if rotation is overdue
        if (daysSinceRotation && daysSinceRotation > 35) {
          console.warn(`⚠️  Secret ${secretName} rotation is overdue (${daysSinceRotation} days)`);
        }
      } catch (error) {
        console.warn(`⚠️  Could not check secret ${secretName}:`, error.message);
      }
    }
    
    this.secretsResults = rotationStatus;
  }

  logScanSummary(scanResults) {
    console.log('\n📊 ECR Scan Summary:');
    console.log('─'.repeat(80));
    
    let totalCritical = 0;
    let totalHigh = 0;
    let totalMedium = 0;
    
    for (const result of scanResults) {
      const { repository, tag, findings } = result;
      totalCritical += findings.critical;
      totalHigh += findings.high;
      totalMedium += findings.medium;
      
      console.log(`${repository}:${tag}`);
      console.log(`  Critical: ${findings.critical}, High: ${findings.high}, Medium: ${findings.medium}, Total: ${findings.total}`);
    }
    
    console.log('─'.repeat(80));
    console.log(`Total across all images: Critical: ${totalCritical}, High: ${totalHigh}, Medium: ${totalMedium}`);
    
    if (totalCritical > 0) {
      console.log('🚨 CRITICAL vulnerabilities found! Immediate action required.');
    } else if (totalHigh > 0) {
      console.log('⚠️  HIGH severity vulnerabilities found. Please review and remediate.');
    } else {
      console.log('✅ No critical or high severity vulnerabilities found.');
    }
  }

  async generateSecurityReport() {
    console.log('📝 Generating security report...');
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        ecr: this.ecrResults,
        inspector: this.inspectorResults,
        secrets: this.secretsResults
      },
      recommendations: this.generateRecommendations()
    };
    
    // Write report to file
    const reportPath = path.join(process.cwd(), 'security-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`📄 Security report saved to: ${reportPath}`);
    
    // Generate markdown report
    const markdownReport = this.generateMarkdownReport(report);
    const markdownPath = path.join(process.cwd(), 'security-report.md');
    fs.writeFileSync(markdownPath, markdownReport);
    
    console.log(`📄 Markdown report saved to: ${markdownPath}`);
  }

  generateRecommendations() {
    const recommendations = [];
    
    // ECR recommendations
    const totalCritical = this.ecrResults.reduce((sum, r) => sum + r.findings.critical, 0);
    const totalHigh = this.ecrResults.reduce((sum, r) => sum + r.findings.high, 0);
    
    if (totalCritical > 0) {
      recommendations.push({
        priority: 'CRITICAL',
        category: 'Container Security',
        description: `${totalCritical} critical vulnerabilities found in container images`,
        action: 'Update base images and dependencies immediately'
      });
    }
    
    if (totalHigh > 0) {
      recommendations.push({
        priority: 'HIGH',
        category: 'Container Security',
        description: `${totalHigh} high severity vulnerabilities found in container images`,
        action: 'Schedule remediation within 7 days'
      });
    }
    
    // Inspector recommendations
    if (this.inspectorResults.critical > 0) {
      recommendations.push({
        priority: 'CRITICAL',
        category: 'Runtime Security',
        description: `${this.inspectorResults.critical} critical runtime vulnerabilities found`,
        action: 'Review Inspector findings and patch affected systems'
      });
    }
    
    // Secrets recommendations
    for (const secret of this.secretsResults) {
      if (!secret.rotationEnabled) {
        recommendations.push({
          priority: 'MEDIUM',
          category: 'Secrets Management',
          description: `Secret ${secret.name} does not have automatic rotation enabled`,
          action: 'Enable automatic rotation for this secret'
        });
      }
      
      if (secret.daysSinceRotation && secret.daysSinceRotation > 35) {
        recommendations.push({
          priority: 'HIGH',
          category: 'Secrets Management',
          description: `Secret ${secret.name} has not been rotated for ${secret.daysSinceRotation} days`,
          action: 'Rotate secret immediately'
        });
      }
    }
    
    return recommendations;
  }

  generateMarkdownReport(report) {
    const { summary, recommendations } = report;
    
    let markdown = `# Security Monitoring Report\n\n`;
    markdown += `**Generated:** ${report.timestamp}\n\n`;
    
    // ECR Summary
    markdown += `## Container Image Security (ECR)\n\n`;
    markdown += `| Repository | Tag | Critical | High | Medium | Total |\n`;
    markdown += `|------------|-----|----------|------|--------| ----- |\n`;
    
    for (const result of summary.ecr) {
      const { repository, tag, findings } = result;
      markdown += `| ${repository} | ${tag} | ${findings.critical} | ${findings.high} | ${findings.medium} | ${findings.total} |\n`;
    }
    
    // Inspector Summary
    markdown += `\n## Runtime Security (Inspector)\n\n`;
    markdown += `- **Total Findings:** ${summary.inspector.total}\n`;
    markdown += `- **Critical:** ${summary.inspector.critical}\n`;
    markdown += `- **High:** ${summary.inspector.high}\n`;
    
    // Secrets Summary
    markdown += `\n## Secrets Management\n\n`;
    markdown += `| Secret Name | Rotation Enabled | Days Since Rotation |\n`;
    markdown += `|-------------|------------------|--------------------|\n`;
    
    for (const secret of summary.secrets) {
      markdown += `| ${secret.name} | ${secret.rotationEnabled ? '✅' : '❌'} | ${secret.daysSinceRotation || 'N/A'} |\n`;
    }
    
    // Recommendations
    if (recommendations.length > 0) {
      markdown += `\n## Recommendations\n\n`;
      
      const criticalRecs = recommendations.filter(r => r.priority === 'CRITICAL');
      const highRecs = recommendations.filter(r => r.priority === 'HIGH');
      const mediumRecs = recommendations.filter(r => r.priority === 'MEDIUM');
      
      if (criticalRecs.length > 0) {
        markdown += `### 🚨 Critical Priority\n\n`;
        for (const rec of criticalRecs) {
          markdown += `- **${rec.category}:** ${rec.description}\n`;
          markdown += `  - *Action:* ${rec.action}\n\n`;
        }
      }
      
      if (highRecs.length > 0) {
        markdown += `### ⚠️ High Priority\n\n`;
        for (const rec of highRecs) {
          markdown += `- **${rec.category}:** ${rec.description}\n`;
          markdown += `  - *Action:* ${rec.action}\n\n`;
        }
      }
      
      if (mediumRecs.length > 0) {
        markdown += `### 📋 Medium Priority\n\n`;
        for (const rec of mediumRecs) {
          markdown += `- **${rec.category}:** ${rec.description}\n`;
          markdown += `  - *Action:* ${rec.action}\n\n`;
        }
      }
    } else {
      markdown += `\n## ✅ No Security Issues Found\n\nAll security checks passed successfully.\n`;
    }
    
    return markdown;
  }
}

// Run the security monitor
if (require.main === module) {
  const monitor = new SecurityMonitor();
  monitor.run().catch(error => {
    console.error('Security monitoring failed:', error);
    process.exit(1);
  });
}

module.exports = SecurityMonitor;