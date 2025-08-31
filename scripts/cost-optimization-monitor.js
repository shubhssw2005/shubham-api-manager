#!/usr/bin/env node

/**
 * Cost Optimization Monitoring Script
 * Monitors AWS costs and provides optimization recommendations
 */

const AWS = require('aws-sdk');
const fs = require('fs');
const path = require('path');

// Configure AWS SDK
AWS.config.update({
  region: process.env.AWS_REGION || 'us-east-1'
});

const ce = new AWS.CostExplorer();
const cloudwatch = new AWS.CloudWatch();
const s3 = new AWS.S3();
const ec2 = new AWS.EC2();

class CostOptimizationMonitor {
  constructor() {
    this.projectName = process.env.PROJECT_NAME || 'strapi-platform';
    this.environment = process.env.ENVIRONMENT || 'production';
  }

  async getCostAndUsage(days = 7) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);

    const params = {
      TimePeriod: {
        Start: startDate.toISOString().split('T')[0],
        End: endDate.toISOString().split('T')[0]
      },
      Granularity: 'DAILY',
      Metrics: ['BlendedCost', 'UsageQuantity'],
      GroupBy: [
        {
          Type: 'DIMENSION',
          Key: 'SERVICE'
        }
      ]
    };

    try {
      const result = await ce.getCostAndUsage(params).promise();
      return result.ResultsByTime;
    } catch (error) {
      console.error('Error getting cost and usage:', error);
      throw error;
    }
  }

  async analyzeS3Costs() {
    console.log('🔍 Analyzing S3 costs and usage...');
    
    try {
      // List buckets
      const buckets = await s3.listBuckets().promise();
      const recommendations = [];

      for (const bucket of buckets.Buckets) {
        if (bucket.Name.includes(this.projectName)) {
          console.log(`  📦 Analyzing bucket: ${bucket.Name}`);

          // Check lifecycle configuration
          try {
            await s3.getBucketLifecycleConfiguration({ Bucket: bucket.Name }).promise();
            console.log(`    ✅ Lifecycle policy exists`);
          } catch (error) {
            if (error.code === 'NoSuchLifecycleConfiguration') {
              recommendations.push({
                type: 'S3_LIFECYCLE',
                resource: bucket.Name,
                description: 'No lifecycle policy configured. Consider adding lifecycle rules to reduce storage costs.',
                priority: 'Medium',
                estimatedSavings: '15-30%'
              });
              console.log(`    ⚠️  No lifecycle policy found`);
            }
          }

          // Check intelligent tiering
          try {
            await s3.listBucketIntelligentTieringConfigurations({ Bucket: bucket.Name }).promise();
            console.log(`    ✅ Intelligent tiering configured`);
          } catch (error) {
            recommendations.push({
              type: 'S3_INTELLIGENT_TIERING',
              resource: bucket.Name,
              description: 'Enable S3 Intelligent Tiering for automatic cost optimization.',
              priority: 'Low',
              estimatedSavings: '10-20%'
            });
            console.log(`    ⚠️  Intelligent tiering not configured`);
          }

          // Check bucket size and suggest optimization
          try {
            const metrics = await cloudwatch.getMetricStatistics({
              Namespace: 'AWS/S3',
              MetricName: 'BucketSizeBytes',
              Dimensions: [
                { Name: 'BucketName', Value: bucket.Name },
                { Name: 'StorageType', Value: 'StandardStorage' }
              ],
              StartTime: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
              EndTime: new Date(),
              Period: 86400,
              Statistics: ['Average']
            }).promise();

            if (metrics.Datapoints.length > 0) {
              const avgSize = metrics.Datapoints[metrics.Datapoints.length - 1].Average;
              const sizeGB = avgSize / (1024 * 1024 * 1024);
              console.log(`    📊 Average size: ${sizeGB.toFixed(2)} GB`);

              if (sizeGB > 100) {
                recommendations.push({
                  type: 'S3_STORAGE_CLASS',
                  resource: bucket.Name,
                  description: `Large bucket (${sizeGB.toFixed(2)} GB). Review storage classes and access patterns.`,
                  priority: 'High',
                  estimatedSavings: '20-50%'
                });
              }
            }
          } catch (error) {
            console.log(`    ⚠️  Could not get bucket metrics: ${error.message}`);
          }
        }
      }

      return recommendations;
    } catch (error) {
      console.error('Error analyzing S3 costs:', error);
      return [];
    }
  }

  async analyzeEC2Costs() {
    console.log('🔍 Analyzing EC2 costs and usage...');
    
    try {
      const instances = await ec2.describeInstances().promise();
      const recommendations = [];

      for (const reservation of instances.Reservations) {
        for (const instance of reservation.Instances) {
          if (instance.State.Name === 'running') {
            console.log(`  🖥️  Instance: ${instance.InstanceId} (${instance.InstanceType})`);

            // Check if instance has been running for a long time
            const launchTime = new Date(instance.LaunchTime);
            const runningDays = (Date.now() - launchTime.getTime()) / (1000 * 60 * 60 * 24);

            if (runningDays > 30) {
              recommendations.push({
                type: 'EC2_RESERVED_INSTANCE',
                resource: instance.InstanceId,
                description: `Instance running for ${Math.floor(runningDays)} days. Consider Reserved Instances for cost savings.`,
                priority: 'High',
                estimatedSavings: '30-60%'
              });
            }

            // Check for oversized instances (this would need CloudWatch metrics)
            recommendations.push({
              type: 'EC2_RIGHTSIZING',
              resource: instance.InstanceId,
              description: 'Review CPU and memory utilization to ensure proper sizing.',
              priority: 'Medium',
              estimatedSavings: '10-30%'
            });
          }
        }
      }

      return recommendations;
    } catch (error) {
      console.error('Error analyzing EC2 costs:', error);
      return [];
    }
  }

  async generateReport() {
    console.log(`📊 Generating cost optimization report for ${this.projectName} (${this.environment})`);
    console.log('=' * 60);

    try {
      // Get cost data
      const costData = await this.getCostAndUsage(7);
      
      // Calculate total cost
      let totalCost = 0;
      const serviceCosts = {};

      costData.forEach(day => {
        day.Groups.forEach(group => {
          const service = group.Keys[0];
          const cost = parseFloat(group.Metrics.BlendedCost.Amount);
          
          if (!serviceCosts[service]) {
            serviceCosts[service] = 0;
          }
          serviceCosts[service] += cost;
          totalCost += cost;
        });
      });

      console.log(`💰 Total 7-day cost: $${totalCost.toFixed(2)}`);
      console.log('\n📈 Top services by cost:');
      
      const sortedServices = Object.entries(serviceCosts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);

      sortedServices.forEach(([service, cost], index) => {
        console.log(`  ${index + 1}. ${service}: $${cost.toFixed(2)}`);
      });

      // Get recommendations
      console.log('\n🔧 Cost Optimization Recommendations:');
      
      const s3Recommendations = await this.analyzeS3Costs();
      const ec2Recommendations = await this.analyzeEC2Costs();
      
      const allRecommendations = [...s3Recommendations, ...ec2Recommendations];

      if (allRecommendations.length === 0) {
        console.log('  ✅ No immediate optimization opportunities found.');
      } else {
        allRecommendations.forEach((rec, index) => {
          console.log(`\n  ${index + 1}. ${rec.type} - ${rec.priority} Priority`);
          console.log(`     Resource: ${rec.resource}`);
          console.log(`     Description: ${rec.description}`);
          console.log(`     Estimated Savings: ${rec.estimatedSavings}`);
        });
      }

      // Generate summary
      const report = {
        timestamp: new Date().toISOString(),
        project: this.projectName,
        environment: this.environment,
        totalCost: totalCost,
        topServices: sortedServices,
        recommendations: allRecommendations,
        summary: {
          totalRecommendations: allRecommendations.length,
          highPriority: allRecommendations.filter(r => r.priority === 'High').length,
          mediumPriority: allRecommendations.filter(r => r.priority === 'Medium').length,
          lowPriority: allRecommendations.filter(r => r.priority === 'Low').length
        }
      };

      // Save report
      const reportPath = path.join(__dirname, '..', 'reports', `cost-optimization-${Date.now()}.json`);
      
      // Ensure reports directory exists
      const reportsDir = path.dirname(reportPath);
      if (!fs.existsSync(reportsDir)) {
        fs.mkdirSync(reportsDir, { recursive: true });
      }

      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
      console.log(`\n📄 Report saved to: ${reportPath}`);

      return report;

    } catch (error) {
      console.error('Error generating report:', error);
      throw error;
    }
  }
}

// CLI execution
if (require.main === module) {
  const monitor = new CostOptimizationMonitor();
  
  monitor.generateReport()
    .then(report => {
      console.log('\n✅ Cost optimization analysis completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Error during cost optimization analysis:', error);
      process.exit(1);
    });
}

module.exports = CostOptimizationMonitor;