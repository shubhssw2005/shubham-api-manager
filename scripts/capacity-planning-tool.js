#!/usr/bin/env node

/**
 * Capacity Planning Tool
 * 
 * This tool analyzes system performance data and provides capacity planning
 * recommendations for the AWS deployment system.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

class CapacityPlanningTool {
  constructor() {
    this.config = {
      // SLO targets from requirements
      sloTargets: {
        availability: 0.9999, // 99.99%
        p99Latency: 500, // ms
        p95Latency: 200, // ms
        errorRate: 0.01, // 1%
        throughput: 100000 // RPS target
      },
      
      // Resource limits and costs
      resources: {
        cpu: {
          limit: 80, // 80% utilization target
          cost: 0.0464, // per vCPU hour (m5.large)
          scalingFactor: 1.5
        },
        memory: {
          limit: 75, // 75% utilization target
          cost: 0.0058, // per GB hour
          scalingFactor: 1.3
        },
        storage: {
          iopsLimit: 3000, // IOPS per volume
          cost: 0.125, // per GB month (gp3)
          scalingFactor: 2.0
        },
        network: {
          bandwidthLimit: 10000, // Mbps
          cost: 0.09, // per GB transfer
          scalingFactor: 1.8
        }
      },
      
      // Growth projections
      growth: {
        userGrowth: 0.15, // 15% monthly
        dataGrowth: 0.25, // 25% monthly
        trafficGrowth: 0.20, // 20% monthly
        featureComplexity: 0.05 // 5% monthly
      }
    };
    
    this.metrics = {
      historical: [],
      current: {},
      projections: {}
    };
  }

  async run() {
    console.log('🚀 Starting Capacity Planning Analysis...\n');
    
    try {
      // Step 1: Collect current metrics
      await this.collectCurrentMetrics();
      
      // Step 2: Analyze historical data
      await this.analyzeHistoricalData();
      
      // Step 3: Generate load projections
      await this.generateLoadProjections();
      
      // Step 4: Calculate resource requirements
      await this.calculateResourceRequirements();
      
      // Step 5: Generate cost projections
      await this.generateCostProjections();
      
      // Step 6: Create scaling recommendations
      await this.createScalingRecommendations();
      
      // Step 7: Generate capacity plan
      await this.generateCapacityPlan();
      
      console.log('✅ Capacity planning analysis completed successfully!');
      
    } catch (error) {
      console.error('❌ Error during capacity planning:', error.message);
      process.exit(1);
    }
  }

  async collectCurrentMetrics() {
    console.log('📊 Collecting current system metrics...');
    
    try {
      // Collect metrics from Prometheus/CloudWatch
      const metricsQueries = [
        'rate(http_requests_total[5m])', // Request rate
        'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))', // P95 latency
        'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))', // P99 latency
        'rate(http_requests_total{status=~"5.."}[5m])', // Error rate
        'node_cpu_seconds_total', // CPU usage
        'node_memory_MemAvailable_bytes', // Memory usage
        'node_filesystem_avail_bytes', // Disk usage
        'node_network_receive_bytes_total', // Network usage
      ];
      
      this.metrics.current = {
        timestamp: Date.now(),
        requestRate: await this.queryMetric(metricsQueries[0]) || 1000,
        p95Latency: await this.queryMetric(metricsQueries[1]) || 150,
        p99Latency: await this.queryMetric(metricsQueries[2]) || 300,
        errorRate: await this.queryMetric(metricsQueries[3]) || 0.005,
        cpuUtilization: await this.queryMetric(metricsQueries[4]) || 45,
        memoryUtilization: await this.queryMetric(metricsQueries[5]) || 60,
        diskUtilization: await this.queryMetric(metricsQueries[6]) || 40,
        networkUtilization: await this.queryMetric(metricsQueries[7]) || 30
      };
      
      console.log('Current Metrics:', this.metrics.current);
      
    } catch (error) {
      console.warn('⚠️  Could not collect live metrics, using defaults:', error.message);
      
      // Use default values for testing
      this.metrics.current = {
        timestamp: Date.now(),
        requestRate: 1000,
        p95Latency: 150,
        p99Latency: 300,
        errorRate: 0.005,
        cpuUtilization: 45,
        memoryUtilization: 60,
        diskUtilization: 40,
        networkUtilization: 30
      };
    }
  }

  async analyzeHistoricalData() {
    console.log('📈 Analyzing historical performance data...');
    
    // Load historical data from files or metrics store
    const historicalDataPath = 'tests/performance/historical-data.json';
    
    try {
      if (fs.existsSync(historicalDataPath)) {
        const data = JSON.parse(fs.readFileSync(historicalDataPath, 'utf8'));
        this.metrics.historical = data.metrics || [];
      }
    } catch (error) {
      console.warn('⚠️  No historical data found, generating sample data');
      this.generateSampleHistoricalData();
    }
    
    // Analyze trends
    const trends = this.analyzeTrends();
    console.log('Performance Trends:', trends);
    
    return trends;
  }

  generateSampleHistoricalData() {
    // Generate 30 days of sample data
    const days = 30;
    const baseDate = Date.now() - (days * 24 * 60 * 60 * 1000);
    
    for (let i = 0; i < days; i++) {
      const date = new Date(baseDate + (i * 24 * 60 * 60 * 1000));
      const growth = 1 + (i * 0.02); // 2% daily growth
      
      this.metrics.historical.push({
        timestamp: date.getTime(),
        requestRate: 800 * growth + (Math.random() * 200),
        p95Latency: 120 + (Math.random() * 60),
        p99Latency: 250 + (Math.random() * 100),
        errorRate: 0.003 + (Math.random() * 0.004),
        cpuUtilization: 35 + (growth * 5) + (Math.random() * 10),
        memoryUtilization: 50 + (growth * 3) + (Math.random() * 15),
        diskUtilization: 30 + (growth * 2) + (Math.random() * 10),
        networkUtilization: 20 + (growth * 4) + (Math.random() * 15)
      });
    }
  }

  analyzeTrends() {
    if (this.metrics.historical.length < 2) {
      return { growth: 0, trend: 'stable' };
    }
    
    const recent = this.metrics.historical.slice(-7); // Last 7 days
    const older = this.metrics.historical.slice(-14, -7); // Previous 7 days
    
    const recentAvg = this.calculateAverage(recent, 'requestRate');
    const olderAvg = this.calculateAverage(older, 'requestRate');
    
    const growth = (recentAvg - olderAvg) / olderAvg;
    
    return {
      growth,
      trend: growth > 0.1 ? 'growing' : growth < -0.1 ? 'declining' : 'stable',
      requestRateGrowth: growth,
      latencyTrend: this.calculateTrend(recent, 'p95Latency'),
      errorRateTrend: this.calculateTrend(recent, 'errorRate'),
      resourceUtilizationTrend: {
        cpu: this.calculateTrend(recent, 'cpuUtilization'),
        memory: this.calculateTrend(recent, 'memoryUtilization'),
        disk: this.calculateTrend(recent, 'diskUtilization'),
        network: this.calculateTrend(recent, 'networkUtilization')
      }
    };
  }

  async generateLoadProjections() {
    console.log('🔮 Generating load projections...');
    
    const currentLoad = this.metrics.current.requestRate;
    const projectionPeriods = [1, 3, 6, 12]; // months
    
    this.metrics.projections = {};
    
    for (const months of projectionPeriods) {
      const userGrowthFactor = Math.pow(1 + this.config.growth.userGrowth, months);
      const trafficGrowthFactor = Math.pow(1 + this.config.growth.trafficGrowth, months);
      const complexityFactor = Math.pow(1 + this.config.growth.featureComplexity, months);
      
      const projectedLoad = currentLoad * userGrowthFactor * trafficGrowthFactor * complexityFactor;
      
      this.metrics.projections[`${months}months`] = {
        requestRate: projectedLoad,
        peakRequestRate: projectedLoad * 3, // 3x peak factor
        dataVolume: this.calculateDataVolumeProjection(months),
        userCount: this.calculateUserCountProjection(months),
        storageRequirements: this.calculateStorageProjection(months)
      };
    }
    
    console.log('Load Projections:', this.metrics.projections);
  }

  async calculateResourceRequirements() {
    console.log('💻 Calculating resource requirements...');
    
    const requirements = {};
    
    for (const [period, projection] of Object.entries(this.metrics.projections)) {
      const peakRPS = projection.peakRequestRate;
      
      // CPU requirements (assuming 1000 RPS per vCPU at 80% utilization)
      const requiredCPUs = Math.ceil(peakRPS / (1000 * (this.config.resources.cpu.limit / 100)));
      
      // Memory requirements (assuming 2GB per 1000 RPS)
      const requiredMemoryGB = Math.ceil((peakRPS / 1000) * 2 / (this.config.resources.memory.limit / 100));
      
      // Storage requirements
      const requiredStorageGB = projection.storageRequirements;
      const requiredIOPS = Math.ceil(peakRPS * 0.1); // 0.1 IOPS per request
      
      // Network requirements
      const requiredBandwidthMbps = Math.ceil(peakRPS * 0.05); // 50KB per request
      
      requirements[period] = {
        compute: {
          vCPUs: requiredCPUs,
          memoryGB: requiredMemoryGB,
          instances: Math.ceil(requiredCPUs / 4), // 4 vCPUs per instance
          instanceType: this.recommendInstanceType(requiredCPUs, requiredMemoryGB)
        },
        storage: {
          volumeGB: requiredStorageGB,
          iops: requiredIOPS,
          volumes: Math.ceil(requiredIOPS / this.config.resources.storage.iopsLimit)
        },
        network: {
          bandwidthMbps: requiredBandwidthMbps,
          dataTransferGB: projection.dataVolume * 0.1 // 10% of data volume
        },
        database: {
          readReplicas: Math.ceil(peakRPS / 10000), // 1 replica per 10k RPS
          connectionPoolSize: Math.ceil(peakRPS / 100), // 1 connection per 100 RPS
          shards: Math.ceil(projection.userCount / 100000) // 1 shard per 100k users
        }
      };
    }
    
    this.resourceRequirements = requirements;
    console.log('Resource Requirements:', requirements);
  }

  async generateCostProjections() {
    console.log('💰 Generating cost projections...');
    
    const costProjections = {};
    
    for (const [period, requirements] of Object.entries(this.resourceRequirements)) {
      const monthlyHours = 730; // Average hours per month
      
      const computeCost = requirements.compute.vCPUs * this.config.resources.cpu.cost * monthlyHours;
      const memoryCost = requirements.compute.memoryGB * this.config.resources.memory.cost * monthlyHours;
      const storageCost = requirements.storage.volumeGB * this.config.resources.storage.cost;
      const networkCost = requirements.network.dataTransferGB * this.config.resources.network.cost;
      
      // Additional AWS service costs
      const eksCost = requirements.compute.instances * 0.10 * monthlyHours; // EKS cluster cost
      const albCost = 22.50; // Application Load Balancer
      const natGatewayCost = 45.00; // NAT Gateway
      const rdsProxyCost = 15.00; // RDS Proxy
      
      const totalMonthlyCost = computeCost + memoryCost + storageCost + networkCost + 
                              eksCost + albCost + natGatewayCost + rdsProxyCost;
      
      costProjections[period] = {
        breakdown: {
          compute: computeCost,
          memory: memoryCost,
          storage: storageCost,
          network: networkCost,
          services: eksCost + albCost + natGatewayCost + rdsProxyCost
        },
        totalMonthly: totalMonthlyCost,
        totalAnnual: totalMonthlyCost * 12,
        costPerRequest: totalMonthlyCost / (this.metrics.projections[period].requestRate * monthlyHours * 3600)
      };
    }
    
    this.costProjections = costProjections;
    console.log('Cost Projections:', costProjections);
  }

  async createScalingRecommendations() {
    console.log('📋 Creating scaling recommendations...');
    
    const recommendations = {
      immediate: [],
      shortTerm: [], // 1-3 months
      longTerm: []   // 6-12 months
    };
    
    // Analyze current utilization
    const current = this.metrics.current;
    
    if (current.cpuUtilization > this.config.resources.cpu.limit) {
      recommendations.immediate.push({
        type: 'scale_up',
        resource: 'CPU',
        action: 'Add more compute instances',
        priority: 'high',
        impact: 'Prevents performance degradation'
      });
    }
    
    if (current.memoryUtilization > this.config.resources.memory.limit) {
      recommendations.immediate.push({
        type: 'scale_up',
        resource: 'Memory',
        action: 'Upgrade to memory-optimized instances',
        priority: 'high',
        impact: 'Prevents out-of-memory errors'
      });
    }
    
    // Short-term recommendations based on 3-month projections
    const shortTermReq = this.resourceRequirements['3months'];
    if (shortTermReq.compute.instances > 10) {
      recommendations.shortTerm.push({
        type: 'architecture',
        resource: 'Compute',
        action: 'Implement auto-scaling groups',
        priority: 'medium',
        impact: 'Handles traffic spikes efficiently'
      });
    }
    
    if (shortTermReq.database.shards > 1) {
      recommendations.shortTerm.push({
        type: 'database',
        resource: 'Database',
        action: 'Implement database sharding',
        priority: 'medium',
        impact: 'Improves database performance and scalability'
      });
    }
    
    // Long-term recommendations based on 12-month projections
    const longTermReq = this.resourceRequirements['12months'];
    if (longTermReq.compute.instances > 50) {
      recommendations.longTerm.push({
        type: 'architecture',
        resource: 'Infrastructure',
        action: 'Consider multi-region deployment',
        priority: 'low',
        impact: 'Improves global performance and disaster recovery'
      });
    }
    
    if (this.costProjections['12months'].totalMonthly > 50000) {
      recommendations.longTerm.push({
        type: 'cost_optimization',
        resource: 'All',
        action: 'Implement reserved instances and spot instances',
        priority: 'medium',
        impact: 'Reduces infrastructure costs by 30-60%'
      });
    }
    
    this.scalingRecommendations = recommendations;
    console.log('Scaling Recommendations:', recommendations);
  }

  async generateCapacityPlan() {
    console.log('📄 Generating capacity plan document...');
    
    const plan = {
      generatedAt: new Date().toISOString(),
      summary: {
        currentLoad: this.metrics.current.requestRate,
        projectedPeakLoad: this.metrics.projections['12months'].peakRequestRate,
        growthRate: this.analyzeTrends().requestRateGrowth,
        estimatedAnnualCost: this.costProjections['12months'].totalAnnual
      },
      currentState: this.metrics.current,
      projections: this.metrics.projections,
      resourceRequirements: this.resourceRequirements,
      costProjections: this.costProjections,
      recommendations: this.scalingRecommendations,
      riskAssessment: this.generateRiskAssessment(),
      actionPlan: this.generateActionPlan()
    };
    
    // Save capacity plan
    const outputPath = 'tests/performance/capacity-plan.json';
    fs.writeFileSync(outputPath, JSON.stringify(plan, null, 2));
    
    // Generate human-readable report
    const reportPath = 'tests/performance/capacity-plan-report.md';
    const report = this.generateMarkdownReport(plan);
    fs.writeFileSync(reportPath, report);
    
    console.log(`📊 Capacity plan saved to: ${outputPath}`);
    console.log(`📋 Human-readable report saved to: ${reportPath}`);
    
    return plan;
  }

  // Helper methods
  async queryMetric(query) {
    // This would integrate with actual monitoring systems
    // For now, return simulated values
    return Math.random() * 100;
  }

  calculateAverage(data, field) {
    if (!data || data.length === 0) return 0;
    const sum = data.reduce((acc, item) => acc + (item[field] || 0), 0);
    return sum / data.length;
  }

  calculateTrend(data, field) {
    if (!data || data.length < 2) return 0;
    const first = data[0][field] || 0;
    const last = data[data.length - 1][field] || 0;
    return (last - first) / first;
  }

  calculateDataVolumeProjection(months) {
    const currentVolume = 1000; // GB
    const growthFactor = Math.pow(1 + this.config.growth.dataGrowth, months);
    return Math.ceil(currentVolume * growthFactor);
  }

  calculateUserCountProjection(months) {
    const currentUsers = 10000;
    const growthFactor = Math.pow(1 + this.config.growth.userGrowth, months);
    return Math.ceil(currentUsers * growthFactor);
  }

  calculateStorageProjection(months) {
    const baseStorage = 500; // GB
    const dataGrowth = this.calculateDataVolumeProjection(months);
    const mediaGrowth = dataGrowth * 2; // Media files grow faster
    return baseStorage + dataGrowth + mediaGrowth;
  }

  recommendInstanceType(cpus, memoryGB) {
    const ratio = memoryGB / cpus;
    
    if (ratio > 8) {
      return 'r5.large'; // Memory optimized
    } else if (ratio < 2) {
      return 'c5.large'; // Compute optimized
    } else {
      return 'm5.large'; // General purpose
    }
  }

  generateRiskAssessment() {
    const risks = [];
    
    // Performance risks
    if (this.metrics.current.p99Latency > this.config.sloTargets.p99Latency * 0.8) {
      risks.push({
        type: 'performance',
        severity: 'medium',
        description: 'P99 latency approaching SLO threshold',
        mitigation: 'Scale compute resources or optimize application performance'
      });
    }
    
    // Capacity risks
    if (this.metrics.current.cpuUtilization > 70) {
      risks.push({
        type: 'capacity',
        severity: 'high',
        description: 'CPU utilization high, limited headroom for traffic spikes',
        mitigation: 'Add more compute capacity immediately'
      });
    }
    
    // Cost risks
    const yearlyGrowth = this.costProjections['12months'].totalAnnual / this.costProjections['1months'].totalMonthly / 12;
    if (yearlyGrowth > 3) {
      risks.push({
        type: 'cost',
        severity: 'medium',
        description: 'Rapid cost growth projected',
        mitigation: 'Implement cost optimization strategies'
      });
    }
    
    return risks;
  }

  generateActionPlan() {
    const actions = [];
    
    // Immediate actions (next 30 days)
    this.scalingRecommendations.immediate.forEach(rec => {
      actions.push({
        timeframe: 'immediate',
        action: rec.action,
        priority: rec.priority,
        owner: 'DevOps Team',
        estimatedEffort: '1-2 days'
      });
    });
    
    // Short-term actions (1-3 months)
    this.scalingRecommendations.shortTerm.forEach(rec => {
      actions.push({
        timeframe: 'short-term',
        action: rec.action,
        priority: rec.priority,
        owner: 'Engineering Team',
        estimatedEffort: '1-2 weeks'
      });
    });
    
    // Long-term actions (6-12 months)
    this.scalingRecommendations.longTerm.forEach(rec => {
      actions.push({
        timeframe: 'long-term',
        action: rec.action,
        priority: rec.priority,
        owner: 'Architecture Team',
        estimatedEffort: '1-3 months'
      });
    });
    
    return actions;
  }

  generateMarkdownReport(plan) {
    return `# Capacity Planning Report

Generated: ${plan.generatedAt}

## Executive Summary

- **Current Load**: ${plan.summary.currentLoad.toLocaleString()} RPS
- **Projected Peak Load (12 months)**: ${plan.summary.projectedPeakLoad.toLocaleString()} RPS
- **Growth Rate**: ${(plan.summary.growthRate * 100).toFixed(1)}% monthly
- **Estimated Annual Cost**: $${plan.summary.estimatedAnnualCost.toLocaleString()}

## Current System State

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Request Rate | ${plan.currentState.requestRate} RPS | - | ✅ |
| P95 Latency | ${plan.currentState.p95Latency}ms | <200ms | ${plan.currentState.p95Latency < 200 ? '✅' : '⚠️'} |
| P99 Latency | ${plan.currentState.p99Latency}ms | <500ms | ${plan.currentState.p99Latency < 500 ? '✅' : '⚠️'} |
| Error Rate | ${(plan.currentState.errorRate * 100).toFixed(2)}% | <1% | ${plan.currentState.errorRate < 0.01 ? '✅' : '⚠️'} |
| CPU Utilization | ${plan.currentState.cpuUtilization}% | <80% | ${plan.currentState.cpuUtilization < 80 ? '✅' : '⚠️'} |
| Memory Utilization | ${plan.currentState.memoryUtilization}% | <75% | ${plan.currentState.memoryUtilization < 75 ? '✅' : '⚠️'} |

## Load Projections

| Period | Request Rate | Peak Rate | Storage (GB) | Est. Monthly Cost |
|--------|-------------|-----------|--------------|-------------------|
| 1 Month | ${Math.round(plan.projections['1months'].requestRate).toLocaleString()} | ${Math.round(plan.projections['1months'].peakRequestRate).toLocaleString()} | ${plan.projections['1months'].storageRequirements.toLocaleString()} | $${Math.round(plan.costProjections['1months'].totalMonthly).toLocaleString()} |
| 3 Months | ${Math.round(plan.projections['3months'].requestRate).toLocaleString()} | ${Math.round(plan.projections['3months'].peakRequestRate).toLocaleString()} | ${plan.projections['3months'].storageRequirements.toLocaleString()} | $${Math.round(plan.costProjections['3months'].totalMonthly).toLocaleString()} |
| 6 Months | ${Math.round(plan.projections['6months'].requestRate).toLocaleString()} | ${Math.round(plan.projections['6months'].peakRequestRate).toLocaleString()} | ${plan.projections['6months'].storageRequirements.toLocaleString()} | $${Math.round(plan.costProjections['6months'].totalMonthly).toLocaleString()} |
| 12 Months | ${Math.round(plan.projections['12months'].requestRate).toLocaleString()} | ${Math.round(plan.projections['12months'].peakRequestRate).toLocaleString()} | ${plan.projections['12months'].storageRequirements.toLocaleString()} | $${Math.round(plan.costProjections['12months'].totalMonthly).toLocaleString()} |

## Immediate Actions Required

${plan.recommendations.immediate.map(rec => `- **${rec.priority.toUpperCase()}**: ${rec.action} (${rec.impact})`).join('\n')}

## Risk Assessment

${plan.riskAssessment.map(risk => `- **${risk.severity.toUpperCase()} ${risk.type.toUpperCase()} RISK**: ${risk.description}\n  - *Mitigation*: ${risk.mitigation}`).join('\n\n')}

## Detailed Action Plan

### Immediate (Next 30 Days)
${plan.actionPlan.filter(a => a.timeframe === 'immediate').map(a => `- ${a.action} (${a.priority} priority, ${a.estimatedEffort})`).join('\n')}

### Short-term (1-3 Months)
${plan.actionPlan.filter(a => a.timeframe === 'short-term').map(a => `- ${a.action} (${a.priority} priority, ${a.estimatedEffort})`).join('\n')}

### Long-term (6-12 Months)
${plan.actionPlan.filter(a => a.timeframe === 'long-term').map(a => `- ${a.action} (${a.priority} priority, ${a.estimatedEffort})`).join('\n')}

---

*This report was generated automatically by the Capacity Planning Tool. Review and validate all recommendations before implementation.*
`;
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const tool = new CapacityPlanningTool();
  tool.run().catch(console.error);
}

export default CapacityPlanningTool;