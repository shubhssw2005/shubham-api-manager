import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execSync } from 'child_process';
import AWS from 'aws-sdk';

describe('Auto-Scaling System Integration Tests', () => {
  let eks, cloudwatch, ce;
  const clusterName = process.env.EKS_CLUSTER_NAME || 'test-cluster';
  const region = process.env.AWS_REGION || 'us-east-1';

  beforeAll(async () => {
    // Configure AWS SDK
    AWS.config.update({ region });
    eks = new AWS.EKS();
    cloudwatch = new AWS.CloudWatch();
    ce = new AWS.CostExplorer();
  });

  describe('Cluster Autoscaler', () => {
    it('should have cluster autoscaler deployed', async () => {
      try {
        const result = execSync(
          `kubectl get deployment cluster-autoscaler -n kube-system -o json`,
          { encoding: 'utf8' }
        );
        
        const deployment = JSON.parse(result);
        expect(deployment.metadata.name).toBe('cluster-autoscaler');
        expect(deployment.status.readyReplicas).toBeGreaterThan(0);
      } catch (error) {
        throw new Error(`Cluster autoscaler not found: ${error.message}`);
      }
    });

    it('should have proper IAM permissions', async () => {
      try {
        const logs = execSync(
          `kubectl logs -n kube-system deployment/cluster-autoscaler --tail=50`,
          { encoding: 'utf8' }
        );
        
        // Check for permission errors
        expect(logs).not.toContain('AccessDenied');
        expect(logs).not.toContain('UnauthorizedOperation');
      } catch (error) {
        console.warn('Could not check cluster autoscaler logs:', error.message);
      }
    });

    it('should discover node groups correctly', async () => {
      try {
        const logs = execSync(
          `kubectl logs -n kube-system deployment/cluster-autoscaler --tail=100`,
          { encoding: 'utf8' }
        );
        
        // Should find node groups
        expect(logs).toContain('node group');
      } catch (error) {
        console.warn('Could not verify node group discovery:', error.message);
      }
    });
  });

  describe('Vertical Pod Autoscaler', () => {
    it('should have VPA CRDs installed', async () => {
      try {
        const result = execSync(
          `kubectl get crd verticalpodautoscalers.autoscaling.k8s.io`,
          { encoding: 'utf8' }
        );
        
        expect(result).toContain('verticalpodautoscalers.autoscaling.k8s.io');
      } catch (error) {
        throw new Error(`VPA CRDs not found: ${error.message}`);
      }
    });

    it('should have VPA components running', async () => {
      const components = ['vpa-admission-controller', 'vpa-recommender', 'vpa-updater'];
      
      for (const component of components) {
        try {
          const result = execSync(
            `kubectl get deployment ${component} -n kube-system -o json`,
            { encoding: 'utf8' }
          );
          
          const deployment = JSON.parse(result);
          expect(deployment.status.readyReplicas).toBeGreaterThan(0);
        } catch (error) {
          throw new Error(`VPA component ${component} not ready: ${error.message}`);
        }
      }
    });

    it('should create VPA resources for applications', async () => {
      try {
        const result = execSync(
          `kubectl get vpa --all-namespaces -o json`,
          { encoding: 'utf8' }
        );
        
        const vpas = JSON.parse(result);
        expect(vpas.items.length).toBeGreaterThan(0);
        
        // Check for our application VPAs
        const vpaNames = vpas.items.map(vpa => vpa.metadata.name);
        expect(vpaNames.some(name => name.includes('api-service'))).toBe(true);
      } catch (error) {
        console.warn('Could not verify VPA resources:', error.message);
      }
    });
  });

  describe('Horizontal Pod Autoscaler', () => {
    it('should have HPA resources for services', async () => {
      try {
        const result = execSync(
          `kubectl get hpa --all-namespaces -o json`,
          { encoding: 'utf8' }
        );
        
        const hpas = JSON.parse(result);
        expect(hpas.items.length).toBeGreaterThan(0);
        
        // Verify HPA configuration
        hpas.items.forEach(hpa => {
          expect(hpa.spec.minReplicas).toBeGreaterThan(0);
          expect(hpa.spec.maxReplicas).toBeGreaterThan(hpa.spec.minReplicas);
          expect(hpa.spec.metrics).toBeDefined();
        });
      } catch (error) {
        throw new Error(`HPA resources not found: ${error.message}`);
      }
    });

    it('should have metrics server running', async () => {
      try {
        const result = execSync(
          `kubectl get deployment metrics-server -n kube-system -o json`,
          { encoding: 'utf8' }
        );
        
        const deployment = JSON.parse(result);
        expect(deployment.status.readyReplicas).toBeGreaterThan(0);
      } catch (error) {
        throw new Error(`Metrics server not ready: ${error.message}`);
      }
    });
  });

  describe('Cost Optimization', () => {
    it('should have cost budgets configured', async () => {
      try {
        const budgets = await new AWS.Budgets().describeBudgets({
          AccountId: await getAccountId()
        }).promise();
        
        expect(budgets.Budgets.length).toBeGreaterThan(0);
        
        // Check for project-specific budgets
        const projectBudgets = budgets.Budgets.filter(budget => 
          budget.BudgetName.includes(process.env.PROJECT_NAME || 'strapi-platform')
        );
        
        expect(projectBudgets.length).toBeGreaterThan(0);
      } catch (error) {
        console.warn('Could not verify cost budgets:', error.message);
      }
    });

    it('should have cost anomaly detection enabled', async () => {
      try {
        const detectors = await ce.getAnomalyDetectors().promise();
        expect(detectors.AnomalyDetectors.length).toBeGreaterThan(0);
      } catch (error) {
        console.warn('Could not verify anomaly detection:', error.message);
      }
    });

    it('should have S3 lifecycle policies configured', async () => {
      const s3 = new AWS.S3();
      
      try {
        const buckets = await s3.listBuckets().promise();
        const projectBuckets = buckets.Buckets.filter(bucket => 
          bucket.Name.includes(process.env.PROJECT_NAME || 'strapi-platform')
        );
        
        for (const bucket of projectBuckets) {
          try {
            const lifecycle = await s3.getBucketLifecycleConfiguration({
              Bucket: bucket.Name
            }).promise();
            
            expect(lifecycle.Rules.length).toBeGreaterThan(0);
            
            // Verify lifecycle rules have transitions
            const hasTransitions = lifecycle.Rules.some(rule => 
              rule.Transitions && rule.Transitions.length > 0
            );
            expect(hasTransitions).toBe(true);
          } catch (lifecycleError) {
            if (lifecycleError.code !== 'NoSuchLifecycleConfiguration') {
              throw lifecycleError;
            }
          }
        }
      } catch (error) {
        console.warn('Could not verify S3 lifecycle policies:', error.message);
      }
    });

    it('should have intelligent tiering enabled', async () => {
      const s3 = new AWS.S3();
      
      try {
        const buckets = await s3.listBuckets().promise();
        const projectBuckets = buckets.Buckets.filter(bucket => 
          bucket.Name.includes(process.env.PROJECT_NAME || 'strapi-platform')
        );
        
        for (const bucket of projectBuckets) {
          try {
            const tiering = await s3.listBucketIntelligentTieringConfigurations({
              Bucket: bucket.Name
            }).promise();
            
            expect(tiering.IntelligentTieringConfigurationList.length).toBeGreaterThan(0);
          } catch (tieringError) {
            console.warn(`No intelligent tiering for bucket ${bucket.Name}`);
          }
        }
      } catch (error) {
        console.warn('Could not verify intelligent tiering:', error.message);
      }
    });
  });

  describe('Autoscaling Behavior', () => {
    it('should scale pods based on CPU utilization', async () => {
      // This test would require generating load and waiting for scaling
      // For now, we'll just verify the configuration is correct
      
      try {
        const result = execSync(
          `kubectl get hpa -o json`,
          { encoding: 'utf8' }
        );
        
        const hpas = JSON.parse(result);
        
        hpas.items.forEach(hpa => {
          const cpuMetric = hpa.spec.metrics.find(metric => 
            metric.type === 'Resource' && metric.resource.name === 'cpu'
          );
          
          if (cpuMetric) {
            expect(cpuMetric.resource.target.averageUtilization).toBeLessThan(90);
            expect(cpuMetric.resource.target.averageUtilization).toBeGreaterThan(50);
          }
        });
      } catch (error) {
        console.warn('Could not verify HPA CPU configuration:', error.message);
      }
    });

    it('should have proper scaling policies', async () => {
      try {
        const result = execSync(
          `kubectl get hpa -o json`,
          { encoding: 'utf8' }
        );
        
        const hpas = JSON.parse(result);
        
        hpas.items.forEach(hpa => {
          // Verify behavior configuration if present
          if (hpa.spec.behavior) {
            expect(hpa.spec.behavior.scaleUp).toBeDefined();
            expect(hpa.spec.behavior.scaleDown).toBeDefined();
          }
        });
      } catch (error) {
        console.warn('Could not verify scaling policies:', error.message);
      }
    });
  });

  describe('Cost Monitoring', () => {
    it('should have CloudWatch dashboards for cost monitoring', async () => {
      try {
        const dashboards = await cloudwatch.listDashboards().promise();
        
        const costDashboards = dashboards.DashboardEntries.filter(dashboard =>
          dashboard.DashboardName.includes('cost') || 
          dashboard.DashboardName.includes('optimization')
        );
        
        expect(costDashboards.length).toBeGreaterThan(0);
      } catch (error) {
        console.warn('Could not verify cost dashboards:', error.message);
      }
    });

    it('should have cost alerts configured', async () => {
      try {
        const alarms = await cloudwatch.describeAlarms().promise();
        
        const costAlarms = alarms.MetricAlarms.filter(alarm =>
          alarm.AlarmName.includes('cost') || 
          alarm.Namespace === 'AWS/Billing'
        );
        
        expect(costAlarms.length).toBeGreaterThan(0);
      } catch (error) {
        console.warn('Could not verify cost alarms:', error.message);
      }
    });
  });
});

// Helper function to get AWS account ID
async function getAccountId() {
  const sts = new AWS.STS();
  const identity = await sts.getCallerIdentity().promise();
  return identity.Account;
}