#!/usr/bin/env node

/**
 * Incident Response System Setup Script
 * Sets up the incident response infrastructure and configuration
 */

import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';

class IncidentResponseSetup {
  constructor() {
    this.baseDir = process.cwd();
    this.configDir = path.join(this.baseDir, 'config', 'incident-response');
    this.scriptsDir = path.join(this.baseDir, 'scripts', 'incident-response');
  }

  async setup() {
    console.log('🚨 Setting up Incident Response System...\n');

    try {
      await this.createDirectories();
      await this.createConfiguration();
      await this.createScripts();
      await this.setupKubernetesResources();
      await this.createMonitoringAlerts();
      await this.setupPagerDutyIntegration();
      
      console.log('✅ Incident Response System setup complete!\n');
      this.printNextSteps();
    } catch (error) {
      console.error('❌ Setup failed:', error.message);
      process.exit(1);
    }
  }

  async createDirectories() {
    console.log('📁 Creating directories...');
    
    const directories = [
      this.configDir,
      this.scriptsDir,
      'docs/runbooks',
      'docs/incident-response',
      'docs/incident-response/pirs',
      'k8s/incident-response'
    ];

    for (const dir of directories) {
      await fs.mkdir(dir, { recursive: true });
      console.log(`   Created: ${dir}`);
    }
  }

  async createConfiguration() {
    console.log('\n⚙️  Creating configuration files...');

    // Main incident response configuration
    const mainConfig = {
      pagerduty: {
        apiKey: process.env.PAGERDUTY_API_KEY || 'YOUR_PAGERDUTY_API_KEY',
        serviceKey: process.env.PAGERDUTY_SERVICE_KEY || 'YOUR_SERVICE_KEY',
        escalationKeys: {
          database: process.env.PAGERDUTY_DATABASE_ESCALATION_KEY || 'DATABASE_ESCALATION_KEY',
          platform: process.env.PAGERDUTY_PLATFORM_ESCALATION_KEY || 'PLATFORM_ESCALATION_KEY',
          security: process.env.PAGERDUTY_SECURITY_ESCALATION_KEY || 'SECURITY_ESCALATION_KEY'
        }
      },
      slack: {
        webhookUrl: process.env.SLACK_INCIDENT_WEBHOOK || 'YOUR_SLACK_WEBHOOK_URL',
        channel: '#incident-response'
      },
      alerting: {
        prometheus: {
          enabled: true,
          webhookUrl: '/webhooks/alerts/prometheus'
        },
        cloudwatch: {
          enabled: true,
          webhookUrl: '/webhooks/alerts/cloudwatch'
        }
      },
      escalation: {
        timeThresholds: {
          critical: 10 * 60 * 1000, // 10 minutes
          warning: 30 * 60 * 1000,  // 30 minutes
          info: 60 * 60 * 1000      // 60 minutes
        }
      }
    };

    await fs.writeFile(
      path.join(this.configDir, 'config.json'),
      JSON.stringify(mainConfig, null, 2)
    );
    console.log('   Created: config/incident-response/config.json');

    // Alert rules configuration
    const alertRules = {
      rules: [
        {
          name: 'database-down',
          alertName: 'DatabaseDown',
          severity: 'critical',
          service: 'database',
          runbook: 'https://runbooks.example.com/database-failover'
        },
        {
          name: 'api-high-latency',
          alertName: 'APIHighLatency',
          severity: 'warning',
          service: 'api',
          runbook: 'https://runbooks.example.com/api-service-degradation'
        },
        {
          name: 'cache-storm',
          alertName: 'CacheStorm',
          severity: 'warning',
          service: 'cache',
          runbook: 'https://runbooks.example.com/cache-storm-response'
        }
      ],
      suppressionRules: [
        {
          name: 'maintenance-window',
          timeWindow: 5 * 60 * 1000, // 5 minutes
          description: 'Suppress alerts during maintenance windows'
        }
      ]
    };

    await fs.writeFile(
      path.join(this.configDir, 'alert-rules.json'),
      JSON.stringify(alertRules, null, 2)
    );
    console.log('   Created: config/incident-response/alert-rules.json');
  }

  async createScripts() {
    console.log('\n📜 Creating operational scripts...');

    // Incident response startup script
    const startupScript = `#!/bin/bash

# Incident Response Service Startup Script

set -e

echo "🚨 Starting Incident Response Service..."

# Check environment variables
if [ -z "$PAGERDUTY_API_KEY" ]; then
    echo "❌ PAGERDUTY_API_KEY not set"
    exit 1
fi

if [ -z "$SLACK_INCIDENT_WEBHOOK" ]; then
    echo "⚠️  SLACK_INCIDENT_WEBHOOK not set - Slack notifications disabled"
fi

# Start the incident response service
export NODE_ENV=production
export PORT=3001

node lib/incident-response/IncidentResponseService.js

echo "✅ Incident Response Service started on port 3001"
`;

    await fs.writeFile(
      path.join(this.scriptsDir, 'start-incident-response.sh'),
      startupScript
    );
    await fs.chmod(path.join(this.scriptsDir, 'start-incident-response.sh'), 0o755);
    console.log('   Created: scripts/incident-response/start-incident-response.sh');

    // Emergency response script
    const emergencyScript = `#!/bin/bash

# Emergency Incident Response Script
# Use this script for immediate incident response

INCIDENT_TYPE=\${1:-"unknown"}
SEVERITY=\${2:-"warning"}
DESCRIPTION=\${3:-"Emergency incident triggered manually"}

echo "🚨 EMERGENCY INCIDENT RESPONSE ACTIVATED"
echo "Type: $INCIDENT_TYPE"
echo "Severity: $SEVERITY"
echo "Description: $DESCRIPTION"

# Create incident via API
curl -X POST http://localhost:3001/incidents \\
  -H "Content-Type: application/json" \\
  -d "{
    \\"title\\": \\"Emergency: $INCIDENT_TYPE\\",
    \\"description\\": \\"$DESCRIPTION\\",
    \\"severity\\": \\"$SEVERITY\\",
    \\"service\\": \\"$INCIDENT_TYPE\\",
    \\"source\\": \\"manual\\"
  }"

echo "\\n✅ Emergency incident created"
`;

    await fs.writeFile(
      path.join(this.scriptsDir, 'emergency-response.sh'),
      emergencyScript
    );
    await fs.chmod(path.join(this.scriptsDir, 'emergency-response.sh'), 0o755);
    console.log('   Created: scripts/incident-response/emergency-response.sh');
  }

  async setupKubernetesResources() {
    console.log('\n☸️  Creating Kubernetes resources...');

    // Incident Response Service Deployment
    const deployment = {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: 'incident-response-service',
        namespace: 'default',
        labels: {
          app: 'incident-response-service'
        }
      },
      spec: {
        replicas: 2,
        selector: {
          matchLabels: {
            app: 'incident-response-service'
          }
        },
        template: {
          metadata: {
            labels: {
              app: 'incident-response-service'
            }
          },
          spec: {
            containers: [{
              name: 'incident-response',
              image: 'incident-response:latest',
              ports: [{
                containerPort: 3001
              }],
              env: [
                {
                  name: 'PAGERDUTY_API_KEY',
                  valueFrom: {
                    secretKeyRef: {
                      name: 'incident-response-secrets',
                      key: 'pagerduty-api-key'
                    }
                  }
                },
                {
                  name: 'SLACK_INCIDENT_WEBHOOK',
                  valueFrom: {
                    secretKeyRef: {
                      name: 'incident-response-secrets',
                      key: 'slack-webhook-url'
                    }
                  }
                }
              ],
              resources: {
                requests: {
                  memory: '256Mi',
                  cpu: '100m'
                },
                limits: {
                  memory: '512Mi',
                  cpu: '500m'
                }
              },
              livenessProbe: {
                httpGet: {
                  path: '/health',
                  port: 3001
                },
                initialDelaySeconds: 30,
                periodSeconds: 10
              },
              readinessProbe: {
                httpGet: {
                  path: '/health',
                  port: 3001
                },
                initialDelaySeconds: 5,
                periodSeconds: 5
              }
            }]
          }
        }
      }
    };

    await fs.writeFile(
      'k8s/incident-response/deployment.yaml',
      `# Incident Response Service Deployment
${this.yamlStringify(deployment)}`
    );
    console.log('   Created: k8s/incident-response/deployment.yaml');

    // Service
    const service = {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: {
        name: 'incident-response-service',
        labels: {
          app: 'incident-response-service'
        }
      },
      spec: {
        selector: {
          app: 'incident-response-service'
        },
        ports: [{
          port: 80,
          targetPort: 3001,
          protocol: 'TCP'
        }],
        type: 'ClusterIP'
      }
    };

    await fs.writeFile(
      'k8s/incident-response/service.yaml',
      `# Incident Response Service
${this.yamlStringify(service)}`
    );
    console.log('   Created: k8s/incident-response/service.yaml');
  }

  async createMonitoringAlerts() {
    console.log('\n📊 Creating monitoring alerts...');

    // Prometheus alert rules
    const prometheusRules = {
      apiVersion: 'monitoring.coreos.com/v1',
      kind: 'PrometheusRule',
      metadata: {
        name: 'incident-response-alerts',
        labels: {
          app: 'incident-response'
        }
      },
      spec: {
        groups: [{
          name: 'incident-response.rules',
          rules: [
            {
              alert: 'IncidentResponseServiceDown',
              expr: 'up{job="incident-response-service"} == 0',
              for: '1m',
              labels: {
                severity: 'critical'
              },
              annotations: {
                summary: 'Incident Response Service is down',
                description: 'The incident response service has been down for more than 1 minute.'
              }
            },
            {
              alert: 'HighIncidentRate',
              expr: 'rate(incidents_created_total[5m]) > 0.1',
              for: '2m',
              labels: {
                severity: 'warning'
              },
              annotations: {
                summary: 'High incident creation rate',
                description: 'Incident creation rate is {{ $value }} incidents per second.'
              }
            }
          ]
        }]
      }
    };

    await fs.writeFile(
      'k8s/incident-response/prometheus-rules.yaml',
      `# Incident Response Prometheus Rules
${this.yamlStringify(prometheusRules)}`
    );
    console.log('   Created: k8s/incident-response/prometheus-rules.yaml');
  }

  async setupPagerDutyIntegration() {
    console.log('\n📟 Setting up PagerDuty integration...');

    const pagerdutyConfig = `# PagerDuty Integration Configuration

## Service Keys
- **Main Service Key**: Used for general incidents
- **Database Escalation Key**: Used for database-related escalations  
- **Platform Escalation Key**: Used for platform-wide issues
- **Security Escalation Key**: Used for security incidents

## Webhook Configuration
Configure PagerDuty webhooks to point to:
\`\`\`
https://your-domain.com/webhooks/alerts/pagerduty
\`\`\`

## Event Rules
The following event rules are automatically configured:
- Database incidents escalate after 10 minutes
- API incidents escalate after 15 minutes
- Security incidents escalate immediately

## Testing
Test the integration with:
\`\`\`bash
curl -X POST https://events.pagerduty.com/v2/enqueue \\
  -H "Content-Type: application/json" \\
  -d '{
    "routing_key": "YOUR_SERVICE_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Test incident from setup",
      "severity": "info",
      "source": "incident-response-setup"
    }
  }'
\`\`\`
`;

    await fs.writeFile(
      'docs/incident-response/pagerduty-setup.md',
      pagerdutyConfig
    );
    console.log('   Created: docs/incident-response/pagerduty-setup.md');
  }

  yamlStringify(obj) {
    // Simple YAML stringifier for basic objects
    return JSON.stringify(obj, null, 2)
      .replace(/"/g, '')
      .replace(/,$/gm, '')
      .replace(/^(\s*)([\w-]+):/gm, '$1$2:');
  }

  printNextSteps() {
    console.log(`
📋 Next Steps:

1. **Configure Environment Variables:**
   export PAGERDUTY_API_KEY="your-api-key"
   export PAGERDUTY_SERVICE_KEY="your-service-key"
   export SLACK_INCIDENT_WEBHOOK="your-slack-webhook"

2. **Create Kubernetes Secrets:**
   kubectl create secret generic incident-response-secrets \\
     --from-literal=pagerduty-api-key="$PAGERDUTY_API_KEY" \\
     --from-literal=slack-webhook-url="$SLACK_INCIDENT_WEBHOOK"

3. **Deploy to Kubernetes:**
   kubectl apply -f k8s/incident-response/

4. **Start the Service Locally (for testing):**
   ./scripts/incident-response/start-incident-response.sh

5. **Test Emergency Response:**
   ./scripts/incident-response/emergency-response.sh database critical "Test incident"

6. **Configure Monitoring:**
   - Set up Prometheus to scrape metrics from the service
   - Configure alert manager to send alerts to the webhook endpoints
   - Test alert routing with sample alerts

7. **Update Runbooks:**
   - Review and customize runbooks in docs/runbooks/
   - Add your specific infrastructure details
   - Update contact information and escalation procedures

📚 Documentation:
   - Runbooks: docs/runbooks/
   - PIR Templates: docs/incident-response/
   - Configuration: config/incident-response/

🔗 Useful URLs (once deployed):
   - Health Check: http://localhost:3001/health
   - Incident API: http://localhost:3001/incidents
   - Alert Stats: http://localhost:3001/stats/alerts
`);
  }
}

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const setup = new IncidentResponseSetup();
  setup.setup().catch(console.error);
}

export default IncidentResponseSetup;