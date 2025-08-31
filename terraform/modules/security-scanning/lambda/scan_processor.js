const AWS = require('aws-sdk');
const https = require('https');
const url = require('url');

const ecr = new AWS.ECR();
const sns = new AWS.SNS();
const cloudwatch = new AWS.CloudWatch();

exports.handler = async (event) => {
    console.log('Processing ECR scan event:', JSON.stringify(event, null, 2));
    
    try {
        const detail = event.detail;
        const repositoryName = detail['repository-name'];
        const imageTag = detail['image-tags'][0] || 'latest';
        const scanStatus = detail['scan-status'];
        
        if (scanStatus !== 'COMPLETE') {
            console.log(`Scan not complete for ${repositoryName}:${imageTag}, status: ${scanStatus}`);
            return;
        }
        
        // Get scan findings
        const findings = await getScanFindings(repositoryName, imageTag);
        
        // Process and categorize findings
        const summary = processScanFindings(findings);
        
        // Send metrics to CloudWatch
        await sendMetrics(repositoryName, summary);
        
        // Send alerts if critical or high severity vulnerabilities found
        if (summary.critical > 0 || summary.high > 0) {
            await sendAlert(repositoryName, imageTag, summary);
        }
        
        // Send to Slack if webhook configured
        if (process.env.SLACK_WEBHOOK) {
            await sendSlackNotification(repositoryName, imageTag, summary);
        }
        
        console.log(`Processed scan results for ${repositoryName}:${imageTag}`, summary);
        
    } catch (error) {
        console.error('Error processing scan results:', error);
        throw error;
    }
};

async function getScanFindings(repositoryName, imageTag) {
    const params = {
        repositoryName: repositoryName,
        imageId: {
            imageTag: imageTag
        },
        maxResults: 1000
    };
    
    try {
        const response = await ecr.describeImageScanFindings(params).promise();
        return response.imageScanFindings.findings || [];
    } catch (error) {
        if (error.code === 'ScanNotFoundException') {
            console.log(`No scan findings found for ${repositoryName}:${imageTag}`);
            return [];
        }
        throw error;
    }
}

function processScanFindings(findings) {
    const summary = {
        total: findings.length,
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        informational: 0,
        undefined: 0
    };
    
    const vulnerabilities = [];
    
    findings.forEach(finding => {
        const severity = finding.severity || 'UNDEFINED';
        summary[severity.toLowerCase()]++;
        
        if (severity === 'CRITICAL' || severity === 'HIGH') {
            vulnerabilities.push({
                name: finding.name,
                severity: severity,
                description: finding.description,
                uri: finding.uri,
                attributes: finding.attributes
            });
        }
    });
    
    summary.vulnerabilities = vulnerabilities;
    return summary;
}

async function sendMetrics(repositoryName, summary) {
    const metricData = [
        {
            MetricName: 'TotalVulnerabilities',
            Value: summary.total,
            Unit: 'Count',
            Dimensions: [
                {
                    Name: 'Repository',
                    Value: repositoryName
                }
            ]
        },
        {
            MetricName: 'CriticalVulnerabilities',
            Value: summary.critical,
            Unit: 'Count',
            Dimensions: [
                {
                    Name: 'Repository',
                    Value: repositoryName
                }
            ]
        },
        {
            MetricName: 'HighSeverityVulnerabilities',
            Value: summary.high,
            Unit: 'Count',
            Dimensions: [
                {
                    Name: 'Repository',
                    Value: repositoryName
                }
            ]
        }
    ];
    
    await cloudwatch.putMetricData({
        Namespace: 'Custom/Security',
        MetricData: metricData
    }).promise();
}

async function sendAlert(repositoryName, imageTag, summary) {
    const message = {
        alert: 'Security Vulnerability Detected',
        repository: repositoryName,
        imageTag: imageTag,
        summary: {
            critical: summary.critical,
            high: summary.high,
            total: summary.total
        },
        vulnerabilities: summary.vulnerabilities.slice(0, 10), // Limit to first 10
        timestamp: new Date().toISOString()
    };
    
    await sns.publish({
        TopicArn: process.env.SNS_TOPIC_ARN,
        Subject: `Security Alert: Vulnerabilities in ${repositoryName}:${imageTag}`,
        Message: JSON.stringify(message, null, 2)
    }).promise();
}

async function sendSlackNotification(repositoryName, imageTag, summary) {
    const color = summary.critical > 0 ? 'danger' : summary.high > 0 ? 'warning' : 'good';
    
    const payload = {
        username: 'Security Scanner',
        icon_emoji: ':shield:',
        attachments: [
            {
                color: color,
                title: `Security Scan Results: ${repositoryName}:${imageTag}`,
                fields: [
                    {
                        title: 'Critical',
                        value: summary.critical.toString(),
                        short: true
                    },
                    {
                        title: 'High',
                        value: summary.high.toString(),
                        short: true
                    },
                    {
                        title: 'Medium',
                        value: summary.medium.toString(),
                        short: true
                    },
                    {
                        title: 'Total',
                        value: summary.total.toString(),
                        short: true
                    }
                ],
                footer: 'AWS ECR Security Scanner',
                ts: Math.floor(Date.now() / 1000)
            }
        ]
    };
    
    if (summary.vulnerabilities.length > 0) {
        const vulnText = summary.vulnerabilities
            .slice(0, 5)
            .map(v => `• ${v.name} (${v.severity})`)
            .join('\n');
        
        payload.attachments[0].text = `Top vulnerabilities:\n${vulnText}`;
    }
    
    return new Promise((resolve, reject) => {
        const webhookUrl = url.parse(process.env.SLACK_WEBHOOK);
        const postData = JSON.stringify(payload);
        
        const options = {
            hostname: webhookUrl.hostname,
            port: 443,
            path: webhookUrl.path,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        };
        
        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                if (res.statusCode === 200) {
                    resolve(data);
                } else {
                    reject(new Error(`Slack webhook failed: ${res.statusCode} ${data}`));
                }
            });
        });
        
        req.on('error', reject);
        req.write(postData);
        req.end();
    });
}