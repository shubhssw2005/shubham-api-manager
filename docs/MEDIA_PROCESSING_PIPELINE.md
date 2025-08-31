# Media Processing Pipeline

This document describes the Event-Driven Media Processing Pipeline implementation for handling S3 uploads with automatic processing, thumbnail generation, and transcoding.

## Overview

The media processing pipeline consists of several components working together to provide scalable, reliable media processing:

1. **S3 Event Processor** - Handles S3 event notifications from SNS
2. **Media Processor Workers** - Process media files from SQS queues
3. **Job Status Tracker** - Tracks processing status using Redis
4. **Specialized Processors** - Handle different media types (image, video, audio, document)

## Architecture

```
S3 Upload → SNS Topic → SQS Queue → Media Processor → Processed Files
                                         ↓
                                   Job Status Tracker (Redis)
```

## Components

### S3 Event Processor (`lib/events/S3EventProcessor.js`)

Handles S3 event notifications and routes them to appropriate processing queues.

**Features:**
- Processes S3 ObjectCreated/ObjectRemoved events
- Tenant-based file organization validation
- Automatic file type detection
- SQS message queuing with retry logic
- Dead letter queue support
- SNS notifications for processing events

### Media Processor Worker (`workers/MediaProcessor.js`)

Main worker that processes media files from SQS queues.

**Features:**
- Concurrent processing with configurable workers
- Automatic retry with exponential backoff
- Processing timeout handling
- Comprehensive metrics and monitoring
- Graceful shutdown support

### Job Status Tracker (`lib/jobs/JobStatusTracker.js`)

Tracks job status and provides persistence using Redis.

**Features:**
- Real-time job status updates
- Job history and timeline tracking
- Status-based job querying
- Automatic cleanup of old jobs
- Comprehensive statistics and metrics

### Specialized Processors

#### Image Processor (`workers/processors/ImageProcessor.js`)
- Thumbnail generation (small, medium, large)
- Format optimization (JPEG, PNG, WebP)
- Size optimization and compression
- EXIF data extraction
- Progressive JPEG support

#### Video Processor (`workers/processors/VideoProcessor.js`)
- Multi-resolution transcoding (480p, 720p, 1080p)
- HLS streaming playlist generation
- Video thumbnail extraction
- Metadata extraction (duration, bitrate, codec)
- Progress tracking

#### Audio Processor (`workers/processors/AudioProcessor.js`)
- Multi-format transcoding (MP3, AAC, OGG)
- Waveform visualization generation
- Bitrate optimization
- Metadata extraction

#### Document Processor (`workers/processors/DocumentProcessor.js`)
- PDF thumbnail generation
- Text extraction from documents
- Multi-page processing
- Format conversion support
- Word count and metadata extraction

## Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# S3 Configuration
MEDIA_BUCKET=your-media-bucket

# SQS Configuration
MEDIA_PROCESSING_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/account/queue-name
MEDIA_DLQ_URL=https://sqs.us-east-1.amazonaws.com/account/dlq-name

# SNS Configuration (optional)
MEDIA_PROCESSING_TOPIC_ARN=arn:aws:sns:us-east-1:account:topic-name

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password
REDIS_DB=0

# Processing Configuration
MEDIA_PROCESSOR_CONCURRENCY=5
MAX_PROCESSING_RETRIES=3
PROCESSING_TIMEOUT=300000
POLL_INTERVAL=1000
SQS_VISIBILITY_TIMEOUT=300
SQS_BATCH_SIZE=10

# Job Configuration
JOB_STATUS_TTL=604800
METRICS_REPORT_INTERVAL=60000

# Feature Flags
ENABLE_HLS_TRANSCODING=false
```

### AWS Infrastructure Setup

#### S3 Bucket Configuration

```json
{
  "Rules": [
    {
      "Id": "MediaProcessingNotification",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "tenants/"
      },
      "CloudWatchConfiguration": {
        "TopicArn": "arn:aws:sns:us-east-1:account:media-processing"
      }
    }
  ]
}
```

#### SQS Queue Configuration

```json
{
  "QueueName": "media-processing-queue",
  "Attributes": {
    "VisibilityTimeoutSeconds": "300",
    "MessageRetentionPeriod": "1209600",
    "RedrivePolicy": {
      "deadLetterTargetArn": "arn:aws:sqs:us-east-1:account:media-dlq",
      "maxReceiveCount": 3
    }
  }
}
```

#### SNS Topic Configuration

```json
{
  "TopicName": "media-processing",
  "Subscriptions": [
    {
      "Protocol": "sqs",
      "Endpoint": "arn:aws:sqs:us-east-1:account:media-processing-queue"
    }
  ]
}
```

## Usage

### Starting the Media Processor

```bash
# Using npm script
npm run media:processor

# Using Docker
docker-compose -f docker-compose.media-processor.yml up

# Direct execution
node scripts/start-media-processor.js
```

### API Endpoints

#### Process S3 Events
```
POST /api/media/process-s3-event
```

#### Job Management
```
GET /api/media/jobs - List jobs
GET /api/media/jobs/[jobId] - Get job status
POST /api/media/jobs/[jobId] - Retry job
DELETE /api/media/jobs/[jobId] - Cancel job
```

#### Job Statistics
```
GET /api/media/jobs/stats - Get processing statistics
```

### Example Usage

#### Processing an Image Upload

1. File uploaded to S3: `tenants/tenant-123/users/user-456/media/1234567890-abc123-image.jpg`
2. S3 triggers SNS notification
3. SNS sends message to SQS queue
4. Media processor picks up job from queue
5. Image processor generates thumbnails and optimized versions
6. Processed files uploaded to S3 with keys like:
   - `tenants/tenant-123/users/user-456/media/processed/1234567890-abc123-image_thumb_small.jpg`
   - `tenants/tenant-123/users/user-456/media/processed/1234567890-abc123-image_optimized.jpg`
   - `tenants/tenant-123/users/user-456/media/processed/1234567890-abc123-image_webp.webp`

#### Checking Job Status

```javascript
// Get job status
const response = await fetch('/api/media/jobs/job_1234567890_abc123');
const jobStatus = await response.json();

console.log(jobStatus.job.status); // 'completed'
console.log(jobStatus.job.result.thumbnails); // Array of thumbnail info
```

## Monitoring and Metrics

### Job Statistics

The system provides comprehensive statistics:

- **Processing rates** - Jobs per minute
- **Success rates** - Percentage of successful jobs
- **Error rates** - By file type and error category
- **Processing times** - Average, min, max by file type
- **Queue depths** - Current queue sizes
- **Worker utilization** - Active workers and capacity

### Health Checks

```javascript
// Check system health
const health = await fetch('/api/media/jobs/stats');
const stats = await health.json();

console.log(stats.healthCheck.status); // 'healthy' or 'unhealthy'
console.log(stats.statistics.successRate); // Overall success rate
```

### Logging

The system provides structured logging with:
- Job processing events
- Error details with context
- Performance metrics
- Queue statistics
- Worker lifecycle events

## Error Handling

### Retry Logic

- **Automatic retries** with exponential backoff
- **Maximum retry attempts** configurable per job type
- **Dead letter queue** for failed jobs
- **Manual retry** capability through API

### Error Categories

1. **Transient errors** - Network issues, temporary AWS service issues
2. **Processing errors** - Corrupt files, unsupported formats
3. **Configuration errors** - Missing credentials, invalid settings
4. **Resource errors** - Insufficient memory, disk space

### Recovery Procedures

1. **Failed jobs** can be retried manually or automatically
2. **Dead letter queue** jobs can be reprocessed after fixing issues
3. **System recovery** includes graceful shutdown and restart
4. **Data consistency** maintained through job status tracking

## Testing

### Running Tests

```bash
# Run all media processing tests
npm run media:test

# Run specific test suites
npm test -- tests/integration/media-processing-pipeline.test.js
```

### Test Coverage

- S3 event processing
- Job status tracking
- Media processor workers
- Individual processor components
- Error handling scenarios
- Queue management
- Health checks

## Deployment

### Docker Deployment

```bash
# Build and start services
docker-compose -f docker-compose.media-processor.yml up -d

# Scale workers
docker-compose -f docker-compose.media-processor.yml up -d --scale media-processor=3

# View logs
docker-compose -f docker-compose.media-processor.yml logs -f media-processor
```

### Kubernetes Deployment

The system includes Helm charts for Kubernetes deployment with:
- Horizontal Pod Autoscaling based on SQS queue depth
- Resource limits and requests
- Health checks and readiness probes
- ConfigMaps for configuration
- Secrets for sensitive data

### Production Considerations

1. **Scaling** - Configure worker concurrency based on available resources
2. **Monitoring** - Set up CloudWatch alarms for queue depths and error rates
3. **Security** - Use IAM roles instead of access keys in production
4. **Backup** - Regular backups of Redis job status data
5. **Disaster Recovery** - Multi-region deployment for high availability

## Troubleshooting

### Common Issues

1. **Jobs stuck in processing** - Check worker health and restart if needed
2. **High error rates** - Review error logs and check AWS service status
3. **Queue buildup** - Scale up workers or check for processing bottlenecks
4. **Memory issues** - Adjust worker concurrency or increase memory limits

### Debug Tools

- Redis Commander for job status inspection
- CloudWatch logs for AWS service issues
- Application metrics for performance monitoring
- Health check endpoints for system status

## Performance Optimization

### Recommendations

1. **Worker Scaling** - Monitor queue depth and scale workers accordingly
2. **Resource Allocation** - Allocate sufficient CPU and memory for media processing
3. **Caching** - Use Redis for job status and metadata caching
4. **Batch Processing** - Process multiple files in batches when possible
5. **Format Optimization** - Choose optimal output formats for file size vs quality

### Benchmarks

Typical processing times (varies by file size and complexity):
- **Images** - 1-5 seconds for thumbnail generation
- **Videos** - 1-10 minutes for transcoding depending on length and resolution
- **Audio** - 30 seconds to 2 minutes for format conversion
- **Documents** - 10-30 seconds for PDF processing

## Security

### Best Practices

1. **IAM Roles** - Use least-privilege IAM roles for AWS access
2. **Encryption** - Enable S3 server-side encryption
3. **Network Security** - Use VPC and security groups to restrict access
4. **Input Validation** - Validate file types and sizes before processing
5. **Audit Logging** - Log all processing activities for compliance

### Compliance

The system supports compliance requirements through:
- Audit trails of all processing activities
- Data retention policies
- Secure file handling and processing
- Access controls and authentication