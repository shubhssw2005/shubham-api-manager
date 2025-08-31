import Outbox from '../models/Outbox.js';
import AWS from 'aws-sdk';

/**
 * Outbox Worker - Processes events and streams to S3/Kafka
 */
class OutboxWorker {
    constructor() {
        this.s3 = new AWS.S3({
            region: process.env.AWS_REGION || 'us-east-1',
            accessKeyId: process.env.AWS_ACCESS_KEY_ID,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
        });
        this.eventBucket = process.env.EVENT_BUCKET_NAME || 'event-stream-bucket';
        this.batchSize = parseInt(process.env.OUTBOX_BATCH_SIZE) || 100;
        this.isRunning = false;
    }

    /**
     * Start processing outbox events
     */
    async start() {
        if (this.isRunning) {
            console.log('Outbox worker already running');
            return;
        }

        this.isRunning = true;
        console.log('Starting outbox worker...');

        while (this.isRunning) {
            try {
                await this.processBatch();
                await this.sleep(5000); // 5 second interval
            } catch (error) {
                console.error('Outbox worker error:', error);
                await this.sleep(10000); // Wait longer on error
            }
        }
    }

    /**
     * Stop the worker
     */
    stop() {
        console.log('Stopping outbox worker...');
        this.isRunning = false;
    }

    /**
     * Process a batch of outbox events
     */
    async processBatch() {
        const events = await Outbox.find({ processed: false })
            .sort({ createdAt: 1 })
            .limit(this.batchSize);

        if (events.length === 0) {
            return;
        }

        console.log(`Processing ${events.length} outbox events`);

        const eventsByAggregate = this.groupEventsByAggregate(events);

        for (const [aggregate, aggregateEvents] of Object.entries(eventsByAggregate)) {
            try {
                await this.streamEventsToS3(aggregate, aggregateEvents);
                await this.markEventsAsProcessed(aggregateEvents);
            } catch (error) {
                console.error(`Failed to process events for ${aggregate}:`, error);
                await this.handleFailedEvents(aggregateEvents, error);
            }
        }
    }

    /**
     * Group events by aggregate type
     */
    groupEventsByAggregate(events) {
        return events.reduce((groups, event) => {
            const aggregate = event.aggregate;
            if (!groups[aggregate]) {
                groups[aggregate] = [];
            }
            groups[aggregate].push(event);
            return groups;
        }, {});
    }

    /**
     * Stream events to S3 in Parquet-like format
     */
    async streamEventsToS3(aggregate, events) {
        const date = new Date().toISOString().split('T')[0];
        const hour = new Date().getHours().toString().padStart(2, '0');
        const timestamp = Date.now();
        
        const key = `events/aggregate=${aggregate}/dt=${date}/hour=${hour}/${timestamp}.json.gz`;

        const eventData = events.map(event => ({
            eventId: event._id,
            aggregate: event.aggregate,
            aggregateId: event.aggregateId,
            eventType: event.eventType,
            payload: event.payload,
            version: event.version,
            timestamp: event.createdAt,
            idempotencyKey: event.idempotencyKey
        }));

        const compressed = await this.compressData(JSON.stringify(eventData));

        const params = {
            Bucket: this.eventBucket,
            Key: key,
            Body: compressed,
            ContentType: 'application/gzip',
            Metadata: {
                'aggregate': aggregate,
                'event-count': events.length.toString(),
                'date': date,
                'hour': hour
            }
        };

        await this.s3.upload(params).promise();
        console.log(`Streamed ${events.length} ${aggregate} events to S3: ${key}`);
    }

    /**
     * Mark events as processed
     */
    async markEventsAsProcessed(events) {
        const eventIds = events.map(event => event._id);
        
        await Outbox.updateMany(
            { _id: { $in: eventIds } },
            {
                $set: {
                    processed: true,
                    processedAt: new Date()
                }
            }
        );
    }

    /**
     * Handle failed events with retry logic
     */
    async handleFailedEvents(events, error) {
        const eventIds = events.map(event => event._id);
        
        await Outbox.updateMany(
            { _id: { $in: eventIds } },
            {
                $inc: { retryCount: 1 },
                $set: { lastError: error.message }
            }
        );

        // Mark as failed if too many retries
        await Outbox.updateMany(
            { 
                _id: { $in: eventIds },
                retryCount: { $gte: 5 }
            },
            {
                $set: {
                    processed: true,
                    processedAt: new Date(),
                    lastError: `Max retries exceeded: ${error.message}`
                }
            }
        );
    }

    /**
     * Compress data for S3 storage
     */
    async compressData(data) {
        const zlib = await import('zlib');
        return new Promise((resolve, reject) => {
            zlib.gzip(data, (err, compressed) => {
                if (err) reject(err);
                else resolve(compressed);
            });
        });
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get processing stats
     */
    async getStats() {
        const [pending, processed, failed] = await Promise.all([
            Outbox.countDocuments({ processed: false, retryCount: { $lt: 5 } }),
            Outbox.countDocuments({ processed: true }),
            Outbox.countDocuments({ retryCount: { $gte: 5 } })
        ]);

        return { pending, processed, failed };
    }

    /**
     * Replay events for a specific aggregate
     */
    async replayEvents(aggregate, fromDate, toDate) {
        const events = await Outbox.find({
            aggregate: aggregate,
            createdAt: {
                $gte: new Date(fromDate),
                $lte: new Date(toDate)
            }
        }).sort({ createdAt: 1 });

        console.log(`Replaying ${events.length} events for ${aggregate}`);
        
        // Process events in order
        for (const event of events) {
            try {
                await this.streamEventsToS3(aggregate, [event]);
                console.log(`Replayed event ${event._id}`);
            } catch (error) {
                console.error(`Failed to replay event ${event._id}:`, error);
            }
        }
    }
}

export default OutboxWorker;