import mongoose from 'mongoose';

const OutboxSchema = new mongoose.Schema({
    aggregate: {
        type: String,
        required: true,
        index: true
    },
    aggregateId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        index: true
    },
    eventType: {
        type: String,
        required: true,
        index: true
    },
    payload: {
        type: mongoose.Schema.Types.Mixed,
        required: true
    },
    version: {
        type: Number,
        default: 1
    },
    processed: {
        type: Boolean,
        default: false,
        index: true
    },
    processedAt: {
        type: Date,
        sparse: true
    },
    retryCount: {
        type: Number,
        default: 0
    },
    lastError: {
        type: String
    },
    idempotencyKey: {
        type: String,
        unique: true,
        sparse: true
    }
}, {
    timestamps: true
});

// Compound index for efficient processing
OutboxSchema.index({ processed: 1, createdAt: 1 });

// TTL index to auto-cleanup processed events after 30 days
OutboxSchema.index({ processedAt: 1 }, { 
    expireAfterSeconds: 30 * 24 * 60 * 60 
});

export default mongoose.models.Outbox || mongoose.model('Outbox', OutboxSchema);