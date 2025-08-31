import mongoose from 'mongoose';

const DataExportJobSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
        index: true
    },
    requestedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    status: {
        type: String,
        enum: ['pending', 'in_progress', 'ready', 'failed', 'expired'],
        default: 'pending',
        index: true
    },
    exportType: {
        type: String,
        enum: ['full', 'posts_only', 'media_only', 'profile_only'],
        default: 'full'
    },
    includeMedia: {
        type: Boolean,
        default: true
    },
    includeDeleted: {
        type: Boolean,
        default: false
    },
    progress: {
        type: Number,
        default: 0,
        min: 0,
        max: 100
    },
    totalItems: {
        type: Number,
        default: 0
    },
    processedItems: {
        type: Number,
        default: 0
    },
    downloadUrl: {
        type: String
    },
    downloadExpiry: {
        type: Date
    },
    fileSize: {
        type: Number // in bytes
    },
    errorMessage: {
        type: String
    },
    metadata: {
        collections: [{
            name: String,
            count: Number,
            size: Number
        }],
        mediaFiles: {
            count: Number,
            totalSize: Number
        },
        exportedAt: Date,
        zipPath: String
    },
    // Auto-expire completed jobs after 48 hours
    expiresAt: {
        type: Date,
        default: () => new Date(Date.now() + 48 * 60 * 60 * 1000), // 48 hours
        index: { expireAfterSeconds: 0 }
    }
}, {
    timestamps: true
});

// Compound indexes for efficient queries
DataExportJobSchema.index({ userId: 1, status: 1 });
DataExportJobSchema.index({ status: 1, createdAt: 1 });

export default mongoose.models.DataExportJob || mongoose.model('DataExportJob', DataExportJobSchema);