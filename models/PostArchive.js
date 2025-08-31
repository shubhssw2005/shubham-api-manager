import mongoose from 'mongoose';

const PostArchiveSchema = new mongoose.Schema({
    originalId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        index: true
    },
    title: String,
    content: String,
    mediaIds: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Media'
    }],
    status: String,
    tags: [String],
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    },
    isDeleted: Boolean,
    deletedAt: Date,
    deletedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    },
    tombstoneReason: String,
    originalCreatedAt: Date,
    originalUpdatedAt: Date,
    archivedAt: {
        type: Date,
        default: Date.now
    },
    archiveReason: {
        type: String,
        default: 'retention_policy'
    }
}, {
    timestamps: true
});

// Index for efficient archival queries
PostArchiveSchema.index({ archivedAt: 1 });
PostArchiveSchema.index({ originalId: 1, archivedAt: 1 });

export default mongoose.models.PostArchive || mongoose.model('PostArchive', PostArchiveSchema);