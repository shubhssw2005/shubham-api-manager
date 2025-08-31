import mongoose from 'mongoose';

/**
 * Universal Soft Delete Plugin
 * Adds soft delete functionality to any Mongoose schema
 */
const softDeletePlugin = function(schema, options = {}) {
    const defaults = {
        indexFields: true,
        deletedField: 'isDeleted',
        deletedAtField: 'deletedAt',
        deletedByField: 'deletedBy',
        tombstoneReasonField: 'tombstoneReason',
        useTimestamps: true
    };

    const opts = { ...defaults, ...options };

    // Add soft delete fields to schema
    const softDeleteFields = {
        [opts.deletedField]: {
            type: Boolean,
            default: false,
            index: opts.indexFields
        },
        [opts.deletedAtField]: {
            type: Date,
            index: opts.indexFields,
            sparse: true
        },
        [opts.deletedByField]: {
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User',
            index: opts.indexFields,
            sparse: true
        },
        [opts.tombstoneReasonField]: {
            type: String,
            enum: ['user_request', 'admin_action', 'policy_violation', 'retention_policy', 'gdpr_erasure', 'system_cleanup']
        }
    };

    schema.add(softDeleteFields);

    // Add version field for event sourcing
    schema.add({
        version: {
            type: Number,
            default: 1
        }
    });

    // Override default find methods to exclude deleted documents
    const excludeDeleted = function() {
        if (!this.getQuery()[opts.deletedField]) {
            this.where({ [opts.deletedField]: { $ne: true } });
        }
        return this;
    };

    // Apply to all find methods
    schema.pre(['find', 'findOne', 'findOneAndUpdate', 'count', 'countDocuments'], excludeDeleted);

    // Instance methods
    schema.methods.softDelete = function(deletedBy, reason = 'user_request') {
        this[opts.deletedField] = true;
        this[opts.deletedAtField] = new Date();
        this[opts.deletedByField] = deletedBy;
        this[opts.tombstoneReasonField] = reason;
        return this.save();
    };

    schema.methods.restore = function() {
        this[opts.deletedField] = false;
        this[opts.deletedAtField] = undefined;
        this[opts.deletedByField] = undefined;
        this[opts.tombstoneReasonField] = undefined;
        return this.save();
    };

    // Static methods
    schema.statics.findDeleted = function(filter = {}) {
        return this.find({ ...filter, [opts.deletedField]: true });
    };

    schema.statics.findWithDeleted = function(filter = {}) {
        return this.find(filter);
    };

    schema.statics.softDeleteById = function(id, deletedBy, reason = 'user_request') {
        return this.updateOne(
            { _id: id, [opts.deletedField]: { $ne: true } },
            {
                $set: {
                    [opts.deletedField]: true,
                    [opts.deletedAtField]: new Date(),
                    [opts.deletedByField]: deletedBy,
                    [opts.tombstoneReasonField]: reason
                },
                $inc: { version: 1 }
            }
        );
    };

    schema.statics.restoreById = function(id) {
        return this.updateOne(
            { _id: id },
            {
                $set: {
                    [opts.deletedField]: false,
                    [opts.deletedAtField]: null,
                    [opts.deletedByField]: null,
                    [opts.tombstoneReasonField]: null
                },
                $inc: { version: 1 }
            }
        );
    };

    // Hard delete (use with caution)
    schema.statics.hardDelete = function(filter = {}) {
        return this.deleteMany({ ...filter, [opts.deletedField]: true });
    };
};

export default softDeletePlugin;