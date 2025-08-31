import mongoose from 'mongoose';
import Outbox from '../models/Outbox.js';

/**
 * Universal Event Sourcing Plugin
 * Automatically creates events for all CRUD operations
 */
const eventSourcingPlugin = function(schema, options = {}) {
    const defaults = {
        aggregateName: null, // Must be provided
        enableCreate: true,
        enableUpdate: true,
        enableDelete: true,
        excludeFields: ['__v', 'updatedAt', 'version']
    };

    const opts = { ...defaults, ...options };

    if (!opts.aggregateName) {
        throw new Error('aggregateName is required for eventSourcing plugin');
    }

    // Helper to create event
    const createEvent = async function(doc, eventType, changes = null) {
        const payload = {
            aggregateId: doc._id,
            data: doc.toObject(),
            changes: changes,
            timestamp: new Date(),
            version: doc.version || 1
        };

        // Remove excluded fields from payload
        opts.excludeFields.forEach(field => {
            delete payload.data[field];
        });

        const idempotencyKey = `${doc._id}-${doc.version || 1}-${eventType}`;

        try {
            await Outbox.create({
                aggregate: opts.aggregateName,
                aggregateId: doc._id,
                eventType: eventType,
                payload: payload,
                version: doc.version || 1,
                idempotencyKey: idempotencyKey
            });
        } catch (error) {
            if (error.code !== 11000) { // Ignore duplicate key errors (idempotency)
                console.error(`Failed to create event for ${opts.aggregateName}:`, error);
            }
        }
    };

    // Post save hook for create and update
    schema.post('save', async function(doc, next) {
        try {
            if (this.isNew && opts.enableCreate) {
                await createEvent(doc, `${opts.aggregateName}Created`);
            } else if (!this.isNew && opts.enableUpdate) {
                const changes = this.getChanges ? this.getChanges() : null;
                await createEvent(doc, `${opts.aggregateName}Updated`, changes);
            }
        } catch (error) {
            console.error(`Event sourcing error in ${opts.aggregateName}:`, error);
        }
        next();
    });

    // Post remove hook for delete
    schema.post('remove', async function(doc, next) {
        try {
            if (opts.enableDelete) {
                await createEvent(doc, `${opts.aggregateName}Deleted`);
            }
        } catch (error) {
            console.error(`Event sourcing error in ${opts.aggregateName}:`, error);
        }
        next();
    });

    // Post findOneAndUpdate hook
    schema.post('findOneAndUpdate', async function(doc, next) {
        try {
            if (doc && opts.enableUpdate) {
                await createEvent(doc, `${opts.aggregateName}Updated`);
            }
        } catch (error) {
            console.error(`Event sourcing error in ${opts.aggregateName}:`, error);
        }
        next();
    });

    // Track changes for better event payload
    schema.pre('save', function(next) {
        if (!this.isNew) {
            this._changes = this.modifiedPaths().reduce((changes, path) => {
                changes[path] = {
                    from: this._original ? this._original[path] : undefined,
                    to: this[path]
                };
                return changes;
            }, {});
        }
        next();
    });

    schema.methods.getChanges = function() {
        return this._changes || {};
    };
};

export default eventSourcingPlugin;