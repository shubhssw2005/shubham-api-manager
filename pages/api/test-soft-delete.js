import mongoose from 'mongoose';
import dbConnect from '../../lib/dbConnect.js';
import { requireApprovedUser } from '../../middleware/auth.js';
import Post from '../../models/Post.js';
import Outbox from '../../models/Outbox.js';

/**
 * Test Soft Delete Functionality
 */
export default async function handler(req, res) {
    try {
        await dbConnect();

        const user = await requireApprovedUser(req, res);
        if (!user) return;

        const { id } = req.query;

        switch (req.method) {
            case 'DELETE':
                return await softDeletePost(req, res, user, id);
            case 'POST':
                return await restorePost(req, res, user, id);
            case 'GET':
                return await getDeletedPosts(req, res);
            default:
                return res.status(405).json({
                    success: false,
                    message: `Method ${req.method} not allowed`
                });
        }
    } catch (error) {
        console.error('Soft delete test error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: error.message
        });
    }
}

async function softDeletePost(req, res, user, id) {
    const session = await mongoose.startSession();
    
    try {
        session.startTransaction();
        
        const { reason = 'user_request' } = req.body;
        
        // Soft delete the post
        const result = await Post.updateOne(
            { _id: id, isDeleted: { $ne: true } },
            {
                $set: {
                    isDeleted: true,
                    deletedAt: new Date(),
                    deletedBy: user._id,
                    tombstoneReason: reason
                },
                $inc: { version: 1 }
            },
            { session }
        );
        
        if (result.matchedCount === 0) {
            await session.abortTransaction();
            return res.status(404).json({
                success: false,
                message: 'Post not found or already deleted'
            });
        }
        
        // Create delete event
        await Outbox.create([{
            aggregate: 'Post',
            aggregateId: id,
            eventType: 'PostSoftDeleted',
            payload: {
                aggregateId: id,
                deletedBy: user._id,
                reason: reason,
                timestamp: new Date()
            },
            version: 1,
            idempotencyKey: `${id}-${Date.now()}-PostSoftDeleted`
        }], { session });
        
        await session.commitTransaction();
        
        res.status(200).json({
            success: true,
            message: 'Post soft deleted successfully'
        });
        
    } catch (error) {
        await session.abortTransaction();
        throw error;
    } finally {
        session.endSession();
    }
}

async function restorePost(req, res, user, id) {
    const session = await mongoose.startSession();
    
    try {
        session.startTransaction();
        
        // Restore the post
        const result = await Post.updateOne(
            { _id: id, isDeleted: true },
            {
                $set: {
                    isDeleted: false,
                    deletedAt: null,
                    deletedBy: null,
                    tombstoneReason: null
                },
                $inc: { version: 1 }
            },
            { session }
        );
        
        if (result.matchedCount === 0) {
            await session.abortTransaction();
            return res.status(404).json({
                success: false,
                message: 'Post not found in deleted items'
            });
        }
        
        // Create restore event
        await Outbox.create([{
            aggregate: 'Post',
            aggregateId: id,
            eventType: 'PostRestored',
            payload: {
                aggregateId: id,
                restoredBy: user._id,
                timestamp: new Date()
            },
            version: 1,
            idempotencyKey: `${id}-${Date.now()}-PostRestored`
        }], { session });
        
        await session.commitTransaction();
        
        res.status(200).json({
            success: true,
            message: 'Post restored successfully'
        });
        
    } catch (error) {
        await session.abortTransaction();
        throw error;
    } finally {
        session.endSession();
    }
}

async function getDeletedPosts(req, res) {
    try {
        const deletedPosts = await Post.find({ isDeleted: true })
            .populate('author', 'name email')
            .populate('deletedBy', 'name email')
            .sort({ deletedAt: -1 });
        
        res.status(200).json({
            success: true,
            data: deletedPosts,
            count: deletedPosts.length
        });
        
    } catch (error) {
        throw error;
    }
}