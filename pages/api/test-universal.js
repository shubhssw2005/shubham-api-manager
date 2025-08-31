import mongoose from 'mongoose';
import dbConnect from '../../lib/dbConnect.js';
import { requireApprovedUser } from '../../middleware/auth.js';
import Post from '../../models/Post.js';
import Outbox from '../../models/Outbox.js';

/**
 * Test Universal API for Posts
 */
export default async function handler(req, res) {
    try {
        await dbConnect();

        // Authentication check
        const user = await requireApprovedUser(req, res);
        if (!user) return;

        switch (req.method) {
            case 'POST':
                return await createPost(req, res, user);
            case 'GET':
                return await getPosts(req, res);
            default:
                return res.status(405).json({
                    success: false,
                    message: `Method ${req.method} not allowed`
                });
        }
    } catch (error) {
        console.error('Test API error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: error.message
        });
    }
}

async function createPost(req, res, user) {
    const session = await mongoose.startSession();
    
    try {
        session.startTransaction();
        
        // Create post with audit fields
        const postData = {
            ...req.body,
            author: user._id,
            createdBy: user._id,
            updatedBy: user._id,
            publishedAt: req.body.status === 'published' ? new Date() : null
        };
        
        const post = new Post(postData);
        await post.save({ session });
        
        // Create event for audit trail
        await Outbox.create([{
            aggregate: 'Post',
            aggregateId: post._id,
            eventType: 'PostCreated',
            payload: {
                aggregateId: post._id,
                data: post.toObject(),
                timestamp: new Date(),
                version: 1
            },
            version: 1,
            idempotencyKey: `${post._id}-1-PostCreated`
        }], { session });
        
        await session.commitTransaction();
        
        res.status(201).json({
            success: true,
            data: post,
            message: 'Post created successfully'
        });
        
    } catch (error) {
        await session.abortTransaction();
        
        if (error.name === 'ValidationError') {
            return res.status(400).json({
                success: false,
                message: 'Validation error',
                errors: Object.values(error.errors).map(e => e.message)
            });
        }
        
        throw error;
    } finally {
        session.endSession();
    }
}

async function getPosts(req, res) {
    try {
        const {
            page = 1,
            limit = 10,
            status,
            search,
            featured
        } = req.query;
        
        // Build filter (soft delete filtering)
        const filter = { $or: [{ isDeleted: { $exists: false } }, { isDeleted: false }] };
        if (status) filter.status = status;
        if (featured !== undefined) filter.featured = featured === 'true';
        if (search) {
            filter.$or = [
                { title: { $regex: search, $options: 'i' } },
                { content: { $regex: search, $options: 'i' } },
                { tags: { $in: [new RegExp(search, 'i')] } }
            ];
        }
        
        const skip = (page - 1) * limit;
        
        const [posts, total] = await Promise.all([
            Post.find(filter)
                .populate('author', 'name email')
                .populate('mediaIds')
                .sort({ publishedAt: -1, createdAt: -1 })
                .skip(skip)
                .limit(parseInt(limit)),
            Post.countDocuments(filter)
        ]);
        
        res.status(200).json({
            success: true,
            data: posts,
            pagination: {
                page: parseInt(page),
                pages: Math.ceil(total / limit),
                total,
                hasNext: page < Math.ceil(total / limit),
                hasPrev: page > 1
            }
        });
        
    } catch (error) {
        throw error;
    }
}