import mongoose from 'mongoose';
import dbConnect from '../../lib/dbConnect.js';
import { requireApprovedUser } from '../../middleware/auth.js';
import Post from '../../models/Post.js';

/**
 * Debug Posts - Check actual database state
 */
export default async function handler(req, res) {
    try {
        await dbConnect();

        const user = await requireApprovedUser(req, res);
        if (!user) return;

        // Get all posts including deleted ones
        const allPosts = await Post.find({}).lean();
        
        // Get posts with isDeleted field
        const deletedPosts = await Post.find({ isDeleted: true }).lean();
        
        // Get posts without isDeleted field or isDeleted: false
        const activePosts = await Post.find({
            $or: [
                { isDeleted: { $exists: false } },
                { isDeleted: false }
            ]
        }).lean();

        res.status(200).json({
            success: true,
            data: {
                allPosts: allPosts.map(p => ({
                    _id: p._id,
                    title: p.title,
                    isDeleted: p.isDeleted,
                    deletedAt: p.deletedAt,
                    deletedBy: p.deletedBy,
                    tombstoneReason: p.tombstoneReason
                })),
                deletedPosts: deletedPosts.length,
                activePosts: activePosts.length,
                totalPosts: allPosts.length
            }
        });
        
    } catch (error) {
        console.error('Debug posts error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: error.message
        });
    }
}