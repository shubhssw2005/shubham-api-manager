import dbConnect from '../../../lib/dbConnect';
import { verifyToken } from '../../../lib/jwt';
import Post from '../../../models/Post';

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ message: 'Method not allowed' });
    }

    try {
        const auth = req.headers.authorization;
        if (!auth || !auth.startsWith('Bearer ')) {
            return res.status(401).json({ message: 'Unauthorized' });
        }

        const token = auth.split(' ')[1];
        const decoded = await verifyToken(token);
        if (!decoded) {
            return res.status(401).json({ message: 'Invalid token' });
        }

        await dbConnect();

        const { title, content, mediaIds, status, tags } = req.body;
        
        // Create blog post
        const post = await Post.create({
            title,
            content,
            mediaIds,
            status,
            tags,
            author: decoded.userId
        });

        res.status(201).json(post);
    } catch (error) {
        console.error('Error creating blog post:', error);
        res.status(500).json({ message: 'Internal server error', error: error.message });
    }
}
