import dbConnect from '../../../lib/dbConnect';
import { verifyToken } from '../../../lib/jwt';
import Post from '../../../models/Post';
import Media from '../../../models/Media';

export default async function handler(req, res) {
    const {
        query: { id },
        method,
    } = req;

    try {
        // Verify authentication
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

        switch (method) {
            case 'GET':
                try {
                    const post = await Post.findById(id)
                        .populate('mediaIds')
                        .populate('author', 'name email');
                    
                    if (!post) {
                        return res.status(404).json({ message: 'Post not found' });
                    }
                    
                    res.status(200).json(post);
                } catch (error) {
                    res.status(400).json({ message: 'Error retrieving post', error: error.message });
                }
                break;

            case 'PUT':
                try {
                    const post = await Post.findById(id);
                    
                    if (!post) {
                        return res.status(404).json({ message: 'Post not found' });
                    }

                    // Check if user is the author
                    if (post.author.toString() !== decoded.userId) {
                        return res.status(403).json({ message: 'Not authorized to update this post' });
                    }

                    const updatedPost = await Post.findByIdAndUpdate(
                        id,
                        { ...req.body, updatedAt: Date.now() },
                        { new: true, runValidators: true }
                    );

                    res.status(200).json(updatedPost);
                } catch (error) {
                    res.status(400).json({ message: 'Error updating post', error: error.message });
                }
                break;

            case 'DELETE':
                try {
                    const post = await Post.findById(id);
                    
                    if (!post) {
                        return res.status(404).json({ message: 'Post not found' });
                    }

                    // Check if user is the author
                    if (post.author.toString() !== decoded.userId) {
                        return res.status(403).json({ message: 'Not authorized to delete this post' });
                    }

                    await Post.deleteOne({ _id: id });
                    res.status(200).json({ message: 'Post deleted successfully' });
                } catch (error) {
                    res.status(400).json({ message: 'Error deleting post', error: error.message });
                }
                break;

            default:
                res.setHeader('Allow', ['GET', 'PUT', 'DELETE']);
                res.status(405).json({ message: `Method ${method} Not Allowed` });
        }
    } catch (error) {
        console.error('Error in post handler:', error);
        res.status(500).json({ message: 'Internal server error', error: error.message });
    }
}
