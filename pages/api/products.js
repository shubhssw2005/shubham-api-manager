import { verifyToken } from '../../lib/jwt';
import mongoose from 'mongoose';
import connectDb from '../../lib/dbConnect';

// Define Product schema inline to avoid webpack issues
const productSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Please provide a product name'],
        trim: true,
        maxlength: [100, 'Product name cannot be more than 100 characters']
    },
    description: {
        type: String,
        required: [true, 'Please provide a description']
    },
    price: {
        type: Number,
        required: [true, 'Please provide a price'],
        min: [0, 'Price cannot be negative']
    },
    category: {
        type: String,
        required: [true, 'Please provide a category']
    },
    tags: [{
        type: String,
        trim: true
    }],
    mediaIds: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Media'
    }],
    status: {
        type: String,
        enum: ['draft', 'active', 'inactive'],
        default: 'draft'
    },
    createdBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

export default async function handler(req, res) {
    try {
        // Verify authentication
        const auth = req.headers.authorization;
        if (!auth || !auth.startsWith('Bearer ')) {
            res.status(401).json({ message: 'Unauthorized' });
            return;
        }

        const token = auth.split(' ')[1];
        const decoded = await verifyToken(token);
        if (!decoded) {
            res.status(401).json({ message: 'Invalid token' });
            return;
        }

        // Connect to database and get Product model
        await connectDb();
        const Product = mongoose.models.Product || mongoose.model('Product', productSchema);

        // Handle routes with ID parameter
        if (req.query.id) {
            const { id } = req.query;

            switch (req.method) {
                case 'GET':
                    try {
                        const product = await Product.findById(id)
                            .populate('mediaIds')
                            .populate('createdBy', 'name email');
                        
                        if (!product) {
                            res.status(404).json({ message: 'Product not found' });
                            return;
                        }
                        
                        res.status(200).json(product);
                    } catch (error) {
                        res.status(400).json({ message: 'Error retrieving product', error: error.message });
                    }
                    break;

                case 'PUT':
                    try {
                        const product = await Product.findById(id);
                        
                        if (!product) {
                            res.status(404).json({ message: 'Product not found' });
                            return;
                        }

                        // Check if user is the creator
                        if (product.createdBy.toString() !== decoded.userId) {
                            res.status(403).json({ message: 'Not authorized to update this product' });
                            return;
                        }

                        const updatedProduct = await Product.findByIdAndUpdate(
                            id,
                            { ...req.body, updatedAt: Date.now() },
                            { new: true, runValidators: true }
                        ).populate('mediaIds');

                        res.status(200).json(updatedProduct);
                    } catch (error) {
                        res.status(400).json({ message: 'Error updating product', error: error.message });
                    }
                    break;

                case 'DELETE':
                    try {
                        const product = await Product.findById(id);
                        
                        if (!product) {
                            res.status(404).json({ message: 'Product not found' });
                            return;
                        }

                        // Check if user is the creator
                        if (product.createdBy.toString() !== decoded.userId) {
                            res.status(403).json({ message: 'Not authorized to delete this product' });
                            return;
                        }

                        await Product.deleteOne({ _id: id });
                        res.status(200).json({ message: 'Product deleted successfully' });
                    } catch (error) {
                        res.status(400).json({ message: 'Error deleting product', error: error.message });
                    }
                    break;

                default:
                    res.setHeader('Allow', ['GET', 'PUT', 'DELETE']);
                    res.status(405).json({ message: `Method ${req.method} Not Allowed` });
            }
        } else {
            // Handle root routes
            switch (req.method) {
                case 'POST':
                    try {
                        const product = await Product.create({
                            ...req.body,
                            createdBy: decoded.userId
                        });
                        res.status(201).json(product);
                    } catch (error) {
                        console.error('Error creating product:', error);
                        res.status(400).json({ message: 'Error creating product', error: error.message });
                    }
                    break;

                case 'GET':
                    try {
                        const products = await Product.find({})
                            .populate('mediaIds')
                            .populate('createdBy', 'name email');
                        
                        res.status(200).json(products);
                    } catch (error) {
                        res.status(400).json({ message: 'Error retrieving products', error: error.message });
                    }
                    break;

                default:
                    res.setHeader('Allow', ['GET', 'POST']);
                    res.status(405).json({ message: `Method ${req.method} Not Allowed` });
            }
        }
    } catch (error) {
        console.error('Error in product handler:', error);
        res.status(500).json({ message: 'Internal server error', error: error.message });
    }
}
