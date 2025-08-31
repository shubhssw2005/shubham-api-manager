import Product from '../../../models/SimpleProduct.js';
import dbConnect from '../../../lib/dbConnect.js';

export default async function handler(req, res) {
  await dbConnect();

  const { id } = req.query;

  switch (req.method) {
    case 'GET':
      return getProduct(req, res, id);
    case 'PUT':
      return updateProduct(req, res, id);
    case 'DELETE':
      return deleteProduct(req, res, id);
    default:
      return res.status(405).json({ message: 'Method not allowed' });
  }
}

async function getProduct(req, res, id) {
  try {
    const product = await Product.findById(id)
      .populate('featuredImage', 'url alt filename')
      .populate('images', 'url alt filename')
      .populate('createdBy', 'name email')
      .populate('relatedProducts', 'title slug price featuredImage')
      .populate({
        path: 'reviews.user',
        select: 'name'
      });

    if (!product) {
      return res.status(404).json({
        success: false,
        message: 'Product not found'
      });
    }

    res.status(200).json({
      success: true,
      data: product
    });
  } catch (error) {
    console.error('Get product error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch product',
      error: error.message 
    });
  }
}

async function updateProduct(req, res, id) {
  try {
    const product = await Product.findByIdAndUpdate(
      id,
      { ...req.body, updatedBy: req.user?.id },
      { new: true, runValidators: true }
    )
      .populate('featuredImage', 'url alt filename')
      .populate('images', 'url alt filename')
      .populate('createdBy', 'name email');

    if (!product) {
      return res.status(404).json({
        success: false,
        message: 'Product not found'
      });
    }

    res.status(200).json({
      success: true,
      data: product,
      message: 'Product updated successfully'
    });
  } catch (error) {
    console.error('Update product error:', error);
    
    if (error.code === 11000) {
      return res.status(400).json({
        success: false,
        message: 'Product with this slug already exists'
      });
    }
    
    if (error.name === 'ValidationError') {
      const errors = Object.values(error.errors).map(err => err.message);
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors
      });
    }

    res.status(500).json({ 
      success: false, 
      message: 'Failed to update product',
      error: error.message 
    });
  }
}

async function deleteProduct(req, res, id) {
  try {
    const product = await Product.findByIdAndDelete(id);

    if (!product) {
      return res.status(404).json({
        success: false,
        message: 'Product not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Product deleted successfully'
    });
  } catch (error) {
    console.error('Delete product error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to delete product',
      error: error.message 
    });
  }
}