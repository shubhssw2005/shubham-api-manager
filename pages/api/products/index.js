import Product from '../../../models/SimpleProduct.js';
import Category from '../../../models/Category.js';
import dbConnect from '../../../lib/dbConnect.js';

export default async function handler(req, res) {
  await dbConnect();

  switch (req.method) {
    case 'GET':
      return getProducts(req, res);
    case 'POST':
      return createProduct(req, res);
    default:
      return res.status(405).json({ message: 'Method not allowed' });
  }
}

async function getProducts(req, res) {
  try {
    const {
      page = 1,
      limit = 20,
      category,
      brand,
      minPrice,
      maxPrice,
      search,
      sort = '-createdAt',
      featured,
      status = 'active'
    } = req.query;

    const query = { status, visibility: 'public' };
    
    // Category filter
    if (category) {
      query.category = new RegExp(category, 'i');
    }
    
    // Brand filter
    if (brand) {
      query.brand = new RegExp(brand, 'i');
    }
    
    // Price range filter
    if (minPrice || maxPrice) {
      query.price = {};
      if (minPrice) query.price.$gte = parseFloat(minPrice);
      if (maxPrice) query.price.$lte = parseFloat(maxPrice);
    }
    
    // Featured filter
    if (featured === 'true') {
      query.featured = true;
    }
    
    // Search filter
    if (search) {
      query.$text = { $search: search };
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);
    const products = await Product.find(query)
      .populate('featuredImage', 'url alt filename')
      .populate('images', 'url alt filename')
      .populate('createdBy', 'name email')
      .sort(sort)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Product.countDocuments(query);
    const totalPages = Math.ceil(total / parseInt(limit));
    
    res.status(200).json({
      success: true,
      data: products,
      pagination: {
        page: parseInt(page),
        pages: totalPages,
        total,
        limit: parseInt(limit),
        hasNext: parseInt(page) < totalPages,
        hasPrev: parseInt(page) > 1
      }
    });
  } catch (error) {
    console.error('Get products error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch products',
      error: error.message 
    });
  }
}

async function createProduct(req, res) {
  try {
    const product = await Product.create(req.body);
    await product.populate([
      { path: 'featuredImage', select: 'url alt filename' },
      { path: 'images', select: 'url alt filename' },
      { path: 'createdBy', select: 'name email' }
    ]);

    res.status(201).json({
      success: true,
      data: product,
      message: 'Product created successfully'
    });
  } catch (error) {
    console.error('Create product error:', error);
    
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
      message: 'Failed to create product',
      error: error.message 
    });
  }
}