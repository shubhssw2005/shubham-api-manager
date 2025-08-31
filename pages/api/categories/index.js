import Category from '../../../models/Category.js';
import dbConnect from '../../../lib/dbConnect.js';

export default async function handler(req, res) {
  await dbConnect();

  switch (req.method) {
    case 'GET':
      return getCategories(req, res);
    case 'POST':
      return createCategory(req, res);
    default:
      return res.status(405).json({ message: 'Method not allowed' });
  }
}

async function getCategories(req, res) {
  try {
    const { parent, includeChildren = 'false' } = req.query;
    
    let query = { isActive: true };
    
    if (parent) {
      query.parent = parent === 'null' ? null : parent;
    }

    const categories = await Category.find(query)
      .populate('image', 'url alt filename')
      .populate('parent', 'name slug')
      .sort('sortOrder name');

    if (includeChildren === 'true') {
      await Category.populate(categories, {
        path: 'children',
        match: { isActive: true },
        select: 'name slug description image sortOrder',
        populate: { path: 'image', select: 'url alt filename' }
      });
    }

    res.status(200).json({
      success: true,
      data: categories
    });
  } catch (error) {
    console.error('Get categories error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch categories',
      error: error.message 
    });
  }
}

async function createCategory(req, res) {
  try {
    const category = await Category.create(req.body);
    await category.populate([
      { path: 'image', select: 'url alt filename' },
      { path: 'parent', select: 'name slug' }
    ]);

    res.status(201).json({
      success: true,
      data: category,
      message: 'Category created successfully'
    });
  } catch (error) {
    console.error('Create category error:', error);
    
    if (error.code === 11000) {
      return res.status(400).json({
        success: false,
        message: 'Category with this name or slug already exists'
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
      message: 'Failed to create category',
      error: error.message 
    });
  }
}