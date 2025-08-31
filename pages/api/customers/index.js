import Customer from '../../../models/Customer.js';
import dbConnect from '../../../lib/dbConnect.js';

export default async function handler(req, res) {
  await dbConnect();

  switch (req.method) {
    case 'GET':
      return getCustomers(req, res);
    case 'POST':
      return createCustomer(req, res);
    default:
      return res.status(405).json({ message: 'Method not allowed' });
  }
}

async function getCustomers(req, res) {
  try {
    const {
      page = 1,
      limit = 20,
      search,
      isActive,
      sort = '-createdAt'
    } = req.query;

    const query = {};
    
    if (isActive !== undefined) {
      query.isActive = isActive === 'true';
    }
    
    if (search) {
      query.$or = [
        { firstName: new RegExp(search, 'i') },
        { lastName: new RegExp(search, 'i') },
        { email: new RegExp(search, 'i') }
      ];
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);
    const customers = await Customer.find(query)
      .sort(sort)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Customer.countDocuments(query);
    const totalPages = Math.ceil(total / parseInt(limit));
    
    res.status(200).json({
      success: true,
      data: customers,
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
    console.error('Get customers error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch customers',
      error: error.message 
    });
  }
}

async function createCustomer(req, res) {
  try {
    const customer = await Customer.create(req.body);

    res.status(201).json({
      success: true,
      data: customer,
      message: 'Customer created successfully'
    });
  } catch (error) {
    console.error('Create customer error:', error);
    
    if (error.code === 11000) {
      return res.status(400).json({
        success: false,
        message: 'Customer with this email already exists'
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
      message: 'Failed to create customer',
      error: error.message 
    });
  }
}