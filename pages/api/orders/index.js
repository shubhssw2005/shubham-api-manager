import Order from '../../../models/Order.js';
import Customer from '../../../models/Customer.js';
import Product from '../../../models/SimpleProduct.js';
import dbConnect from '../../../lib/dbConnect.js';

export default async function handler(req, res) {
  await dbConnect();

  switch (req.method) {
    case 'GET':
      return getOrders(req, res);
    case 'POST':
      return createOrder(req, res);
    default:
      return res.status(405).json({ message: 'Method not allowed' });
  }
}

async function getOrders(req, res) {
  try {
    const {
      page = 1,
      limit = 20,
      status,
      paymentStatus,
      customer,
      sort = '-createdAt'
    } = req.query;

    const query = {};
    
    if (status) query.status = status;
    if (paymentStatus) query.paymentStatus = paymentStatus;
    if (customer) query.customer = customer;

    const skip = (parseInt(page) - 1) * parseInt(limit);
    const orders = await Order.find(query)
      .populate('customer', 'firstName lastName email')
      .populate('items.product', 'title slug price featuredImage')
      .sort(sort)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Order.countDocuments(query);
    const totalPages = Math.ceil(total / parseInt(limit));
    
    res.status(200).json({
      success: true,
      data: orders,
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
    console.error('Get orders error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch orders',
      error: error.message 
    });
  }
}

async function createOrder(req, res) {
  try {
    const { customerId, items, shippingAddress, billingAddress, paymentMethod } = req.body;

    // Validate customer
    const customer = await Customer.findById(customerId);
    if (!customer) {
      return res.status(400).json({
        success: false,
        message: 'Customer not found'
      });
    }

    // Validate and calculate items
    const orderItems = [];
    let subtotal = 0;

    for (const item of items) {
      const product = await Product.findById(item.productId);
      if (!product) {
        return res.status(400).json({
          success: false,
          message: `Product not found: ${item.productId}`
        });
      }

      if (product.status !== 'active') {
        return res.status(400).json({
          success: false,
          message: `Product is not available: ${product.title}`
        });
      }

      // Check inventory
      if (product.inventory.trackQuantity && product.inventory.quantity < item.quantity) {
        return res.status(400).json({
          success: false,
          message: `Insufficient inventory for: ${product.title}`
        });
      }

      const itemTotal = product.price * item.quantity;
      subtotal += itemTotal;

      orderItems.push({
        product: product._id,
        variant: item.variantId || null,
        quantity: item.quantity,
        price: product.price,
        total: itemTotal
      });

      // Update inventory
      if (product.inventory.trackQuantity) {
        await product.updateInventory(item.quantity, 'subtract');
      }
    }

    // Calculate totals (simplified - you'd implement tax calculation based on location)
    const tax = subtotal * 0.08; // 8% tax rate
    const shipping = subtotal > 100 ? 0 : 15.99; // Free shipping over $100
    const total = subtotal + tax + shipping;

    const orderData = {
      customer: customerId,
      items: orderItems,
      subtotal,
      tax,
      shipping,
      total,
      shippingAddress,
      billingAddress,
      paymentMethod,
      status: 'pending',
      paymentStatus: 'pending'
    };

    const order = await Order.create(orderData);
    await order.populate([
      { path: 'customer', select: 'firstName lastName email' },
      { path: 'items.product', select: 'title slug price featuredImage' }
    ]);

    // Update customer metrics
    await customer.updateMetrics();

    res.status(201).json({
      success: true,
      data: order,
      message: 'Order created successfully'
    });
  } catch (error) {
    console.error('Create order error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to create order',
      error: error.message 
    });
  }
}