import mongoose from 'mongoose';
import User from '../models/User.js';
import Product from '../models/SimpleProduct.js';
import Category from '../models/Category.js';
import Customer from '../models/Customer.js';
import Order from '../models/Order.js';
import { Role } from '../models/Role.js';

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/ecommerce-api-clean');
    console.log('MongoDB connected successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

const clearDatabase = async () => {
  try {
    await User.deleteMany({});
    await Product.deleteMany({});
    await Category.deleteMany({});
    await Customer.deleteMany({});
    await Order.deleteMany({});
    await Role.deleteMany({});
    console.log('Database cleared');
  } catch (error) {
    console.error('Error clearing database:', error);
  }
};

const seedData = async () => {
  try {
    await connectDB();
    await clearDatabase();
    
    console.log('🌱 Starting database seeding...');
    
    // Create roles
    const adminRole = await Role.create({
      name: 'admin',
      permission: { create: true, read: true, update: true, delete: true },
      routes: ['*'],
      description: 'Full system access',
      isSystemRole: true
    });
    
    const customerRole = await Role.create({
      name: 'customer',
      permission: { create: false, read: true, update: false, delete: false },
      routes: ['/api/products', '/api/orders'],
      description: 'Customer access',
      isSystemRole: true
    });
    
    // Create admin user
    const adminUser = await User.create({
      email: 'admin@ecommerce.com',
      password: 'admin123',
      name: 'Admin User',
      role: 'admin',
      status: 'active'
    });
    
    // Create categories
    const electronics = await Category.create({
      name: 'Electronics',
      slug: 'electronics',
      description: 'Electronic devices and gadgets',
      isActive: true
    });
    
    const smartphones = await Category.create({
      name: 'Smartphones',
      slug: 'smartphones',
      description: 'Mobile phones and accessories',
      parent: electronics._id,
      isActive: true
    });
    
    const fashion = await Category.create({
      name: 'Fashion',
      slug: 'fashion',
      description: 'Clothing and accessories',
      isActive: true
    });
    
    // Create products
    const products = [
      {
        title: 'iPhone 15 Pro Max',
        slug: 'iphone-15-pro-max',
        description: 'The most advanced iPhone with titanium design, A17 Pro chip, and professional camera system.',
        shortDescription: 'Latest iPhone with titanium design and A17 Pro chip',
        category: 'Smartphones',
        brand: 'Apple',
        price: 1199.99,
        compareAtPrice: 1299.99,
        cost: 800.00,
        inventory: { quantity: 50, trackQuantity: true },
        tags: ['smartphone', 'apple', 'iphone', 'premium', 'latest'],
        status: 'active',
        featured: true,
        createdBy: adminUser._id
      },
      {
        title: 'Samsung Galaxy S24 Ultra',
        slug: 'samsung-galaxy-s24-ultra',
        description: 'Premium Android smartphone with S Pen, advanced AI features, and exceptional camera capabilities.',
        shortDescription: 'Premium Samsung flagship with S Pen and AI features',
        category: 'Smartphones',
        brand: 'Samsung',
        price: 1099.99,
        compareAtPrice: 1199.99,
        cost: 750.00,
        inventory: { quantity: 40 },
        tags: ['smartphone', 'samsung', 'galaxy', 'android', 's-pen'],
        status: 'active',
        featured: true,
        createdBy: adminUser._id
      },
      {
        title: 'MacBook Pro 16-inch M3 Max',
        slug: 'macbook-pro-16-inch-m3-max',
        description: 'Professional laptop with M3 Max chip, stunning Liquid Retina XDR display, and all-day battery life.',
        shortDescription: 'Professional MacBook with M3 Max chip',
        category: 'Electronics',
        brand: 'Apple',
        price: 2499.99,
        compareAtPrice: 2699.99,
        cost: 1800.00,
        inventory: { quantity: 25 },
        tags: ['laptop', 'apple', 'macbook', 'professional', 'm3-max'],
        status: 'active',
        featured: true,
        createdBy: adminUser._id
      },
      {
        title: 'AirPods Pro (3rd generation)',
        slug: 'airpods-pro-3rd-generation',
        description: 'Premium wireless earbuds with active noise cancellation and spatial audio.',
        shortDescription: 'Premium wireless earbuds with ANC',
        category: 'Electronics',
        brand: 'Apple',
        price: 249.99,
        cost: 150.00,
        inventory: { quantity: 100 },
        tags: ['earbuds', 'apple', 'airpods', 'wireless', 'noise-cancellation'],
        status: 'active',
        featured: true,
        createdBy: adminUser._id
      },
      {
        title: 'Classic Cotton T-Shirt',
        slug: 'classic-cotton-t-shirt',
        description: 'Premium cotton t-shirt with comfortable fit and durable construction.',
        shortDescription: 'Premium cotton t-shirt',
        category: 'Fashion',
        brand: 'BasicWear',
        price: 29.99,
        cost: 12.00,
        inventory: { quantity: 200 },
        tags: ['t-shirt', 'cotton', 'basic', 'comfortable', 'casual'],
        status: 'active',
        createdBy: adminUser._id
      }
    ];
    
    const createdProducts = [];
    for (const productData of products) {
      const product = await Product.create(productData);
      createdProducts.push(product);
      console.log(`Created product: ${product.title}`);
    }
    
    // Create customers
    const customers = [
      {
        email: 'john.doe@example.com',
        password: 'customer123',
        firstName: 'John',
        lastName: 'Doe',
        phone: '+1-555-0101',
        addresses: [{
          type: 'shipping',
          firstName: 'John',
          lastName: 'Doe',
          address1: '123 Main St',
          city: 'New York',
          state: 'NY',
          postalCode: '10001',
          country: 'USA',
          phone: '+1-555-0101',
          isDefault: true
        }]
      },
      {
        email: 'jane.smith@example.com',
        password: 'customer123',
        firstName: 'Jane',
        lastName: 'Smith',
        phone: '+1-555-0102',
        addresses: [{
          type: 'shipping',
          firstName: 'Jane',
          lastName: 'Smith',
          address1: '456 Oak Ave',
          city: 'Los Angeles',
          state: 'CA',
          postalCode: '90210',
          country: 'USA',
          phone: '+1-555-0102',
          isDefault: true
        }]
      }
    ];
    
    const createdCustomers = [];
    for (const customerData of customers) {
      const customer = await Customer.create(customerData);
      createdCustomers.push(customer);
      console.log(`Created customer: ${customer.email}`);
    }
    
    // Create sample orders
    const order1 = await Order.create({
      orderNumber: 'ORD-' + Date.now() + '-001',
      customer: createdCustomers[0]._id,
      items: [{
        product: createdProducts[0]._id,
        quantity: 1,
        price: createdProducts[0].price,
        total: createdProducts[0].price
      }],
      subtotal: createdProducts[0].price,
      tax: createdProducts[0].price * 0.08,
      shipping: 15.99,
      total: createdProducts[0].price + (createdProducts[0].price * 0.08) + 15.99,
      shippingAddress: createdCustomers[0].addresses[0],
      billingAddress: createdCustomers[0].addresses[0],
      status: 'delivered',
      paymentStatus: 'paid',
      paymentMethod: 'credit_card'
    });
    
    console.log(`Created order: ${order1.orderNumber}`);
    
    console.log('✅ Database seeding completed successfully!');
    console.log('\n📊 Summary:');
    console.log(`- Users: ${await User.countDocuments()}`);
    console.log(`- Roles: ${await Role.countDocuments()}`);
    console.log(`- Categories: ${await Category.countDocuments()}`);
    console.log(`- Products: ${await Product.countDocuments()}`);
    console.log(`- Customers: ${await Customer.countDocuments()}`);
    console.log(`- Orders: ${await Order.countDocuments()}`);
    
    console.log('\n🔑 Test Credentials:');
    console.log('Admin: admin@ecommerce.com / admin123');
    console.log('Customer: john.doe@example.com / customer123');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Seeding failed:', error);
    process.exit(1);
  }
};

seedData();