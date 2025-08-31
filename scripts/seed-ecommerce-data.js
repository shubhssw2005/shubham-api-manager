import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

// Import models
import User from '../models/User.js';
import Product from '../models/Product.js';
import Category from '../models/Category.js';
import Order from '../models/Order.js';
import Customer from '../models/Customer.js';
import { Role } from '../models/Role.js';

// Database connection
const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/ecommerce-api', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('MongoDB connected successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

// Sample data generators
const generateUsers = async () => {
  const users = [
    {
      email: 'admin@ecommerce.com',
      password: 'admin123',
      name: 'Admin User',
      role: 'admin',
      status: 'active'
    },
    {
      email: 'manager@ecommerce.com',
      password: 'manager123',
      name: 'Store Manager',
      role: 'admin',
      status: 'active'
    }
  ];

  for (let user of users) {
    const existingUser = await User.findOne({ email: user.email });
    if (!existingUser) {
      await User.create(user);
      console.log(`Created user: ${user.email}`);
    }
  }
};

const generateRoles = async () => {
  const roles = [
    {
      name: 'admin',
      permission: {
        create: true,
        read: true,
        update: true,
        delete: true
      },
      routes: ['*'],
      description: 'Full system access',
      isSystemRole: true
    },
    {
      name: 'customer',
      permission: {
        create: false,
        read: true,
        update: false,
        delete: false
      },
      routes: ['/api/products', '/api/orders'],
      description: 'Customer access',
      isSystemRole: true
    }
  ];

  for (let role of roles) {
    const existingRole = await Role.findOne({ name: role.name });
    if (!existingRole) {
      await Role.create(role);
      console.log(`Created role: ${role.name}`);
    }
  }
};

const generateCategories = async () => {
  const categories = [
    // Electronics
    { name: 'Electronics', description: 'Electronic devices and gadgets' },
    { name: 'Smartphones', description: 'Mobile phones and accessories', parent: 'Electronics' },
    { name: 'Laptops', description: 'Laptops and notebooks', parent: 'Electronics' },
    { name: 'Tablets', description: 'Tablets and e-readers', parent: 'Electronics' },
    { name: 'Audio', description: 'Headphones, speakers, and audio equipment', parent: 'Electronics' },
    { name: 'Gaming', description: 'Gaming consoles and accessories', parent: 'Electronics' },
    
    // Fashion
    { name: 'Fashion', description: 'Clothing and accessories' },
    { name: 'Men\'s Clothing', description: 'Men\'s fashion and apparel', parent: 'Fashion' },
    { name: 'Women\'s Clothing', description: 'Women\'s fashion and apparel', parent: 'Fashion' },
    { name: 'Shoes', description: 'Footwear for all occasions', parent: 'Fashion' },
    { name: 'Accessories', description: 'Fashion accessories', parent: 'Fashion' },
    
    // Home & Garden
    { name: 'Home & Garden', description: 'Home improvement and garden supplies' },
    { name: 'Furniture', description: 'Home and office furniture', parent: 'Home & Garden' },
    { name: 'Kitchen', description: 'Kitchen appliances and tools', parent: 'Home & Garden' },
    { name: 'Garden', description: 'Garden tools and plants', parent: 'Home & Garden' },
    
    // Sports & Outdoors
    { name: 'Sports & Outdoors', description: 'Sports equipment and outdoor gear' },
    { name: 'Fitness', description: 'Fitness equipment and accessories', parent: 'Sports & Outdoors' },
    { name: 'Outdoor Recreation', description: 'Camping, hiking, and outdoor activities', parent: 'Sports & Outdoors' },
    
    // Books & Media
    { name: 'Books & Media', description: 'Books, movies, and music' },
    { name: 'Books', description: 'Physical and digital books', parent: 'Books & Media' },
    { name: 'Movies', description: 'DVDs, Blu-rays, and digital movies', parent: 'Books & Media' },
    { name: 'Music', description: 'CDs, vinyl, and digital music', parent: 'Books & Media' }
  ];

  const createdCategories = {};
  
  // First pass: create parent categories
  for (let categoryData of categories.filter(c => !c.parent)) {
    const existing = await Category.findOne({ name: categoryData.name });
    if (!existing) {
      // Generate slug if not provided
      if (!categoryData.slug) {
        categoryData.slug = categoryData.name
          .toLowerCase()
          .replace(/[^a-z0-9\s]+/g, '')
          .replace(/\s+/g, '-')
          .replace(/^-+|-+$/g, '');
      }
      const category = await Category.create(categoryData);
      createdCategories[categoryData.name] = category._id;
      console.log(`Created category: ${categoryData.name}`);
    } else {
      createdCategories[categoryData.name] = existing._id;
    }
  }
  
  // Second pass: create child categories
  for (let categoryData of categories.filter(c => c.parent)) {
    const existing = await Category.findOne({ name: categoryData.name });
    if (!existing) {
      const parentId = createdCategories[categoryData.parent];
      if (parentId) {
        categoryData.parent = parentId;
        // Generate slug if not provided
        if (!categoryData.slug) {
          categoryData.slug = categoryData.name
            .toLowerCase()
            .replace(/[^a-z0-9\s]+/g, '')
            .replace(/\s+/g, '-')
            .replace(/^-+|-+$/g, '');
        }
        const category = await Category.create(categoryData);
        createdCategories[categoryData.name] = category._id;
        console.log(`Created subcategory: ${categoryData.name}`);
      }
    }
  }
  
  return createdCategories;
};

const generateCustomers = async () => {
  const customers = [
    {
      email: 'john.doe@example.com',
      password: 'customer123',
      firstName: 'John',
      lastName: 'Doe',
      phone: '+1-555-0101',
      addresses: [
        {
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
        }
      ]
    },
    {
      email: 'jane.smith@example.com',
      password: 'customer123',
      firstName: 'Jane',
      lastName: 'Smith',
      phone: '+1-555-0102',
      addresses: [
        {
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
        }
      ]
    },
    {
      email: 'mike.johnson@example.com',
      password: 'customer123',
      firstName: 'Mike',
      lastName: 'Johnson',
      phone: '+1-555-0103',
      addresses: [
        {
          type: 'shipping',
          firstName: 'Mike',
          lastName: 'Johnson',
          address1: '789 Pine Rd',
          city: 'Chicago',
          state: 'IL',
          postalCode: '60601',
          country: 'USA',
          phone: '+1-555-0103',
          isDefault: true
        }
      ]
    }
  ];

  const createdCustomers = [];
  for (let customerData of customers) {
    const existing = await Customer.findOne({ email: customerData.email });
    if (!existing) {
      const customer = await Customer.create(customerData);
      createdCustomers.push(customer);
      console.log(`Created customer: ${customerData.email}`);
    } else {
      createdCustomers.push(existing);
    }
  }
  
  return createdCustomers;
};

const generateProducts = async (adminUser) => {
  const products = [
    // Electronics - Smartphones
    {
      title: 'iPhone 15 Pro Max',
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
      specifications: {
        weight: 0.221,
        dimensions: { length: 159.9, width: 76.7, height: 8.25 },
        color: 'Natural Titanium',
        material: 'Titanium'
      },
      variants: [
        {
          name: '128GB Natural Titanium',
          sku: 'IPH15PM-128-NT',
          price: 1199.99,
          inventory: { quantity: 20 },
          attributes: { storage: '128GB', color: 'Natural Titanium' }
        },
        {
          name: '256GB Natural Titanium',
          sku: 'IPH15PM-256-NT',
          price: 1299.99,
          inventory: { quantity: 15 },
          attributes: { storage: '256GB', color: 'Natural Titanium' }
        },
        {
          name: '512GB Natural Titanium',
          sku: 'IPH15PM-512-NT',
          price: 1499.99,
          inventory: { quantity: 10 },
          attributes: { storage: '512GB', color: 'Natural Titanium' }
        }
      ]
    },
    {
      title: 'Samsung Galaxy S24 Ultra',
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
      featured: true
    },
    {
      title: 'Google Pixel 8 Pro',
      description: 'Google\'s flagship phone with advanced AI photography and pure Android experience.',
      shortDescription: 'Google flagship with AI photography',
      category: 'Smartphones',
      brand: 'Google',
      price: 899.99,
      cost: 600.00,
      inventory: { quantity: 30 },
      tags: ['smartphone', 'google', 'pixel', 'android', 'ai-camera'],
      status: 'active'
    },

    // Electronics - Laptops
    {
      title: 'MacBook Pro 16-inch M3 Max',
      description: 'Professional laptop with M3 Max chip, stunning Liquid Retina XDR display, and all-day battery life.',
      shortDescription: 'Professional MacBook with M3 Max chip',
      category: 'Laptops',
      brand: 'Apple',
      price: 2499.99,
      compareAtPrice: 2699.99,
      cost: 1800.00,
      inventory: { quantity: 25 },
      tags: ['laptop', 'apple', 'macbook', 'professional', 'm3-max'],
      status: 'active',
      featured: true,
      specifications: {
        weight: 2.14,
        dimensions: { length: 355.7, width: 248.1, height: 16.8 },
        material: 'Aluminum'
      }
    },
    {
      title: 'Dell XPS 13 Plus',
      description: 'Ultra-thin laptop with InfinityEdge display, premium materials, and powerful performance.',
      shortDescription: 'Ultra-thin premium laptop',
      category: 'Laptops',
      brand: 'Dell',
      price: 1299.99,
      cost: 900.00,
      inventory: { quantity: 35 },
      tags: ['laptop', 'dell', 'xps', 'ultrabook', 'premium'],
      status: 'active'
    },
    {
      title: 'ThinkPad X1 Carbon Gen 11',
      description: 'Business laptop with legendary ThinkPad reliability, security features, and lightweight design.',
      shortDescription: 'Business laptop with premium build quality',
      category: 'Laptops',
      brand: 'Lenovo',
      price: 1599.99,
      cost: 1100.00,
      inventory: { quantity: 20 },
      tags: ['laptop', 'lenovo', 'thinkpad', 'business', 'carbon-fiber'],
      status: 'active'
    },

    // Electronics - Audio
    {
      title: 'AirPods Pro (3rd generation)',
      description: 'Premium wireless earbuds with active noise cancellation and spatial audio.',
      shortDescription: 'Premium wireless earbuds with ANC',
      category: 'Audio',
      brand: 'Apple',
      price: 249.99,
      cost: 150.00,
      inventory: { quantity: 100 },
      tags: ['earbuds', 'apple', 'airpods', 'wireless', 'noise-cancellation'],
      status: 'active',
      featured: true
    },
    {
      title: 'Sony WH-1000XM5',
      description: 'Industry-leading noise canceling headphones with exceptional sound quality.',
      shortDescription: 'Premium noise-canceling headphones',
      category: 'Audio',
      brand: 'Sony',
      price: 399.99,
      compareAtPrice: 449.99,
      cost: 250.00,
      inventory: { quantity: 60 },
      tags: ['headphones', 'sony', 'noise-cancellation', 'premium', 'wireless'],
      status: 'active'
    },

    // Fashion - Men's Clothing
    {
      title: 'Classic Cotton T-Shirt',
      description: 'Premium cotton t-shirt with comfortable fit and durable construction.',
      shortDescription: 'Premium cotton t-shirt',
      category: 'Men\'s Clothing',
      brand: 'BasicWear',
      price: 29.99,
      cost: 12.00,
      inventory: { quantity: 200 },
      tags: ['t-shirt', 'cotton', 'basic', 'comfortable', 'casual'],
      status: 'active',
      variants: [
        {
          name: 'Small Black',
          sku: 'TSHIRT-S-BLK',
          price: 29.99,
          inventory: { quantity: 50 },
          attributes: { size: 'S', color: 'Black' }
        },
        {
          name: 'Medium Black',
          sku: 'TSHIRT-M-BLK',
          price: 29.99,
          inventory: { quantity: 60 },
          attributes: { size: 'M', color: 'Black' }
        },
        {
          name: 'Large Black',
          sku: 'TSHIRT-L-BLK',
          price: 29.99,
          inventory: { quantity: 50 },
          attributes: { size: 'L', color: 'Black' }
        },
        {
          name: 'Small White',
          sku: 'TSHIRT-S-WHT',
          price: 29.99,
          inventory: { quantity: 40 },
          attributes: { size: 'S', color: 'White' }
        }
      ]
    },
    {
      title: 'Premium Denim Jeans',
      description: 'High-quality denim jeans with modern fit and premium finishing.',
      shortDescription: 'Premium denim jeans with modern fit',
      category: 'Men\'s Clothing',
      brand: 'DenimCo',
      price: 89.99,
      compareAtPrice: 119.99,
      cost: 45.00,
      inventory: { quantity: 150 },
      tags: ['jeans', 'denim', 'premium', 'modern-fit', 'casual'],
      status: 'active'
    },

    // Fashion - Women's Clothing
    {
      title: 'Elegant Summer Dress',
      description: 'Beautiful summer dress perfect for casual and semi-formal occasions.',
      shortDescription: 'Elegant summer dress for various occasions',
      category: 'Women\'s Clothing',
      brand: 'FashionForward',
      price: 79.99,
      cost: 35.00,
      inventory: { quantity: 80 },
      tags: ['dress', 'summer', 'elegant', 'casual', 'semi-formal'],
      status: 'active',
      featured: true
    },
    {
      title: 'Professional Blazer',
      description: 'Tailored blazer perfect for business and professional settings.',
      shortDescription: 'Tailored professional blazer',
      category: 'Women\'s Clothing',
      brand: 'BusinessAttire',
      price: 149.99,
      cost: 75.00,
      inventory: { quantity: 60 },
      tags: ['blazer', 'professional', 'business', 'tailored', 'formal'],
      status: 'active'
    },

    // Home & Garden - Kitchen
    {
      title: 'Professional Chef Knife Set',
      description: 'High-quality stainless steel knife set for professional and home cooking.',
      shortDescription: 'Professional stainless steel knife set',
      category: 'Kitchen',
      brand: 'ChefMaster',
      price: 199.99,
      compareAtPrice: 249.99,
      cost: 100.00,
      inventory: { quantity: 40 },
      tags: ['knives', 'kitchen', 'professional', 'stainless-steel', 'cooking'],
      status: 'active'
    },
    {
      title: 'Smart Coffee Maker',
      description: 'WiFi-enabled coffee maker with app control and programmable brewing.',
      shortDescription: 'Smart WiFi coffee maker with app control',
      category: 'Kitchen',
      brand: 'BrewTech',
      price: 299.99,
      cost: 180.00,
      inventory: { quantity: 30 },
      tags: ['coffee-maker', 'smart', 'wifi', 'programmable', 'kitchen-appliance'],
      status: 'active',
      featured: true
    },

    // Sports & Outdoors - Fitness
    {
      title: 'Adjustable Dumbbell Set',
      description: 'Space-saving adjustable dumbbells perfect for home workouts.',
      shortDescription: 'Adjustable dumbbells for home workouts',
      category: 'Fitness',
      brand: 'FitGear',
      price: 399.99,
      cost: 200.00,
      inventory: { quantity: 25 },
      tags: ['dumbbells', 'fitness', 'home-gym', 'adjustable', 'strength-training'],
      status: 'active'
    },
    {
      title: 'Yoga Mat Premium',
      description: 'High-quality non-slip yoga mat with excellent cushioning and durability.',
      shortDescription: 'Premium non-slip yoga mat',
      category: 'Fitness',
      brand: 'YogaLife',
      price: 59.99,
      cost: 25.00,
      inventory: { quantity: 100 },
      tags: ['yoga-mat', 'fitness', 'yoga', 'non-slip', 'premium'],
      status: 'active'
    },

    // Books & Media
    {
      title: 'The Art of Programming',
      description: 'Comprehensive guide to modern programming practices and methodologies.',
      shortDescription: 'Comprehensive programming guide',
      category: 'Books',
      brand: 'TechBooks',
      price: 49.99,
      cost: 20.00,
      inventory: { quantity: 75 },
      tags: ['book', 'programming', 'technology', 'education', 'guide'],
      status: 'active'
    },
    {
      title: 'Wireless Bluetooth Speaker',
      description: 'Portable Bluetooth speaker with excellent sound quality and long battery life.',
      shortDescription: 'Portable Bluetooth speaker',
      category: 'Audio',
      brand: 'SoundWave',
      price: 79.99,
      compareAtPrice: 99.99,
      cost: 40.00,
      inventory: { quantity: 80 },
      tags: ['speaker', 'bluetooth', 'portable', 'wireless', 'audio'],
      status: 'active'
    }
  ];

  const createdProducts = [];
  for (let productData of products) {
    const existing = await Product.findOne({ title: productData.title });
    if (!existing) {
      productData.createdBy = adminUser._id;
      // Generate slug if not provided
      if (!productData.slug) {
        productData.slug = productData.title
          .toLowerCase()
          .replace(/[^a-z0-9\s]+/g, '')
          .replace(/\s+/g, '-')
          .replace(/^-+|-+$/g, '');
      }
      const product = await Product.create(productData);
      
      // Add some reviews to featured products
      if (productData.featured) {
        const reviews = [
          {
            user: adminUser._id,
            rating: 5,
            title: 'Excellent product!',
            comment: 'Really impressed with the quality and performance. Highly recommended!'
          },
          {
            user: adminUser._id,
            rating: 4,
            title: 'Great value',
            comment: 'Good product for the price. Would buy again.'
          }
        ];
        
        product.reviews = reviews;
        await product.updateRating();
      }
      
      createdProducts.push(product);
      console.log(`Created product: ${productData.title}`);
    } else {
      createdProducts.push(existing);
    }
  }
  
  return createdProducts;
};

const generateOrders = async (customers, products) => {
  const orders = [
    {
      customer: customers[0]._id,
      items: [
        {
          product: products[0]._id,
          quantity: 1,
          price: products[0].price,
          total: products[0].price
        },
        {
          product: products[7]._id,
          quantity: 2,
          price: products[7].price,
          total: products[7].price * 2
        }
      ],
      shippingAddress: customers[0].addresses[0],
      billingAddress: customers[0].addresses[0],
      status: 'delivered',
      paymentStatus: 'paid',
      paymentMethod: 'credit_card',
      shippingMethod: 'standard',
      tax: 120.00,
      shipping: 15.99
    },
    {
      customer: customers[1]._id,
      items: [
        {
          product: products[3]._id,
          quantity: 1,
          price: products[3].price,
          total: products[3].price
        }
      ],
      shippingAddress: customers[1].addresses[0],
      billingAddress: customers[1].addresses[0],
      status: 'shipped',
      paymentStatus: 'paid',
      paymentMethod: 'paypal',
      shippingMethod: 'express',
      tax: 200.00,
      shipping: 25.99,
      trackingNumber: 'TRK123456789'
    },
    {
      customer: customers[2]._id,
      items: [
        {
          product: products[8]._id,
          quantity: 1,
          price: products[8].price,
          total: products[8].price
        },
        {
          product: products[12]._id,
          quantity: 3,
          price: products[12].price,
          total: products[12].price * 3
        }
      ],
      shippingAddress: customers[2].addresses[0],
      billingAddress: customers[2].addresses[0],
      status: 'processing',
      paymentStatus: 'paid',
      paymentMethod: 'credit_card',
      shippingMethod: 'standard',
      tax: 45.00,
      shipping: 12.99
    }
  ];

  for (let orderData of orders) {
    const existing = await Order.findOne({ 
      customer: orderData.customer,
      'items.product': { $in: orderData.items.map(item => item.product) }
    });
    
    if (!existing) {
      const order = await Order.create(orderData);
      console.log(`Created order: ${order.orderNumber}`);
      
      // Update customer metrics
      const customer = await Customer.findById(orderData.customer);
      if (customer) {
        await customer.updateMetrics();
      }
    }
  }
};

// Main seeding function
const seedDatabase = async () => {
  try {
    await connectDB();
    
    console.log('🌱 Starting database seeding...');
    
    // Generate base data
    await generateRoles();
    await generateUsers();
    
    // Get admin user for product creation
    const adminUser = await User.findOne({ email: 'admin@ecommerce.com' });
    if (!adminUser) {
      throw new Error('Admin user not found');
    }
    
    // Generate ecommerce data
    await generateCategories();
    const customers = await generateCustomers();
    const products = await generateProducts(adminUser);
    await generateOrders(customers, products);
    
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
    console.log('Manager: manager@ecommerce.com / manager123');
    console.log('Customer: john.doe@example.com / customer123');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Seeding failed:', error);
    process.exit(1);
  }
};

// Run seeding if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  seedDatabase();
}

export default seedDatabase;