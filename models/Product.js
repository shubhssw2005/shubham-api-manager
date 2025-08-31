import mongoose from 'mongoose';

const VariantSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  sku: {
    type: String,
    required: true,
    trim: true
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  compareAtPrice: {
    type: Number,
    min: 0
  },
  cost: {
    type: Number,
    min: 0
  },
  inventory: {
    quantity: {
      type: Number,
      default: 0,
      min: 0
    },
    trackQuantity: {
      type: Boolean,
      default: true
    },
    allowBackorder: {
      type: Boolean,
      default: false
    }
  },
  attributes: {
    size: String,
    color: String,
    material: String,
    weight: Number,
    dimensions: {
      length: Number,
      width: Number,
      height: Number
    }
  },
  images: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Media'
  }],
  isActive: {
    type: Boolean,
    default: true
  }
}, { _id: true });

const ReviewSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  rating: {
    type: Number,
    required: true,
    min: 1,
    max: 5
  },
  title: {
    type: String,
    trim: true,
    maxlength: 100
  },
  comment: {
    type: String,
    trim: true,
    maxlength: 1000
  },
  verified: {
    type: Boolean,
    default: false
  },
  helpful: {
    count: {
      type: Number,
      default: 0
    },
    users: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }]
  }
}, { timestamps: true });

const ProductSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Product title is required'],
    trim: true,
    maxlength: [200, 'Title cannot exceed 200 characters']
  },
  slug: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  description: {
    type: String,
    required: [true, 'Product description is required'],
    trim: true
  },
  shortDescription: {
    type: String,
    trim: true,
    maxlength: 500
  },
  category: {
    type: String,
    required: true,
    trim: true
  },
  subcategory: {
    type: String,
    trim: true
  },
  brand: {
    type: String,
    trim: true
  },
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  
  // Pricing
  price: {
    type: Number,
    required: true,
    min: 0
  },
  compareAtPrice: {
    type: Number,
    min: 0
  },
  cost: {
    type: Number,
    min: 0
  },
  
  // Inventory
  inventory: {
    quantity: {
      type: Number,
      default: 0,
      min: 0
    },
    trackQuantity: {
      type: Boolean,
      default: true
    },
    allowBackorder: {
      type: Boolean,
      default: false
    },
    lowStockThreshold: {
      type: Number,
      default: 10
    }
  },
  
  // Product variants (sizes, colors, etc.) - simplified for now
  // variants: [VariantSchema],
  
  // Media
  images: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Media'
  }],
  featuredImage: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Media'
  },
  
  // SEO
  seo: {
    metaTitle: String,
    metaDescription: String,
    keywords: [String]
  },
  
  // Product specifications
  specifications: {
    weight: Number,
    dimensions: {
      length: Number,
      width: Number,
      height: Number
    },
    material: String,
    color: String,
    size: String,
    custom: mongoose.Schema.Types.Mixed
  },
  
  // Shipping
  shipping: {
    weight: Number,
    requiresShipping: {
      type: Boolean,
      default: true
    },
    shippingClass: String,
    freeShipping: {
      type: Boolean,
      default: false
    }
  },
  
  // Status and visibility
  status: {
    type: String,
    enum: ['draft', 'active', 'archived', 'out_of_stock'],
    default: 'draft'
  },
  visibility: {
    type: String,
    enum: ['public', 'private', 'hidden'],
    default: 'public'
  },
  featured: {
    type: Boolean,
    default: false
  },
  
  // Reviews and ratings
  reviews: [ReviewSchema],
  rating: {
    average: {
      type: Number,
      default: 0,
      min: 0,
      max: 5
    },
    count: {
      type: Number,
      default: 0
    }
  },
  
  // Sales data
  sales: {
    totalSold: {
      type: Number,
      default: 0
    },
    revenue: {
      type: Number,
      default: 0
    }
  },
  
  // Related products
  relatedProducts: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Product'
  }],
  
  // Timestamps
  publishedAt: Date,
  
  // User who created/manages the product
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  updatedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes
ProductSchema.index({ title: 'text', description: 'text', tags: 'text' });
ProductSchema.index({ slug: 1 });
ProductSchema.index({ category: 1, subcategory: 1 });
ProductSchema.index({ brand: 1 });
ProductSchema.index({ status: 1, visibility: 1 });
ProductSchema.index({ featured: 1 });
ProductSchema.index({ price: 1 });
ProductSchema.index({ 'rating.average': -1 });
ProductSchema.index({ createdAt: -1 });
ProductSchema.index({ publishedAt: -1 });

// Compound indexes
ProductSchema.index({ category: 1, status: 1, visibility: 1 });
ProductSchema.index({ featured: 1, status: 1, visibility: 1 });
ProductSchema.index({ price: 1, status: 1, visibility: 1 });

// Virtuals
ProductSchema.virtual('isOnSale').get(function() {
  return this.compareAtPrice && this.compareAtPrice > this.price;
});

ProductSchema.virtual('discountPercentage').get(function() {
  if (!this.compareAtPrice || this.compareAtPrice <= this.price) return 0;
  return Math.round(((this.compareAtPrice - this.price) / this.compareAtPrice) * 100);
});

ProductSchema.virtual('isInStock').get(function() {
  if (!this.inventory.trackQuantity) return true;
  return this.inventory.quantity > 0 || this.inventory.allowBackorder;
});

ProductSchema.virtual('isLowStock').get(function() {
  if (!this.inventory.trackQuantity) return false;
  return this.inventory.quantity <= this.inventory.lowStockThreshold;
});

// Instance methods
ProductSchema.methods.updateRating = function() {
  if (this.reviews.length === 0) {
    this.rating.average = 0;
    this.rating.count = 0;
  } else {
    const totalRating = this.reviews.reduce((sum, review) => sum + review.rating, 0);
    this.rating.average = Math.round((totalRating / this.reviews.length) * 10) / 10;
    this.rating.count = this.reviews.length;
  }
  return this.save();
};

ProductSchema.methods.addReview = function(userId, rating, title, comment) {
  this.reviews.push({
    user: userId,
    rating,
    title,
    comment
  });
  return this.updateRating();
};

ProductSchema.methods.updateInventory = function(quantity, operation = 'set') {
  if (operation === 'add') {
    this.inventory.quantity += quantity;
  } else if (operation === 'subtract') {
    this.inventory.quantity = Math.max(0, this.inventory.quantity - quantity);
  } else {
    this.inventory.quantity = quantity;
  }
  
  // Update status based on inventory
  if (this.inventory.trackQuantity && this.inventory.quantity === 0 && !this.inventory.allowBackorder) {
    this.status = 'out_of_stock';
  } else if (this.status === 'out_of_stock' && this.inventory.quantity > 0) {
    this.status = 'active';
  }
  
  return this.save();
};

// Static methods
ProductSchema.statics.findPublished = function(options = {}) {
  return this.find({ 
    status: 'active', 
    visibility: 'public',
    publishedAt: { $lte: new Date() }
  }, null, options);
};

ProductSchema.statics.findFeatured = function(options = {}) {
  return this.find({ 
    featured: true,
    status: 'active', 
    visibility: 'public' 
  }, null, options);
};

ProductSchema.statics.findByCategory = function(category, options = {}) {
  return this.find({ 
    category: new RegExp(category, 'i'),
    status: 'active', 
    visibility: 'public' 
  }, null, options);
};

ProductSchema.statics.searchProducts = function(searchTerm, options = {}) {
  return this.find({
    $text: { $search: searchTerm },
    status: 'active',
    visibility: 'public'
  }, { score: { $meta: 'textScore' } }, {
    ...options,
    sort: { score: { $meta: 'textScore' } }
  });
};

ProductSchema.statics.findInPriceRange = function(minPrice, maxPrice, options = {}) {
  return this.find({
    price: { $gte: minPrice, $lte: maxPrice },
    status: 'active',
    visibility: 'public'
  }, null, options);
};

// Pre-save middleware
ProductSchema.pre('save', function(next) {
  // Generate slug from title if not provided
  if (!this.slug && this.title) {
    this.slug = this.title
      .toLowerCase()
      .replace(/[^a-z0-9\s]+/g, '')
      .replace(/\s+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
  
  // Ensure tags are unique and lowercase
  if (this.tags) {
    this.tags = [...new Set(this.tags.map(tag => tag.toLowerCase()))];
  }
  
  // Set published date when status changes to active
  if (this.isModified('status') && this.status === 'active' && !this.publishedAt) {
    this.publishedAt = new Date();
  }
  
  next();
});

export default mongoose.models.Product || mongoose.model('Product', ProductSchema);