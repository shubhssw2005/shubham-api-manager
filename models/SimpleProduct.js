import mongoose from 'mongoose';

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
ProductSchema.index({ category: 1 });
ProductSchema.index({ brand: 1 });
ProductSchema.index({ status: 1, visibility: 1 });
ProductSchema.index({ featured: 1 });
ProductSchema.index({ price: 1 });
ProductSchema.index({ 'rating.average': -1 });
ProductSchema.index({ createdAt: -1 });

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