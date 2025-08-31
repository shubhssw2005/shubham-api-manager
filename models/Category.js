import mongoose from 'mongoose';

const CategorySchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'Category name is required'],
    trim: true,
    unique: true
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
    trim: true
  },
  parent: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Category',
    default: null
  },
  image: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Media'
  },
  isActive: {
    type: Boolean,
    default: true
  },
  sortOrder: {
    type: Number,
    default: 0
  },
  seo: {
    metaTitle: String,
    metaDescription: String,
    keywords: [String]
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

CategorySchema.index({ slug: 1 });
CategorySchema.index({ parent: 1 });
CategorySchema.index({ isActive: 1, sortOrder: 1 });

CategorySchema.virtual('children', {
  ref: 'Category',
  localField: '_id',
  foreignField: 'parent'
});

CategorySchema.virtual('products', {
  ref: 'Product',
  localField: 'name',
  foreignField: 'category'
});

CategorySchema.pre('save', function(next) {
  if (!this.slug && this.name) {
    this.slug = this.name
      .toLowerCase()
      .replace(/[^a-z0-9\s]+/g, '')
      .replace(/\s+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
  next();
});

export default mongoose.models.Category || mongoose.model('Category', CategorySchema);