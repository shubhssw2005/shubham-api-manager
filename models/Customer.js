import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const AddressSchema = new mongoose.Schema({
  type: {
    type: String,
    enum: ['shipping', 'billing'],
    required: true
  },
  firstName: { type: String, required: true, trim: true },
  lastName: { type: String, required: true, trim: true },
  company: { type: String, trim: true },
  address1: { type: String, required: true, trim: true },
  address2: { type: String, trim: true },
  city: { type: String, required: true, trim: true },
  state: { type: String, required: true, trim: true },
  postalCode: { type: String, required: true, trim: true },
  country: { type: String, required: true, trim: true },
  phone: { type: String, trim: true },
  isDefault: { type: Boolean, default: false }
}, { timestamps: true });

const CustomerSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  firstName: {
    type: String,
    required: true,
    trim: true
  },
  lastName: {
    type: String,
    required: true,
    trim: true
  },
  phone: {
    type: String,
    trim: true
  },
  dateOfBirth: Date,
  gender: {
    type: String,
    enum: ['male', 'female', 'other', 'prefer_not_to_say']
  },
  
  // Account status
  isActive: {
    type: Boolean,
    default: true
  },
  emailVerified: {
    type: Boolean,
    default: false
  },
  emailVerificationToken: String,
  
  // Addresses
  addresses: [AddressSchema],
  
  // Preferences
  preferences: {
    newsletter: { type: Boolean, default: true },
    smsMarketing: { type: Boolean, default: false },
    language: { type: String, default: 'en' },
    currency: { type: String, default: 'USD' }
  },
  
  // Customer metrics
  totalOrders: { type: Number, default: 0 },
  totalSpent: { type: Number, default: 0 },
  averageOrderValue: { type: Number, default: 0 },
  lastOrderDate: Date,
  
  // Tags for segmentation
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  
  // Notes
  notes: String,
  
  // Password reset
  resetPasswordToken: String,
  resetPasswordExpires: Date,
  
  // Login tracking
  lastLoginAt: Date,
  loginCount: { type: Number, default: 0 }
}, {
  timestamps: true,
  toJSON: { 
    virtuals: true,
    transform: function(doc, ret) {
      delete ret.password;
      delete ret.resetPasswordToken;
      delete ret.emailVerificationToken;
      return ret;
    }
  },
  toObject: { virtuals: true }
});

// Indexes
CustomerSchema.index({ email: 1 });
CustomerSchema.index({ firstName: 1, lastName: 1 });
CustomerSchema.index({ isActive: 1 });
CustomerSchema.index({ totalSpent: -1 });
CustomerSchema.index({ lastOrderDate: -1 });
CustomerSchema.index({ createdAt: -1 });

// Virtuals
CustomerSchema.virtual('fullName').get(function() {
  return `${this.firstName} ${this.lastName}`;
});

CustomerSchema.virtual('defaultShippingAddress').get(function() {
  return this.addresses.find(addr => addr.type === 'shipping' && addr.isDefault) ||
         this.addresses.find(addr => addr.type === 'shipping');
});

CustomerSchema.virtual('defaultBillingAddress').get(function() {
  return this.addresses.find(addr => addr.type === 'billing' && addr.isDefault) ||
         this.addresses.find(addr => addr.type === 'billing');
});

// Hash password before saving
CustomerSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(12);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

// Compare password method
CustomerSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// Update customer metrics
CustomerSchema.methods.updateMetrics = async function() {
  const Order = mongoose.model('Order');
  const orders = await Order.find({ customer: this._id, status: { $ne: 'cancelled' } });
  
  this.totalOrders = orders.length;
  this.totalSpent = orders.reduce((total, order) => total + order.total, 0);
  this.averageOrderValue = this.totalOrders > 0 ? this.totalSpent / this.totalOrders : 0;
  this.lastOrderDate = orders.length > 0 ? 
    orders.sort((a, b) => b.createdAt - a.createdAt)[0].createdAt : null;
  
  return this.save();
};

// Add address method
CustomerSchema.methods.addAddress = function(addressData) {
  // If this is the first address of its type, make it default
  const existingAddressesOfType = this.addresses.filter(addr => addr.type === addressData.type);
  if (existingAddressesOfType.length === 0) {
    addressData.isDefault = true;
  }
  
  this.addresses.push(addressData);
  return this.save();
};

// Set default address method
CustomerSchema.methods.setDefaultAddress = function(addressId) {
  const address = this.addresses.id(addressId);
  if (!address) return false;
  
  // Remove default from other addresses of the same type
  this.addresses.forEach(addr => {
    if (addr.type === address.type && addr._id.toString() !== addressId) {
      addr.isDefault = false;
    }
  });
  
  address.isDefault = true;
  return this.save();
};

export default mongoose.models.Customer || mongoose.model('Customer', CustomerSchema);