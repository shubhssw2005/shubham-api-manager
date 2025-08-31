import mongoose from 'mongoose';

const OrderItemSchema = new mongoose.Schema({
  product: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Product',
    required: true
  },
  variant: {
    type: mongoose.Schema.Types.ObjectId,
    required: false
  },
  quantity: {
    type: Number,
    required: true,
    min: 1
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  total: {
    type: Number,
    required: true,
    min: 0
  }
}, { _id: false });

const ShippingAddressSchema = new mongoose.Schema({
  firstName: { type: String, required: true, trim: true },
  lastName: { type: String, required: true, trim: true },
  company: { type: String, trim: true },
  address1: { type: String, required: true, trim: true },
  address2: { type: String, trim: true },
  city: { type: String, required: true, trim: true },
  state: { type: String, required: true, trim: true },
  postalCode: { type: String, required: true, trim: true },
  country: { type: String, required: true, trim: true },
  phone: { type: String, trim: true }
}, { _id: false });

const OrderSchema = new mongoose.Schema({
  orderNumber: {
    type: String,
    required: true,
    unique: true
  },
  customer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  items: [OrderItemSchema],
  
  // Pricing
  subtotal: {
    type: Number,
    required: true,
    min: 0
  },
  tax: {
    type: Number,
    default: 0,
    min: 0
  },
  shipping: {
    type: Number,
    default: 0,
    min: 0
  },
  discount: {
    type: Number,
    default: 0,
    min: 0
  },
  total: {
    type: Number,
    required: true,
    min: 0
  },
  
  // Addresses
  shippingAddress: ShippingAddressSchema,
  billingAddress: ShippingAddressSchema,
  
  // Status
  status: {
    type: String,
    enum: ['pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled', 'refunded'],
    default: 'pending'
  },
  paymentStatus: {
    type: String,
    enum: ['pending', 'paid', 'failed', 'refunded', 'partially_refunded'],
    default: 'pending'
  },
  fulfillmentStatus: {
    type: String,
    enum: ['unfulfilled', 'partial', 'fulfilled'],
    default: 'unfulfilled'
  },
  
  // Payment
  paymentMethod: {
    type: String,
    enum: ['credit_card', 'debit_card', 'paypal', 'stripe', 'cash_on_delivery'],
    required: true
  },
  paymentId: String,
  
  // Shipping
  shippingMethod: String,
  trackingNumber: String,
  shippedAt: Date,
  deliveredAt: Date,
  
  // Notes
  notes: String,
  customerNotes: String,
  
  // Timestamps
  confirmedAt: Date,
  cancelledAt: Date,
  refundedAt: Date
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes
OrderSchema.index({ orderNumber: 1 });
OrderSchema.index({ customer: 1 });
OrderSchema.index({ status: 1 });
OrderSchema.index({ paymentStatus: 1 });
OrderSchema.index({ createdAt: -1 });
OrderSchema.index({ customer: 1, createdAt: -1 });

// Virtuals
OrderSchema.virtual('itemCount').get(function() {
  return this.items.reduce((total, item) => total + item.quantity, 0);
});

// Pre-save middleware
OrderSchema.pre('save', function(next) {
  // Generate order number if not provided
  if (!this.orderNumber) {
    const timestamp = Date.now().toString();
    const random = Math.floor(Math.random() * 1000).toString().padStart(3, '0');
    this.orderNumber = `ORD-${timestamp}-${random}`;
  }
  
  // Calculate totals
  this.subtotal = this.items.reduce((total, item) => total + item.total, 0);
  this.total = this.subtotal + this.tax + this.shipping - this.discount;
  
  // Set status timestamps
  if (this.isModified('status')) {
    if (this.status === 'confirmed' && !this.confirmedAt) {
      this.confirmedAt = new Date();
    } else if (this.status === 'cancelled' && !this.cancelledAt) {
      this.cancelledAt = new Date();
    }
  }
  
  if (this.isModified('paymentStatus') && this.paymentStatus === 'refunded' && !this.refundedAt) {
    this.refundedAt = new Date();
  }
  
  next();
});

export default mongoose.models.Order || mongoose.model('Order', OrderSchema);