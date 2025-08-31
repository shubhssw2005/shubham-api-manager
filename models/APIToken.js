import mongoose from 'mongoose';
import crypto from 'crypto';

const APITokenSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true,
    datatype: "textinput"
  },
  token: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  hashedToken: {
    type: String,
    index: true
  },
  permissions: [{
    model: {
      type: String,
      required: true
    },
    actions: [{
      type: String,
      enum: ['create', 'read', 'update', 'delete'],
      required: true
    }]
  }],
  rateLimit: {
    requests: {
      type: Number,
      default: 1000,
      min: 1,
      max: 10000,
      datatype: "numberinput"
    },
    window: {
      type: Number,
      default: 3600, // seconds
      min: 60,
      max: 86400,
      datatype: "numberinput"
    }
  },
  usage: {
    totalRequests: {
      type: Number,
      default: 0
    },
    lastUsed: Date,
    lastIP: String,
    lastUserAgent: String
  },
  expiresAt: {
    type: Date,
    datatype: "dateinput"
  },
  isActive: {
    type: Boolean,
    default: true,
    datatype: "toggleinput"
  },
  description: {
    type: String,
    trim: true,
    datatype: "textarea"
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  }
}, { 
  timestamps: true,
  toJSON: { 
    virtuals: true,
    transform: function(doc, ret) {
      // Never return the actual token or hashed token in JSON
      delete ret.token;
      delete ret.hashedToken;
      return ret;
    }
  },
  toObject: { virtuals: true }
});

// Indexes for efficient querying
APITokenSchema.index({ createdBy: 1 });
APITokenSchema.index({ isActive: 1 });
APITokenSchema.index({ expiresAt: 1 });

// Virtual for checking if token is expired
APITokenSchema.virtual('isExpired').get(function() {
  return this.expiresAt && this.expiresAt < new Date();
});

// Virtual for checking if token is valid (active and not expired)
APITokenSchema.virtual('isValid').get(function() {
  return this.isActive && !this.isExpired;
});

// Virtual for masked token display
APITokenSchema.virtual('maskedToken').get(function() {
  if (!this.token) return null;
  return this.token.substring(0, 8) + '...' + this.token.substring(this.token.length - 4);
});

// Static method to generate a new token
APITokenSchema.statics.generateToken = function() {
  return crypto.randomBytes(32).toString('hex');
};

// Static method to hash a token
APITokenSchema.statics.hashToken = function(token) {
  return crypto.createHash('sha256').update(token).digest('hex');
};

// Static method to find token by hashed value
APITokenSchema.statics.findByToken = function(token) {
  const hashedToken = this.hashToken(token);
  return this.findOne({ 
    hashedToken, 
    isActive: true,
    $or: [
      { expiresAt: { $exists: false } },
      { expiresAt: null },
      { expiresAt: { $gt: new Date() } }
    ]
  }).populate('createdBy', 'name email');
};

// Method to check if token has permission for a specific action on a model
APITokenSchema.methods.hasPermission = function(modelName, action) {
  if (!this.isValid) return false;
  
  const permission = this.permissions.find(p => p.model === modelName || p.model === '*');
  if (!permission) return false;
  
  return permission.actions.includes(action) || permission.actions.includes('*');
};

// Method to record token usage
APITokenSchema.methods.recordUsage = async function(ip, userAgent) {
  this.usage.totalRequests += 1;
  this.usage.lastUsed = new Date();
  this.usage.lastIP = ip;
  this.usage.lastUserAgent = userAgent;
  return this.save();
};

// Method to revoke token
APITokenSchema.methods.revoke = function() {
  this.isActive = false;
  return this.save();
};

// Pre-save middleware to hash token
APITokenSchema.pre('save', function(next) {
  if (this.token && (!this.hashedToken || this.isModified('token'))) {
    this.hashedToken = this.constructor.hashToken(this.token);
  }
  next();
});

// Pre-save middleware to set expiration if not provided
APITokenSchema.pre('save', function(next) {
  if (this.isNew && !this.expiresAt) {
    // Default expiration: 1 year from now
    this.expiresAt = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000);
  }
  next();
});

export default mongoose.models.APIToken || mongoose.model('APIToken', APITokenSchema);