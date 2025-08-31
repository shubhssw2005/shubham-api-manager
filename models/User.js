import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const UserSchema = new mongoose.Schema({
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
  name: {
    type: String,
    required: true,
    trim: true
  },
  role: {
    type: String,
    enum: ['user', 'admin'],
    default: 'user'
  },
  status: {
    type: String,
    enum: ['pending', 'active', 'rejected'],
    default: 'active' // Change default to active since this is a test environment
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  approvedAt: {
    type: Date
  },
  approvedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  },
  // Groot.com integration fields
  isGrootUser: {
    type: Boolean,
    default: false,
    index: true
  },
  grootUserId: {
    type: String,
    sparse: true,
    index: true
  },
  lastLoginAt: {
    type: Date
  },
  // Data export preferences
  dataExportPreferences: {
    autoIncludeMedia: {
      type: Boolean,
      default: true
    },
    autoIncludeDeleted: {
      type: Boolean,
      default: false
    },
    notifyOnExportReady: {
      type: Boolean,
      default: true
    }
  }
});

// Hash password before saving
UserSchema.pre('save', async function(next) {
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
UserSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// Check if user is approved
UserSchema.methods.isApproved = function() {
  return this.status === 'approved';
};

export default mongoose.models.User || mongoose.model('User', UserSchema);