import mongoose from 'mongoose';

const SettingsSchema = new mongoose.Schema({
  key: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    datatype: "textinput"
  },
  value: {
    type: mongoose.Schema.Types.Mixed,
    required: true
  },
  type: {
    type: String,
    enum: ['string', 'number', 'boolean', 'object', 'array'],
    required: true,
    datatype: "selectinput"
  },
  category: {
    type: String,
    enum: ['system', 'media', 'api', 'security', 'ui'],
    required: true,
    datatype: "selectinput"
  },
  description: {
    type: String,
    trim: true,
    datatype: "textarea"
  },
  isPublic: {
    type: Boolean,
    default: false,
    datatype: "toggleinput"
  },
  isEditable: {
    type: Boolean,
    default: true,
    datatype: "toggleinput"
  },
  validation: {
    min: Number,
    max: Number,
    pattern: String,
    required: Boolean
  },
  lastModifiedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }
}, { 
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Index for efficient querying
SettingsSchema.index({ category: 1, key: 1 });
SettingsSchema.index({ isPublic: 1 });

// Virtual for formatted value based on type
SettingsSchema.virtual('formattedValue').get(function() {
  switch (this.type) {
    case 'boolean':
      return Boolean(this.value);
    case 'number':
      return Number(this.value);
    case 'array':
      return Array.isArray(this.value) ? this.value : [];
    case 'object':
      return typeof this.value === 'object' ? this.value : {};
    default:
      return String(this.value);
  }
});

// Static method to get settings by category
SettingsSchema.statics.getByCategory = function(category, includePrivate = false) {
  const query = { category };
  if (!includePrivate) {
    query.isPublic = true;
  }
  return this.find(query).sort({ key: 1 });
};

// Static method to get public settings
SettingsSchema.statics.getPublicSettings = function() {
  return this.find({ isPublic: true }).sort({ category: 1, key: 1 });
};

// Static method to set a setting value
SettingsSchema.statics.setSetting = async function(key, value, userId = null) {
  const setting = await this.findOne({ key });
  if (!setting) {
    throw new Error(`Setting with key '${key}' not found`);
  }
  
  if (!setting.isEditable) {
    throw new Error(`Setting '${key}' is not editable`);
  }
  
  // Validate value based on type
  if (!this.validateValue(value, setting.type, setting.validation)) {
    throw new Error(`Invalid value for setting '${key}'`);
  }
  
  setting.value = value;
  setting.lastModifiedBy = userId;
  return setting.save();
};

// Static method to validate setting value
SettingsSchema.statics.validateValue = function(value, type, validation = {}) {
  switch (type) {
    case 'string':
      if (typeof value !== 'string') return false;
      if (validation.pattern && !new RegExp(validation.pattern).test(value)) return false;
      if (validation.min && value.length < validation.min) return false;
      if (validation.max && value.length > validation.max) return false;
      break;
    case 'number':
      if (typeof value !== 'number' || isNaN(value)) return false;
      if (validation.min !== undefined && value < validation.min) return false;
      if (validation.max !== undefined && value > validation.max) return false;
      break;
    case 'boolean':
      if (typeof value !== 'boolean') return false;
      break;
    case 'array':
      if (!Array.isArray(value)) return false;
      break;
    case 'object':
      if (typeof value !== 'object' || Array.isArray(value)) return false;
      break;
    default:
      return false;
  }
  return true;
};

// Pre-save middleware to validate value
SettingsSchema.pre('save', function(next) {
  if (this.isModified('value')) {
    if (!this.constructor.validateValue(this.value, this.type, this.validation)) {
      return next(new Error(`Invalid value for setting type '${this.type}'`));
    }
  }
  next();
});

export default mongoose.models.Settings || mongoose.model('Settings', SettingsSchema);