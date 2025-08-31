import mongoose from 'mongoose';

const ThumbnailSchema = new mongoose.Schema({
  size: {
    type: String,
    required: true,
    enum: ['small', 'medium', 'large']
  },
  url: {
    type: String,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  width: {
    type: Number,
    required: true
  },
  height: {
    type: Number,
    required: true
  },
  fileSize: {
    type: Number,
    required: true
  }
}, { _id: false });

const UsageSchema = new mongoose.Schema({
  modelName: {
    type: String,
    required: true
  },
  documentId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true
  },
  fieldName: {
    type: String,
    required: true
  },
  usedAt: {
    type: Date,
    default: Date.now
  }
}, { _id: false });

const MediaSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true,
    trim: true
  },
  originalName: {
    type: String,
    required: true,
    trim: true
  },
  mimeType: {
    type: String,
    required: true
  },
  size: {
    type: Number,
    required: true,
    min: 0
  },
  path: {
    type: String,
    required: true
  },
  url: {
    type: String,
    required: true
  },
  storageProvider: {
    type: String,
    required: true,
    enum: ['local', 's3', 'gcs'],
    default: 'local'
  },
  thumbnails: [ThumbnailSchema],
  metadata: {
    // Image metadata
    width: Number,
    height: Number,
    format: String,
    channels: Number,
    density: Number,
    hasAlpha: Boolean,
    
    // Video metadata
    duration: Number,
    bitrate: Number,
    codec: String,
    fps: Number,
    
    // Document metadata
    pages: Number,
    lineCount: Number,
    wordCount: Number,
    charCount: Number,
    
    // Generic metadata
    exif: mongoose.Schema.Types.Mixed,
    custom: mongoose.Schema.Types.Mixed
  },
  folder: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'MediaFolder',
    default: null
  },
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  alt: {
    type: String,
    trim: true,
    default: ''
  },
  caption: {
    type: String,
    trim: true,
    default: ''
  },
  description: {
    type: String,
    trim: true,
    default: ''
  },
  usage: [UsageSchema],
  isPublic: {
    type: Boolean,
    default: false
  },
  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  processedAt: {
    type: Date
  },
  processingStatus: {
    type: String,
    enum: ['pending', 'processing', 'completed', 'failed'],
    default: 'pending'
  },
  processingError: {
    type: String
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for efficient querying
MediaSchema.index({ filename: 1 });
MediaSchema.index({ mimeType: 1 });
MediaSchema.index({ folder: 1 });
MediaSchema.index({ tags: 1 });
MediaSchema.index({ uploadedBy: 1 });
MediaSchema.index({ createdAt: -1 });
MediaSchema.index({ 'usage.modelName': 1, 'usage.documentId': 1 });

// Compound indexes
MediaSchema.index({ folder: 1, createdAt: -1 });
MediaSchema.index({ mimeType: 1, createdAt: -1 });
MediaSchema.index({ uploadedBy: 1, createdAt: -1 });

// Text index for search
MediaSchema.index({
  filename: 'text',
  originalName: 'text',
  alt: 'text',
  caption: 'text',
  description: 'text',
  tags: 'text'
});

// Virtual for file type category
MediaSchema.virtual('fileType').get(function() {
  if (this.mimeType.startsWith('image/')) return 'image';
  if (this.mimeType.startsWith('video/')) return 'video';
  if (this.mimeType.startsWith('audio/')) return 'audio';
  if (this.mimeType === 'application/pdf' || this.mimeType.startsWith('text/')) return 'document';
  return 'other';
});

// Virtual for human-readable file size
MediaSchema.virtual('formattedSize').get(function() {
  const bytes = this.size;
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
});

// Virtual for usage count
MediaSchema.virtual('usageCount').get(function() {
  return this.usage ? this.usage.length : 0;
});

// Instance methods
MediaSchema.methods.addUsage = function(modelName, documentId, fieldName) {
  // Check if usage already exists
  const existingUsage = this.usage.find(u => 
    u.modelName === modelName && 
    u.documentId.toString() === documentId.toString() && 
    u.fieldName === fieldName
  );
  
  if (!existingUsage) {
    this.usage.push({ modelName, documentId, fieldName });
  }
  
  return this.save();
};

MediaSchema.methods.removeUsage = function(modelName, documentId, fieldName) {
  this.usage = this.usage.filter(u => 
    !(u.modelName === modelName && 
      u.documentId.toString() === documentId.toString() && 
      u.fieldName === fieldName)
  );
  
  return this.save();
};

MediaSchema.methods.isUsed = function() {
  return this.usage && this.usage.length > 0;
};

MediaSchema.methods.canDelete = function() {
  return !this.isUsed();
};

MediaSchema.methods.getThumbnail = function(size = 'medium') {
  return this.thumbnails.find(thumb => thumb.size === size);
};

// Static methods
MediaSchema.statics.findByFolder = function(folderId, options = {}) {
  const query = folderId ? { folder: folderId } : { folder: null };
  return this.find(query, null, options);
};

MediaSchema.statics.findByType = function(fileType, options = {}) {
  let mimeTypePattern;
  
  switch (fileType) {
    case 'image':
      mimeTypePattern = /^image\//;
      break;
    case 'video':
      mimeTypePattern = /^video\//;
      break;
    case 'audio':
      mimeTypePattern = /^audio\//;
      break;
    case 'document':
      mimeTypePattern = /^(application\/pdf|text\/)/;
      break;
    default:
      return this.find({}, null, options);
  }
  
  return this.find({ mimeType: mimeTypePattern }, null, options);
};

MediaSchema.statics.searchFiles = function(searchTerm, options = {}) {
  return this.find(
    { $text: { $search: searchTerm } },
    { score: { $meta: 'textScore' } },
    { ...options, sort: { score: { $meta: 'textScore' } } }
  );
};

MediaSchema.statics.findUnused = function(options = {}) {
  return this.find({ 'usage.0': { $exists: false } }, null, options);
};

MediaSchema.statics.getStorageStats = function() {
  return this.aggregate([
    {
      $group: {
        _id: '$storageProvider',
        totalFiles: { $sum: 1 },
        totalSize: { $sum: '$size' },
        avgSize: { $avg: '$size' }
      }
    }
  ]);
};

MediaSchema.statics.getTypeStats = function() {
  return this.aggregate([
    {
      $addFields: {
        fileType: {
          $switch: {
            branches: [
              { case: { $regexMatch: { input: '$mimeType', regex: /^image\// } }, then: 'image' },
              { case: { $regexMatch: { input: '$mimeType', regex: /^video\// } }, then: 'video' },
              { case: { $regexMatch: { input: '$mimeType', regex: /^audio\// } }, then: 'audio' },
              { case: { $regexMatch: { input: '$mimeType', regex: /^(application\/pdf|text\/)/ } }, then: 'document' }
            ],
            default: 'other'
          }
        }
      }
    },
    {
      $group: {
        _id: '$fileType',
        count: { $sum: 1 },
        totalSize: { $sum: '$size' }
      }
    }
  ]);
};

// Pre-save middleware
MediaSchema.pre('save', function(next) {
  // Ensure tags are unique and lowercase
  if (this.tags) {
    this.tags = [...new Set(this.tags.map(tag => tag.toLowerCase()))];
  }
  
  next();
});

// Pre-remove middleware to clean up thumbnails and storage
MediaSchema.pre('remove', async function(next) {
  try {
    // Here you would typically clean up the actual files from storage
    // This would integrate with the StorageProvider
    console.log(`Cleaning up media file: ${this.filename}`);
    next();
  } catch (error) {
    next(error);
  }
});

export default mongoose.models.Media || mongoose.model('Media', MediaSchema);