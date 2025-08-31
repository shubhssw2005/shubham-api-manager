import mongoose from 'mongoose';

const MediaFolderSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true,
    maxlength: 100
  },
  slug: {
    type: String,
    required: true,
    trim: true,
    lowercase: true,
    maxlength: 100
  },
  description: {
    type: String,
    trim: true,
    maxlength: 500,
    default: ''
  },
  parent: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'MediaFolder',
    default: null
  },
  path: {
    type: String,
    required: true,
    trim: true
  },
  level: {
    type: Number,
    required: true,
    min: 0,
    default: 0
  },
  isPublic: {
    type: Boolean,
    default: false
  },
  permissions: {
    read: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }],
    write: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }],
    admin: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }]
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  color: {
    type: String,
    trim: true,
    match: /^#[0-9A-F]{6}$/i,
    default: '#3498db'
  },
  icon: {
    type: String,
    trim: true,
    default: 'folder'
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for efficient querying
MediaFolderSchema.index({ slug: 1 }, { unique: true });
MediaFolderSchema.index({ parent: 1 });
MediaFolderSchema.index({ path: 1 });
MediaFolderSchema.index({ level: 1 });
MediaFolderSchema.index({ createdBy: 1 });
MediaFolderSchema.index({ createdAt: -1 });

// Compound indexes
MediaFolderSchema.index({ parent: 1, name: 1 });
MediaFolderSchema.index({ parent: 1, createdAt: -1 });

// Text index for search
MediaFolderSchema.index({
  name: 'text',
  description: 'text'
});

// Virtual for children folders
MediaFolderSchema.virtual('children', {
  ref: 'MediaFolder',
  localField: '_id',
  foreignField: 'parent'
});

// Virtual for media files in this folder
MediaFolderSchema.virtual('mediaFiles', {
  ref: 'Media',
  localField: '_id',
  foreignField: 'folder'
});

// Virtual for full path display
MediaFolderSchema.virtual('fullPath').get(function() {
  return this.path;
});

// Virtual for breadcrumb navigation
MediaFolderSchema.virtual('breadcrumbs').get(function() {
  if (!this.path) return [];
  
  const pathParts = this.path.split('/').filter(part => part);
  return pathParts.map((part, index) => ({
    name: part,
    path: '/' + pathParts.slice(0, index + 1).join('/')
  }));
});

// Instance methods
MediaFolderSchema.methods.getAncestors = async function() {
  const ancestors = [];
  let current = this;
  
  while (current.parent) {
    current = await this.constructor.findById(current.parent);
    if (current) {
      ancestors.unshift(current);
    } else {
      break;
    }
  }
  
  return ancestors;
};

MediaFolderSchema.methods.getDescendants = async function() {
  const descendants = [];
  
  const findChildren = async (parentId) => {
    const children = await this.constructor.find({ parent: parentId });
    
    for (const child of children) {
      descendants.push(child);
      await findChildren(child._id);
    }
  };
  
  await findChildren(this._id);
  return descendants;
};

MediaFolderSchema.methods.canUserAccess = function(userId, permission = 'read') {
  // If folder is public and permission is read, allow access
  if (this.isPublic && permission === 'read') {
    return true;
  }
  
  // Check if user is the creator
  if (this.createdBy.toString() === userId.toString()) {
    return true;
  }
  
  // Check specific permissions
  const userIdStr = userId.toString();
  
  switch (permission) {
    case 'admin':
      return this.permissions.admin.some(id => id.toString() === userIdStr);
    case 'write':
      return this.permissions.write.some(id => id.toString() === userIdStr) ||
             this.permissions.admin.some(id => id.toString() === userIdStr);
    case 'read':
    default:
      return this.permissions.read.some(id => id.toString() === userIdStr) ||
             this.permissions.write.some(id => id.toString() === userIdStr) ||
             this.permissions.admin.some(id => id.toString() === userIdStr);
  }
};

MediaFolderSchema.methods.addPermission = function(userId, permission = 'read') {
  const userIdStr = userId.toString();
  
  if (!this.permissions[permission].some(id => id.toString() === userIdStr)) {
    this.permissions[permission].push(userId);
  }
  
  return this.save();
};

MediaFolderSchema.methods.removePermission = function(userId, permission = 'read') {
  const userIdStr = userId.toString();
  this.permissions[permission] = this.permissions[permission].filter(
    id => id.toString() !== userIdStr
  );
  
  return this.save();
};

MediaFolderSchema.methods.getMediaCount = async function() {
  const Media = mongoose.model('Media');
  return await Media.countDocuments({ folder: this._id });
};

MediaFolderSchema.methods.getSubfolderCount = async function() {
  return await this.constructor.countDocuments({ parent: this._id });
};

MediaFolderSchema.methods.getTotalSize = async function() {
  const Media = mongoose.model('Media');
  const result = await Media.aggregate([
    { $match: { folder: this._id } },
    { $group: { _id: null, totalSize: { $sum: '$size' } } }
  ]);
  
  return result.length > 0 ? result[0].totalSize : 0;
};

// Static methods
MediaFolderSchema.statics.findByPath = function(path) {
  return this.findOne({ path });
};

MediaFolderSchema.statics.getRootFolders = function(userId = null, options = {}) {
  const query = { parent: null };
  
  if (userId) {
    query.$or = [
      { isPublic: true },
      { createdBy: userId },
      { 'permissions.read': userId },
      { 'permissions.write': userId },
      { 'permissions.admin': userId }
    ];
  }
  
  return this.find(query, null, options);
};

MediaFolderSchema.statics.searchFolders = function(searchTerm, userId = null, options = {}) {
  const query = { $text: { $search: searchTerm } };
  
  if (userId) {
    query.$and = [{
      $or: [
        { isPublic: true },
        { createdBy: userId },
        { 'permissions.read': userId },
        { 'permissions.write': userId },
        { 'permissions.admin': userId }
      ]
    }];
  }
  
  return this.find(
    query,
    { score: { $meta: 'textScore' } },
    { ...options, sort: { score: { $meta: 'textScore' } } }
  );
};

MediaFolderSchema.statics.buildFolderTree = async function(parentId = null, userId = null) {
  const query = { parent: parentId };
  
  if (userId) {
    query.$or = [
      { isPublic: true },
      { createdBy: userId },
      { 'permissions.read': userId },
      { 'permissions.write': userId },
      { 'permissions.admin': userId }
    ];
  }
  
  const folders = await this.find(query).sort({ name: 1 });
  
  const tree = [];
  for (const folder of folders) {
    const folderObj = folder.toObject();
    folderObj.children = await this.buildFolderTree(folder._id, userId);
    tree.push(folderObj);
  }
  
  return tree;
};

MediaFolderSchema.statics.validatePath = function(path, parentId = null) {
  // Check if path is valid format
  if (!path || typeof path !== 'string') {
    return { valid: false, error: 'Path is required and must be a string' };
  }
  
  // Check path format
  if (!path.startsWith('/')) {
    return { valid: false, error: 'Path must start with /' };
  }
  
  // Check for invalid characters
  if (!/^[a-zA-Z0-9\-_\/]+$/.test(path)) {
    return { valid: false, error: 'Path contains invalid characters' };
  }
  
  return { valid: true };
};

// Pre-save middleware
MediaFolderSchema.pre('save', async function(next) {
  try {
    // Generate slug from name if not provided
    if (!this.slug && this.name) {
      this.slug = this.name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    }
    
    // Build path based on parent
    if (this.parent) {
      const parent = await this.constructor.findById(this.parent);
      if (parent) {
        this.path = `${parent.path}/${this.slug}`;
        this.level = parent.level + 1;
      } else {
        return next(new Error('Parent folder not found'));
      }
    } else {
      this.path = `/${this.slug}`;
      this.level = 0;
    }
    
    // Validate maximum nesting level
    if (this.level > 10) {
      return next(new Error('Maximum folder nesting level (10) exceeded'));
    }
    
    next();
  } catch (error) {
    next(error);
  }
});

// Also add pre-validate middleware to ensure slug and path are set before validation
MediaFolderSchema.pre('validate', function(next) {
  // Generate slug from name if not provided
  if (!this.slug && this.name) {
    this.slug = this.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
  
  // Set path for root folders (parent will be handled in pre-save)
  if (!this.parent && !this.path && this.slug) {
    this.path = `/${this.slug}`;
    this.level = 0;
  }
  
  next();
});

// Pre-remove middleware
MediaFolderSchema.pre('remove', async function(next) {
  try {
    // Check if folder has children
    const childrenCount = await this.constructor.countDocuments({ parent: this._id });
    if (childrenCount > 0) {
      return next(new Error('Cannot delete folder that contains subfolders'));
    }
    
    // Check if folder has media files
    const Media = mongoose.model('Media');
    const mediaCount = await Media.countDocuments({ folder: this._id });
    if (mediaCount > 0) {
      return next(new Error('Cannot delete folder that contains media files'));
    }
    
    next();
  } catch (error) {
    next(error);
  }
});

export default mongoose.models.MediaFolder || mongoose.model('MediaFolder', MediaFolderSchema);