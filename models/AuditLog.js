import mongoose from 'mongoose';

const AuditLogSchema = new mongoose.Schema({
  action: {
    type: String,
    required: true,
    enum: [
      'create', 'update', 'delete', 'login', 'logout', 
      'token_create', 'token_revoke', 'settings_change',
      'permission_change', 'user_approve', 'user_reject'
    ],
    datatype: "selectinput"
  },
  resource: {
    type: String,
    required: true, // e.g., 'settings', 'user', 'api_token', 'role'
    datatype: "textinput"
  },
  resourceId: {
    type: mongoose.Schema.Types.ObjectId,
    datatype: "textinput"
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  details: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  changes: {
    before: mongoose.Schema.Types.Mixed,
    after: mongoose.Schema.Types.Mixed
  },
  ipAddress: {
    type: String,
    datatype: "textinput"
  },
  userAgent: {
    type: String,
    datatype: "textinput"
  },
  success: {
    type: Boolean,
    default: true,
    datatype: "toggleinput"
  },
  errorMessage: {
    type: String,
    datatype: "textarea"
  }
}, { 
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for efficient querying
AuditLogSchema.index({ userId: 1, createdAt: -1 });
AuditLogSchema.index({ resource: 1, createdAt: -1 });
AuditLogSchema.index({ action: 1, createdAt: -1 });
AuditLogSchema.index({ createdAt: -1 });
AuditLogSchema.index({ success: 1, createdAt: -1 });

// Virtual for formatted timestamp
AuditLogSchema.virtual('formattedTimestamp').get(function() {
  return this.createdAt.toISOString();
});

// Static method to log an action
AuditLogSchema.statics.logAction = async function(data) {
  const {
    action,
    resource,
    resourceId,
    userId,
    details = {},
    changes = {},
    ipAddress,
    userAgent,
    success = true,
    errorMessage
  } = data;

  return this.create({
    action,
    resource,
    resourceId,
    userId,
    details,
    changes,
    ipAddress,
    userAgent,
    success,
    errorMessage
  });
};

// Static method to get recent activity
AuditLogSchema.statics.getRecentActivity = function(limit = 50, filters = {}) {
  const query = { ...filters };
  return this.find(query)
    .populate('userId', 'name email')
    .sort({ createdAt: -1 })
    .limit(limit);
};

// Static method to get activity by user
AuditLogSchema.statics.getByUser = function(userId, limit = 50) {
  return this.find({ userId })
    .sort({ createdAt: -1 })
    .limit(limit);
};

// Static method to get activity by resource
AuditLogSchema.statics.getByResource = function(resource, resourceId = null, limit = 50) {
  const query = { resource };
  if (resourceId) {
    query.resourceId = resourceId;
  }
  return this.find(query)
    .populate('userId', 'name email')
    .sort({ createdAt: -1 })
    .limit(limit);
};

// Static method to get failed actions
AuditLogSchema.statics.getFailedActions = function(limit = 50) {
  return this.find({ success: false })
    .populate('userId', 'name email')
    .sort({ createdAt: -1 })
    .limit(limit);
};

// Static method to get statistics
AuditLogSchema.statics.getStatistics = async function(days = 30) {
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  const pipeline = [
    { $match: { createdAt: { $gte: startDate } } },
    {
      $group: {
        _id: {
          action: '$action',
          date: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } }
        },
        count: { $sum: 1 }
      }
    },
    { $sort: { '_id.date': -1, '_id.action': 1 } }
  ];

  return this.aggregate(pipeline);
};

// Static method to cleanup old logs
AuditLogSchema.statics.cleanup = function(daysToKeep = 90) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
  
  return this.deleteMany({ createdAt: { $lt: cutoffDate } });
};

export default mongoose.models.AuditLog || mongoose.model('AuditLog', AuditLogSchema);