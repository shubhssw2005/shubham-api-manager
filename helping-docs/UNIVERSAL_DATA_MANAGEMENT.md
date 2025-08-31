# Universal Data Management System

A comprehensive, production-ready data management system that provides:
- **Soft deletes** with audit trails
- **Event sourcing** for complete data lineage
- **Automatic archiving** to cold storage
- **Universal CRUD APIs** for all models
- **Data lifecycle management**
- **Backup and recovery**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install aws-sdk zlib
```

### 2. Configure Environment
Add to your `.env` file:
```env
# AWS Configuration (optional, for S3 archiving)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
EVENT_BUCKET_NAME=your-event-stream-bucket
ARCHIVE_BUCKET_NAME=your-archive-bucket

# Data Management Settings
ENABLE_S3_ARCHIVE=false
OUTBOX_BATCH_SIZE=100
ARCHIVE_RETENTION_DAYS=90
OUTBOX_CLEANUP_DAYS=30
```

### 3. Update Your Models
Replace your existing models with the new universal pattern:

```javascript
// Before (old way)
import mongoose from 'mongoose';

const PostSchema = new mongoose.Schema({
    title: String,
    content: String
});

export default mongoose.model('Post', PostSchema);

// After (new way)
import ModelFactory from '../lib/ModelFactory.js';

const Post = ModelFactory.createModel('Post', {
    title: {
        type: String,
        required: true
    },
    content: {
        type: String,
        required: true
    }
}, {
    enableSoftDelete: true,
    enableEventSourcing: true,
    textSearchFields: ['title', 'content']
});

export default Post;
```

### 4. Use Universal APIs
All your models now have consistent CRUD endpoints:

```bash
# List posts with pagination
GET /api/universal/post?page=1&limit=10&search=google

# Get single post
GET /api/universal/post/60f1b2c3d4e5f6789abcdef0

# Create post
POST /api/universal/post
{
  "title": "My Blog Post",
  "content": "Content here..."
}

# Update post
PUT /api/universal/post/60f1b2c3d4e5f6789abcdef0
{
  "title": "Updated Title"
}

# Soft delete post
DELETE /api/universal/post/60f1b2c3d4e5f6789abcdef0

# Restore deleted post
POST /api/universal/post/60f1b2c3d4e5f6789abcdef0/restore

# Get model statistics
GET /api/universal/post/stats

# Bulk delete
POST /api/universal/post/bulk-delete
{
  "ids": ["id1", "id2", "id3"],
  "reason": "cleanup"
}
```

## 🛠️ Management Commands

### Archive Old Data
```bash
# Archive all models (90+ days old soft-deleted documents)
npm run data:archive

# Archive specific model
node scripts/data-management.js archive Post 30

# Archive with custom retention
node scripts/data-management.js archive Product 60
```

### Process Events
```bash
# Process pending outbox events
npm run data:process

# Start background worker (daemon)
npm run data:worker

# Cleanup old processed events
npm run data:cleanup
```

### Statistics & Health
```bash
# Get all model statistics
npm run data:stats

# Get specific model stats
node scripts/data-management.js stats Post

# Validate data integrity
npm run data:validate
```

### Maintenance
```bash
# Create indexes for all models
npm run data:indexes

# Export data for backup
node scripts/data-management.js export Post ./backups

# Restore from archive (emergency)
node scripts/data-management.js restore Post 60f1b2c3d4e5f6789abcdef0
```

## 📊 Features Overview

### 1. Soft Delete System
- Never lose data with user actions
- Audit trail: who, when, why
- Restore capability
- Automatic exclusion from queries

```javascript
// Soft delete a document
await Post.softDeleteById(postId, userId, 'policy_violation');

// Find only active documents (automatic)
const posts = await Post.find({}); // excludes deleted

// Find deleted documents
const deleted = await Post.findDeleted();

// Find all (including deleted)
const all = await Post.findWithDeleted();
```

### 2. Event Sourcing
- Complete audit log of all changes
- Replay capability for debugging
- Immutable event stream
- Integration with Kafka/S3

```javascript
// Events are automatically created for:
await Post.create(data);        // → PostCreated event
await Post.updateOne(filter, update); // → PostUpdated event
await Post.softDeleteById(id); // → PostDeleted event
```

### 3. Universal Repository Pattern
```javascript
import BaseRepository from '../lib/BaseRepository.js';

const postRepo = new BaseRepository(Post);

// Consistent API across all models
const posts = await postRepo.findWithPagination({}, { page: 1, limit: 10 });
const post = await postRepo.create(data, userId);
await postRepo.softDelete(id, userId, 'user_request');
```

### 4. Automatic Archiving
- Move old soft-deleted documents to cold storage
- S3 integration with lifecycle policies
- Compressed storage (gzip)
- Emergency restore capability

### 5. Data Lifecycle Management
```
Hot Storage (MongoDB)     → 0-30 days
Warm Storage (S3 IA)      → 31-180 days  
Cold Storage (Glacier)    → 181+ days
```

## 🔧 Advanced Configuration

### Custom Model with All Features
```javascript
const Product = ModelFactory.createModel('Product', {
    name: { type: String, required: true },
    price: { type: Number, required: true },
    category: { type: mongoose.Schema.Types.ObjectId, ref: 'Category' }
}, {
    enableSoftDelete: true,
    enableEventSourcing: true,
    enableTimestamps: true,
    textSearchFields: ['name', 'description'],
    uniqueFields: ['sku'],
    indexes: [
        { fields: { category: 1, price: -1 } },
        { fields: { createdAt: -1 } }
    ]
});
```

### Custom Repository
```javascript
class ProductRepository extends BaseRepository {
    async findByCategory(categoryId, options = {}) {
        return this.findWithPagination({ category: categoryId }, options);
    }
    
    async findLowStock(threshold = 10) {
        return this.findActive({ 'inventory.quantity': { $lte: threshold } });
    }
}

const productRepo = new ProductRepository(Product);
```

### Event Handlers
```javascript
// Listen to specific events
import Outbox from './models/Outbox.js';

const processProductEvents = async () => {
    const events = await Outbox.find({
        aggregate: 'Product',
        eventType: 'ProductCreated',
        processed: false
    });
    
    for (const event of events) {
        // Process event (send email, update cache, etc.)
        console.log('New product created:', event.payload);
        
        // Mark as processed
        event.processed = true;
        await event.save();
    }
};
```

## 🚨 Production Checklist

### Before Going Live:
- [ ] Configure AWS credentials for S3 archiving
- [ ] Set up S3 lifecycle policies
- [ ] Configure backup retention policies
- [ ] Set up monitoring for outbox processing
- [ ] Test restore procedures
- [ ] Configure log aggregation
- [ ] Set up alerts for failed events

### Monitoring:
```bash
# Check system health
curl http://localhost:3005/api/universal/post/stats

# Monitor outbox processing
node scripts/data-management.js stats

# Check for failed events
db.outboxes.find({ retryCount: { $gte: 5 } })
```

### Performance Tips:
1. **Indexes**: Ensure proper indexing on query fields
2. **Pagination**: Always use pagination for large datasets
3. **Archiving**: Run archive jobs during off-peak hours
4. **Monitoring**: Set up alerts for high retry counts
5. **Cleanup**: Regularly cleanup processed outbox events

## 🔍 Troubleshooting

### Common Issues:

**Outbox events not processing:**
```bash
# Check worker status
npm run data:process

# Check for errors
db.outboxes.find({ lastError: { $exists: true } })
```

**High storage usage:**
```bash
# Check what needs archiving
node scripts/data-management.js stats

# Run archive job
npm run data:archive
```

**Slow queries:**
```bash
# Create missing indexes
npm run data:indexes

# Check query performance
db.posts.explain("executionStats").find({ status: "published" })
```

## 📚 API Reference

### Universal Endpoints
All models automatically get these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/universal/{model}` | List with pagination/search |
| GET | `/api/universal/{model}/{id}` | Get single document |
| POST | `/api/universal/{model}` | Create new document |
| PUT | `/api/universal/{model}/{id}` | Update document |
| DELETE | `/api/universal/{model}/{id}` | Soft delete document |
| POST | `/api/universal/{model}/{id}/restore` | Restore deleted document |
| GET | `/api/universal/{model}/stats` | Get model statistics |
| POST | `/api/universal/{model}/bulk-delete` | Bulk soft delete |

### Query Parameters (GET requests):
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `sort`: Sort fields (e.g., `-createdAt,name`)
- `search`: Full-text search
- Any model field for filtering

### Response Format:
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "pages": 5,
    "total": 100,
    "hasNext": true,
    "hasPrev": false
  }
}
```

This system gives you enterprise-grade data management with minimal setup. All your models get consistent behavior, audit trails, and lifecycle management automatically! 🎉