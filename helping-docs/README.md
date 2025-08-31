# 🌳 Groot API - Universal Data Management System

A comprehensive, production-ready API system with universal data management, event sourcing, soft deletes, and MinIO backup integration.

## 🚀 Features

### ✅ Universal Data Management
- **Soft Deletes** - Never lose data with complete audit trails
- **Event Sourcing** - Immutable event log for all changes
- **Universal CRUD** - Consistent APIs across all data types
- **Search & Pagination** - Full-text search with efficient pagination
- **Media Management** - File uploads with metadata tracking

### ✅ Blog Management System
- Create, read, update, delete blog posts
- Rich content with SEO optimization
- Media attachments (images, documents, videos)
- Tag-based organization
- Status management (draft, published, archived)

### ✅ Backup & Recovery
- **MinIO Integration** - S3-compatible object storage
- **Automated Backups** - Complete user data backups
- **Presigned URLs** - Secure download links
- **JSON Format** - Human-readable backup format

### ✅ Production Ready
- JWT Authentication & Authorization
- Comprehensive error handling
- Performance optimization with proper indexing
- Health checks and monitoring
- CLI tools for maintenance

## 🛠️ Tech Stack

- **Backend**: Next.js API Routes
- **Database**: MongoDB with Mongoose
- **Storage**: MinIO (S3-compatible)
- **Authentication**: JWT
- **File Processing**: Formidable
- **Event Sourcing**: Custom outbox pattern

## 📦 Installation

### Prerequisites
- Node.js 18+
- MongoDB
- Docker (for MinIO)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/shubhssw2005/groot-api.git
cd groot-api
```

2. **Install dependencies**
```bash
npm install
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start MinIO (Docker)**
```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

5. **Setup MinIO buckets**
```bash
npm run setup:minio
```

6. **Seed the database**
```bash
npm run seed:ecommerce
```

7. **Start the development server**
```bash
npm run dev
```

## 🔧 Configuration

### Environment Variables

```env
# Database
MONGODB_URI=mongodb://localhost:27017/groot-api

# JWT
JWT_SECRET=your-super-secret-jwt-key

# MinIO Configuration
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BACKUP_BUCKET=user-data-backups

# Data Management
ENABLE_S3_ARCHIVE=false
OUTBOX_BATCH_SIZE=100
ARCHIVE_RETENTION_DAYS=90
```

## 📚 API Documentation

### Authentication
```bash
# Login
POST /api/auth/login
{
  "email": "admin@example.com",
  "password": "admin123"
}
```

### Universal CRUD Operations
All models support these endpoints:
```bash
GET    /api/universal/{model}           # List with pagination/search
GET    /api/universal/{model}/{id}      # Get single document
POST   /api/universal/{model}           # Create new document
PUT    /api/universal/{model}/{id}      # Update document
DELETE /api/universal/{model}/{id}      # Soft delete document
POST   /api/universal/{model}/{id}/restore # Restore deleted document
```

### Blog Management
```bash
# Create blog post
POST /api/test-universal
{
  "title": "My Blog Post",
  "content": "Content here...",
  "status": "published",
  "tags": ["tech", "api"]
}

# Search posts
GET /api/test-universal?search=tech&page=1&limit=10

# Soft delete
DELETE /api/test-soft-delete?id={postId}
```

### Backup System
```bash
# Create backup
POST /api/backup-blog
# Returns: Download URL, backup metadata

# List backups
GET /api/backup-blog
```

### Media Management
```bash
# Upload file
POST /api/media
Content-Type: multipart/form-data
file: [binary data]
metadata: {"title": "My File", "description": "..."}
```

## 🗄️ Data Models

### Post Model
```javascript
{
  title: String,
  content: String,
  excerpt: String,
  status: 'draft' | 'published' | 'archived',
  tags: [String],
  author: ObjectId,
  mediaIds: [ObjectId],
  featured: Boolean,
  seoTitle: String,
  seoDescription: String,
  // Soft delete fields
  isDeleted: Boolean,
  deletedAt: Date,
  deletedBy: ObjectId,
  tombstoneReason: String
}
```

### Event Sourcing (Outbox)
```javascript
{
  aggregate: String,
  aggregateId: ObjectId,
  eventType: String,
  payload: Object,
  version: Number,
  processed: Boolean,
  createdAt: Date
}
```

## 🛠️ Management Commands

### Data Management
```bash
npm run data:stats          # Model statistics
npm run data:process        # Process outbox events
npm run data:archive        # Archive old data
npm run data:cleanup        # Cleanup processed events
npm run data:validate       # Validate data integrity
```

### MinIO Operations
```bash
npm run setup:minio         # Setup MinIO buckets
npm run test:minio          # Test MinIO connection
```

### Database Operations
```bash
npm run seed:ecommerce      # Seed sample data
npm run data:indexes        # Create database indexes
```

## 🏗️ Architecture

### Universal Data Management Pattern
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Business Logic  │    │  Data Layer     │
│                 │    │                  │    │                 │
│ Universal CRUD  │───▶│ Soft Deletes     │───▶│ MongoDB         │
│ Authentication  │    │ Event Sourcing   │    │ Indexes         │
│ Validation      │    │ Audit Trails     │    │ Relationships   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Event Stream    │
                       │                  │
                       │ Outbox Pattern   │
                       │ MinIO Storage    │
                       │ Backup System    │
                       └──────────────────┘
```

### Event Sourcing Flow
```
User Action ──▶ API Endpoint ──▶ Business Logic ──▶ Database + Event
                                                          │
                                                          ▼
MinIO Backup ◀── Outbox Worker ◀── Event Stream ◀── Outbox Table
```

## 🧪 Testing

### Manual Testing
```bash
# Get authentication token
JWT_TOKEN=$(curl -s -X POST http://localhost:3005/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@ecommerce.com", "password": "admin123"}' | \
  jq -r '.token')

# Create a blog post
curl -X POST http://localhost:3005/api/test-universal \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Post", "content": "Test content", "status": "published"}'

# Create backup
curl -X POST http://localhost:3005/api/backup-blog \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### Test Credentials
- **Admin**: admin@ecommerce.com / admin123
- **Customer**: john.doe@example.com / customer123

## 📊 Monitoring

### Health Checks
```bash
# System health
GET /api/universal/post/stats

# Event processing
npm run data:stats

# MinIO status
npm run test:minio
```

### Key Metrics
- Active documents vs soft deleted
- Event processing lag
- Backup success rate
- Storage usage

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3005
CMD ["npm", "start"]
```

### Environment Setup
```bash
# Production environment
NODE_ENV=production
MONGODB_URI=mongodb://your-mongo-cluster
MINIO_ENDPOINT=your-minio-server.com
MINIO_USE_SSL=true
```

## 🔒 Security

### Authentication
- JWT-based authentication
- Role-based access control
- Token expiration handling

### Data Protection
- Soft deletes prevent data loss
- Complete audit trails
- Encrypted storage options

### API Security
- Input validation
- Rate limiting ready
- CORS configuration

## 📈 Performance

### Database Optimization
- Proper indexing on query fields
- Pagination for large datasets
- Soft delete filtering

### Storage Optimization
- MinIO lifecycle policies
- Compressed backups
- Presigned URL caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [MinIO Setup Guide](MINIO_SETUP_GUIDE.md)
- [AWS Setup Guide](AWS_SETUP_GUIDE.md)
- [Universal Data Management](UNIVERSAL_DATA_MANAGEMENT.md)
- [System Test Results](SYSTEM_TEST_RESULTS.md)

### Common Issues
- **Connection Refused**: Start MinIO server
- **Authentication Failed**: Check JWT token
- **Bucket Not Found**: Run `npm run setup:minio`

### Getting Help
- Create an issue on GitHub
- Check the documentation
- Review test results

---

## 🎯 Roadmap

### Current Features ✅
- [x] Universal CRUD operations
- [x] Soft delete system
- [x] Event sourcing
- [x] MinIO backup integration
- [x] Blog management
- [x] Media uploads
- [x] Search and pagination

### Upcoming Features 🚧
- [ ] Real-time notifications
- [ ] Advanced analytics
- [ ] Multi-tenant support
- [ ] GraphQL API
- [ ] Webhook system
- [ ] Advanced caching

---

**Built with ❤️ for universal data management**

🌳 **Groot API** - *"I am Groot, and I manage all your data!"*