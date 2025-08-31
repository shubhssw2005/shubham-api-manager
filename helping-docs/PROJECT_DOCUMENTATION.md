# Full-Stack Blog & Media Management System Documentation

## 🏗 Project Structure

### 📁 Directory Layout
```
simple-app-main/
├── components/          # React components
├── context/            # React context providers
├── lib/                # Core library code
├── middleware/         # API middleware
├── models/            # Database models
├── pages/             # Next.js pages & API routes
├── scripts/           # Utility scripts
├── styles/            # CSS styles
└── tests/             # Test files
```

## 🔑 Key Components

### 1. Authentication System
- **Location**: `/pages/api/auth/`
- **Key Files**:
  - `login.js`: Handles user authentication
  - `register.js`: User registration
  - `middleware/auth.js`: Authentication middleware
- **Related Models**: `User.js`, `Role.js`
- **JWT Implementation**: `lib/jwt.js`

### 2. Media Management
- **Location**: `/pages/api/media/`
- **Storage Providers**:
  - Local: `lib/storage/LocalStorageProvider.js`
  - S3: `lib/storage/S3StorageProvider.js`
- **Media Processing**: `lib/fileProcessor/`
  - File validation
  - Image processing
  - Thumbnail generation
- **Media Model**: `models/Media.js`
- **Upload Directory**: `/uploads/`

### 3. Blog System
- **Location**: `/pages/api/posts/`
- **Key Files**:
  - `[id].js`: Individual post operations (GET, PUT, DELETE)
  - `index.js`: Post listing and creation
- **Models**: `models/Post.js`
- **Relations**:
  - Links to Media
  - Author association
  - Tags and categories

### 4. Product Management
- **Location**: `/pages/api/products/`
- **Model**: `models/Product.js`
- **Features**:
  - Product metadata
  - Media associations
  - Category management
  - Pricing information

## 🗄 Database Structure

### MongoDB Collections
1. **Users Collection**
   - User authentication data
   - Profile information
   - Role associations

2. **Media Collection**
   - File metadata
   - Storage information
   - Processing status
   - Usage tracking

3. **Posts Collection**
   - Blog content
   - Media relationships
   - Author information
   - Tags and categories

4. **Products Collection**
   - Product details
   - Associated media
   - Pricing information
   - Inventory status

## 🔌 API Endpoints

### Authentication
- POST `/api/auth/login`
- POST `/api/auth/register`
- GET `/api/auth/profile`

### Media
- POST `/api/media` - Upload media
- GET `/api/media` - List media
- GET `/api/media/:id` - Get specific media
- PUT `/api/media/:id` - Update media
- DELETE `/api/media/:id` - Delete media

### Blog Posts
- POST `/api/posts` - Create post
- GET `/api/posts` - List posts
- GET `/api/posts/:id` - Get specific post
- PUT `/api/posts/:id` - Update post
- DELETE `/api/posts/:id` - Delete post

### Products
- POST `/api/products` - Create product
- GET `/api/products` - List products
- GET `/api/products/:id` - Get specific product
- PUT `/api/products/:id` - Update product
- DELETE `/api/products/:id` - Delete product

## 🧪 Testing

### Test Scripts
1. **Basic Data Test**
   - Location: `/scripts/test-blog-data.js`
   - Purpose: Basic CRUD operations testing

2. **Advanced Integration Test**
   - Location: `/scripts/advanced-test.js`
   - Purpose: Tests relationships and complex scenarios

### Test Media
- Location: `/scripts/test-media/`
- Contains test files for:
  - Images (JPG)
  - Videos (MP4)
  - Documents (PDF)

## 🛠 Configuration

### Environment Variables
- Database connection: `MONGODB_URI`
- JWT secret: `JWT_SECRET`
- Storage configuration: `STORAGE_TYPE`
- API base URL: `API_URL`

### Database Migrations
- Location: `lib/migrations/`
- Key migrations:
  - `001_create_media_indexes.js`

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```

3. Run migrations:
   ```bash
   npm run migrate
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

5. Run test scripts:
   ```bash
   node scripts/advanced-test.js
   ```

## 🔗 Data Relationships

### Media ←→ Posts
- Posts can have multiple media attachments
- Media tracks usage in posts
- Automatic cleanup on post deletion

### Media ←→ Products
- Products can have multiple media views
- Main product image designation
- Gallery support

### Posts ←→ Products
- Posts can reference products
- Product showcase in blog posts
- Related products feature

## 🛡 Security Features

1. **Authentication**
   - JWT-based auth
   - Role-based access control
   - Token expiration

2. **File Security**
   - File type validation
   - Size limits
   - Virus scanning (configurable)

3. **API Security**
   - Rate limiting
   - Request validation
   - Error handling

## 🔄 Workflow Examples

### Creating a Blog Post with Media
1. Upload media files
2. Get media IDs
3. Create blog post with media references
4. Verify relationships

### Managing Products
1. Create product entry
2. Upload product images
3. Link media to product
4. Update product details

## 📝 Maintenance

### Scheduled Tasks
- Media cleanup
- Post archival
- Analytics generation

### Monitoring
- Error logging
- Performance metrics
- Usage statistics
