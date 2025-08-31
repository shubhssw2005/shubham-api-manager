# 🧪 Test Data for Groot API System

This directory contains sample data files for testing your Groot API system with real-world scenarios.

## 📁 Files Included

### 📄 Documents

- **sample-document.pdf** - Sample PDF document for media upload testing
- **sample-blog-post.json** - Complete blog post with metadata for API testing

### 🖼️ Images & Media

- **sample-image.txt** - Base64 encoded sample image data
- **sample-video-metadata.json** - Video file metadata for media processing tests

### 📊 Performance Data

- **large-dataset.json** - Large dataset (100K records) for performance testing
- **sample-events.json** - Event sourcing test data with various event types

## 🚀 How to Use

### Run All Tests

```bash
npm run test:system
```

### Run Individual Test Suites

```bash
# Test Node.js API functionality
npm run test:comprehensive

# Test C++ integration performance
npm run test:cpp

# Quick demo test
npm run test:demo
```

### Manual Testing

You can also use these files manually:

1. **Upload Media**: Use the PDF and image files to test media upload endpoints
2. **Create Posts**: Use the blog post JSON to test content creation
3. **Performance Testing**: Use the large dataset to test system scalability
4. **Event Processing**: Use sample events to test real-time processing

## 📊 Expected Results

When you run the tests, you should see:

- ✅ **API Health Checks** - All endpoints responding
- ✅ **Authentication** - User login/registration working
- ✅ **CRUD Operations** - Create, read, update blog posts
- ✅ **Media Processing** - File uploads and metadata extraction
- ✅ **Performance Metrics** - Response times and throughput
- ✅ **Event Processing** - Real-time event handling
- ✅ **Backup System** - Data backup and recovery

## 🎯 Performance Targets

Your system should achieve:

- **API Response Time**: < 10ms (< 1ms with C++ system)
- **Throughput**: > 1000 RPS (> 100K RPS with C++ system)
- **Media Processing**: < 500ms per file
- **Event Processing**: > 1000 events/second
- **Cache Hit Ratio**: > 90%

## 🔧 Troubleshooting

If tests fail:

1. **Check Server Status**: Make sure your server is running (`npm run dev`)
2. **Database Connection**: Verify MongoDB is accessible
3. **Environment Variables**: Check your `.env` file configuration
4. **Dependencies**: Run `npm install` to ensure all packages are installed

## 📈 Next Steps

After successful testing:

1. **Deploy C++ System** - Add ultra-low latency layer for enterprise performance
2. **Scale Infrastructure** - Implement multi-region deployment
3. **Add Security** - Implement enterprise security features
4. **Monitor Performance** - Set up comprehensive monitoring

Your system is ready for real-world data processing! 🚀
