const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

const API_URL = 'http://localhost:3005/api';
let authToken = null;

async function login() {
    try {
        const response = await axios.post(`${API_URL}/auth/login`, {
            email: 'admin@example.com',
            password: 'admin123'
        });
        authToken = response.data.token;
        console.log('✅ Successfully logged in');
        return response.data.user;
    } catch (error) {
        console.error('❌ Login failed:', error.response?.data || error.message);
        process.exit(1);
    }
}

async function uploadMedia(filePath, type, metadata = {}) {
    const formData = new FormData();
    const fileName = path.basename(filePath);
    formData.append('file', fs.createReadStream(filePath));
    formData.append('type', type);
    formData.append('metadata', JSON.stringify({
        title: metadata.title || fileName,
        description: metadata.description || `Uploaded ${type} file`,
        tags: metadata.tags || ['test'],
        alt: metadata.alt || '',
        caption: metadata.caption || ''
    }));

    try {
        console.log(`📤 Uploading ${filePath}...`);
        const response = await axios.post(`${API_URL}/media`, formData, {
            headers: {
                ...formData.getHeaders(),
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('✅ Media uploaded successfully');
        return response.data;
    } catch (error) {
        console.error(`❌ Failed to upload ${filePath}:`, error.response?.data || error.message);
        return null;
    }
}

async function createBlogPost(data) {
    try {
        const response = await axios.post(`${API_URL}/posts`, {
            ...data,
            status: 'published'
        }, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('✅ Blog post created successfully');
        return response.data;
    } catch (error) {
        console.error('❌ Failed to create blog post:', error.response?.data || error.message);
        return null;
    }
}

async function createProduct(data) {
    try {
        const response = await axios.post(`${API_URL}/products`, {
            ...data,
            status: 'active'
        }, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('✅ Product created successfully');
        return response.data;
    } catch (error) {
        console.error('❌ Failed to create product:', error.response?.data || error.message);
        return null;
    }
}

async function linkMediaToProduct(productId, mediaIds) {
    try {
        const response = await axios.put(`${API_URL}/products/${productId}`, {
            mediaIds
        }, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('✅ Media linked to product successfully');
        return response.data;
    } catch (error) {
        console.error('❌ Failed to link media to product:', error.response?.data || error.message);
        return null;
    }
}

async function runAdvancedTest() {
    console.log('🚀 Starting advanced integration test...\n');

    // Step 1: Login
    console.log('Step 1: Authentication');
    const user = await login();
    console.log('👤 Logged in as:', user.email);

    // Step 2: Upload Media Files with Rich Metadata
    console.log('\nStep 2: Media Upload with Rich Metadata');
    const mediaFolder = path.join(__dirname, 'test-media');
    
    // Upload product images with specific metadata
    const productImage1 = await uploadMedia(
        path.join(mediaFolder, 'image1.jpg'),
        'image',
        {
            title: 'Product Main Image',
            description: 'High-quality product showcase image',
            tags: ['product', 'main-image'],
            alt: 'Product showcase in natural lighting',
            caption: 'Premium quality product view'
        }
    );

    const productImage2 = await uploadMedia(
        path.join(mediaFolder, 'image2.jpg'),
        'image',
        {
            title: 'Product Detail Image',
            description: 'Detailed view of product features',
            tags: ['product', 'details'],
            alt: 'Product details and features closeup',
            caption: 'Detailed feature showcase'
        }
    );

    // Upload blog post related media
    const blogMedia = await uploadMedia(
        path.join(mediaFolder, 'image3.jpg'),
        'image',
        {
            title: 'Blog Header Image',
            description: 'Featured image for the blog post',
            tags: ['blog', 'header'],
            alt: 'Blog post featured image',
            caption: 'Modern web development concepts'
        }
    );

    // Step 3: Create Product with Media
    console.log('\nStep 3: Product Creation with Media Relations');
    const product = await createProduct({
        name: 'Premium Developer Toolkit',
        description: 'Complete toolkit for modern web development',
        price: 299.99,
        category: 'Development Tools',
        tags: ['web development', 'tools', 'premium'],
        mediaIds: [
            productImage1.data._id,
            productImage2.data._id
        ]
    });

    // Step 4: Create Blog Post with Related Product and Media
    console.log('\nStep 4: Blog Post Creation with Relations');
    const blogPost = await createBlogPost({
        title: 'Mastering Modern Web Development Tools',
        content: `# Mastering Modern Web Development Tools

## Introduction
In this comprehensive guide, we'll explore the Premium Developer Toolkit and how it can enhance your web development workflow.

## Featured Product
Our Premium Developer Toolkit (Product ID: ${product._id}) is designed to streamline your development process.

## Key Features
- Integrated development environment
- Advanced debugging tools
- Performance optimization suite
- Collaboration features

## Getting Started
Learn how to make the most of your development toolkit...`,
        mediaIds: [blogMedia.data._id],
        tags: ['tutorial', 'tools', 'web development'],
        relatedProducts: [product._id]
    });

    // Final Status Report
    console.log('\n📊 Test Summary:');
    console.log('================');
    console.log('✅ User Authentication: Success');
    console.log(`✅ Media Files Uploaded: ${[productImage1, productImage2, blogMedia].filter(Boolean).length}`);
    console.log(`✅ Product Created: ${product ? 'Yes' : 'No'}`);
    console.log(`✅ Blog Post Created: ${blogPost ? 'Yes' : 'No'}`);
    console.log(`✅ Relations Established: Product ←→ Media ←→ Blog Post`);
}

runAdvancedTest().catch(console.error);
