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
        console.log('Successfully logged in');
    } catch (error) {
        console.error('Login failed:', error.response?.data || error.message);
        process.exit(1);
    }
}

async function uploadMedia(filePath, type) {
    const formData = new FormData();
    const fileName = path.basename(filePath);
    formData.append('file', fs.createReadStream(filePath));
    formData.append('type', type);
    formData.append('metadata', JSON.stringify({
      title: fileName,
      description: `Uploaded ${type} file`,
      tags: ['test']
    }));

    try {
        console.log(`Uploading ${filePath}...`);
        const response = await axios.post(`${API_URL}/media`, formData, {
            headers: {
                ...formData.getHeaders(),
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('Upload response:', response.data);
        return response.data;
    } catch (error) {
        console.error(`Failed to upload ${filePath}:`, error.response?.data || error.message);
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
            console.error('Response headers:', error.response.headers);
        } else if (error.request) {
            console.error('No response received:', error.request);
        }
        return null;
    }
}

async function createBlogPost(title, content, mediaIds) {
    try {
        const response = await axios.post(`${API_URL}/posts`, {
            title,
            content,
            mediaIds,
            status: 'published',
            tags: ['tech', 'programming', 'web development']
        }, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('Successfully created blog post:', response.data);
        return response.data;
    } catch (error) {
        console.error('Failed to create blog post:', error.response?.data || error.message);
        return null;
    }
}

async function getBlogPost(postId) {
    try {
        const response = await axios.get(`${API_URL}/posts/${postId}`, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('Successfully retrieved blog post:', response.data);
        return response.data;
    } catch (error) {
        console.error('Failed to get blog post:', error.response?.data || error.message);
        return null;
    }
}

async function updateBlogPost(postId, updates) {
    try {
        const response = await axios.put(`${API_URL}/posts/${postId}`, updates, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('Successfully updated blog post:', response.data);
        return response.data;
    } catch (error) {
        console.error('Failed to update blog post:', error.response?.data || error.message);
        return null;
    }
}

async function deleteBlogPost(postId) {
    try {
        const response = await axios.delete(`${API_URL}/posts/${postId}`, {
            headers: {
                Authorization: `Bearer ${authToken}`
            }
        });
        console.log('Successfully deleted blog post');
        return true;
    } catch (error) {
        console.error('Failed to delete blog post:', error.response?.data || error.message);
        return false;
    }
}

async function schedulePostDeletion(postId, days) {
    setTimeout(async () => {
        console.log(`Deleting post ${postId} after ${days} days...`);
        await deleteBlogPost(postId);
    }, days * 24 * 60 * 60 * 1000);
}

async function uploadTestData() {
    await login();

    // Upload media files
    const mediaFolder = path.join(__dirname, 'test-media');
    const mediaFiles = [
        { file: 'image1.jpg', type: 'image' },
        { file: 'image2.jpg', type: 'image' },
        { file: 'image3.jpg', type: 'image' },
        { file: 'video1.mp4', type: 'video' },
        { file: 'video2.mp4', type: 'video' },
        { file: 'document1.pdf', type: 'document' },
        { file: 'document2.pdf', type: 'document' }
    ];

    const uploadedMedia = [];
    for (const media of mediaFiles) {
        const filePath = path.join(mediaFolder, media.file);
        if (fs.existsSync(filePath)) {
            const result = await uploadMedia(filePath, media.type);
            if (result?.success && result.data?._id) {
                uploadedMedia.push(result.data._id);
                console.log(`Uploaded media with ID: ${result.data._id}`);
            }
        } else {
            console.warn(`File not found: ${filePath}`);
        }
    }

    // Create a blog post with the uploaded media
    const blogContent = `
# The Future of Web Development

As we dive into the world of modern web development, it's crucial to understand the evolving landscape of technologies and methodologies that shape our digital experiences.

## The Rise of AI in Web Development

Artificial Intelligence is revolutionizing how we build and maintain web applications. From automated testing to intelligent code completion, AI tools are becoming an integral part of every developer's toolkit.

## Modern Frontend Frameworks

The frontend development ecosystem continues to evolve with frameworks like Next.js, React, and Vue.js leading the way. These frameworks provide developers with powerful tools to build responsive, efficient, and user-friendly applications.

## Backend Technologies and Microservices

The backend landscape is shifting towards microservices architecture, with technologies like Node.js, Go, and Rust gaining popularity. Cloud-native development practices are becoming the standard for scalable applications.

## The Importance of Performance

Web performance remains a critical factor in user experience and SEO. Modern web applications must be optimized for speed, accessibility, and mobile responsiveness.

## Security Considerations

In an increasingly connected world, web security is more important than ever. Implementing proper authentication, authorization, and data protection measures is crucial for any web application.

## Future Trends

- WebAssembly adoption
- Edge computing
- Progressive Web Apps
- AI-driven development
- Real-time collaboration tools

## Conclusion

The web development landscape continues to evolve rapidly. Staying updated with the latest technologies and best practices is essential for building modern, efficient web applications.
    `;

    // Create a blog post
    const createdPost = await createBlogPost(
        'The Future of Web Development: A Comprehensive Guide',
        blogContent,
        uploadedMedia
    );

    if (createdPost) {
        // Read the blog post
        console.log('\nReading the created blog post...');
        const retrievedPost = await getBlogPost(createdPost._id);

        if (retrievedPost) {
            // Update the blog post
            console.log('\nUpdating the blog post...');
            const updates = {
                title: 'The Future of Web Development: 2025 Edition',
                tags: [...retrievedPost.tags, 'artificial-intelligence', '2025'],
                content: retrievedPost.content + '\n\nUpdated: August 2025'
            };
            const updatedPost = await updateBlogPost(createdPost._id, updates);

            // Demonstrate immediate editing and quick deletion
            if (updatedPost) {
                console.log('\nWaiting 2 seconds before final edit...');
                await new Promise(resolve => setTimeout(resolve, 2000));

                console.log('\nMaking final edit to the blog post...');
                const finalUpdate = {
                    title: 'The Future of Web Development: Final Version',
                    content: updatedPost.content + '\n\nThis is the final edit before deletion.',
                    tags: [...updatedPost.tags, 'final-version']
                };
                const finalPost = await updateBlogPost(updatedPost._id, finalUpdate);

                if (finalPost) {
                    console.log('\nPost will be deleted in 4 seconds...');
                    setTimeout(async () => {
                        console.log('\nDeleting post now...');
                        const deleted = await deleteBlogPost(finalPost._id);
                        if (deleted) {
                            console.log('Post successfully deleted!');
                            
                            // Verify deletion by trying to fetch the post
                            console.log('\nTrying to fetch the deleted post...');
                            const checkPost = await getBlogPost(finalPost._id);
                            if (!checkPost) {
                                console.log('Confirmed: Post no longer exists');
                            }
                        }
                    }, 4000);
                }
            }
        }
    }
}

uploadTestData().catch(console.error);

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nGracefully shutting down...');
    process.exit(0);
});
