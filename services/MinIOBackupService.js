import * as Minio from 'minio';
import mongoose from 'mongoose';

/**
 * MinIO Backup Service
 * S3-compatible object storage for user data backups
 */
class MinIOBackupService {
    constructor() {
        this.minioClient = new Minio.Client({
            endPoint: process.env.MINIO_ENDPOINT || 'localhost',
            port: parseInt(process.env.MINIO_PORT) || 9000,
            useSSL: process.env.MINIO_USE_SSL === 'true',
            accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
            secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
        });
        
        this.bucketName = process.env.MINIO_BACKUP_BUCKET || 'user-data-backups';
        this.isMinIOMode = true;
    }

    /**
     * Create a backup of user's blog data and upload to MinIO
     */
    async backupUserBlogData(userId, userEmail) {
        try {
            console.log(`📦 Creating MinIO backup for user: ${userEmail}`);
            
            // Get user's blog posts
            const Post = mongoose.model('Post');
            const Media = mongoose.model('Media');
            
            const posts = await Post.find({ author: userId })
                .populate('mediaIds')
                .lean();
            
            const userMedia = await Media.find({ uploadedBy: userId }).lean();
            
            // Create backup metadata
            const backupData = {
                user: {
                    id: userId,
                    email: userEmail,
                    backupDate: new Date().toISOString()
                },
                posts: posts.map(post => ({
                    id: post._id,
                    title: post.title,
                    content: post.content,
                    excerpt: post.excerpt,
                    status: post.status,
                    tags: post.tags,
                    featured: post.featured,
                    seoTitle: post.seoTitle,
                    seoDescription: post.seoDescription,
                    seoKeywords: post.seoKeywords,
                    createdAt: post.createdAt,
                    updatedAt: post.updatedAt,
                    mediaFiles: post.mediaIds || []
                })),
                media: userMedia.map(media => ({
                    id: media._id,
                    filename: media.filename,
                    originalName: media.originalName,
                    mimeType: media.mimeType,
                    size: media.size,
                    url: media.url,
                    description: media.description,
                    tags: media.tags,
                    createdAt: media.createdAt
                })),
                statistics: {
                    totalPosts: posts.length,
                    totalMedia: userMedia.length,
                    totalSize: userMedia.reduce((sum, media) => sum + (media.size || 0), 0)
                },
                readme: this.generateReadme(userEmail, posts.length, userMedia.length),
                posts_formatted: posts.map((post, index) => ({
                    filename: `${index + 1}-${this.sanitizeFilename(post.title)}.md`,
                    content: this.formatPostAsMarkdown(post)
                }))
            };
            
            // Create backup buffer
            const backupBuffer = Buffer.from(JSON.stringify(backupData, null, 2), 'utf8');
            
            // Upload to MinIO
            return await this.uploadToMinIO(userId, userEmail, backupBuffer, backupData);
            
        } catch (error) {
            console.error('❌ MinIO backup failed:', error);
            throw error;
        }
    }

    /**
     * Upload backup to MinIO
     */
    async uploadToMinIO(userId, userEmail, buffer, metadata) {
        try {
            // Ensure bucket exists
            await this.ensureBucketExists();
            
            const objectName = `user-backups/${userId}/${Date.now()}-blog-backup.json`;
            
            console.log(`☁️  Uploading to MinIO: ${objectName}`);
            console.log(`📊 Backup size: ${this.formatBytes(buffer.length)}`);
            console.log(`📝 Posts: ${metadata.statistics.totalPosts}`);
            console.log(`📎 Media files: ${metadata.statistics.totalMedia}`);
            
            // Upload to MinIO
            const uploadInfo = await this.minioClient.putObject(
                this.bucketName,
                objectName,
                buffer,
                buffer.length,
                {
                    'Content-Type': 'application/json',
                    'X-User-Email': userEmail,
                    'X-Backup-Date': metadata.user.backupDate,
                    'X-Posts-Count': metadata.statistics.totalPosts.toString(),
                    'X-Media-Count': metadata.statistics.totalMedia.toString()
                }
            );
            
            // Generate presigned URL for download (valid for 1 hour)
            const downloadUrl = await this.minioClient.presignedGetObject(
                this.bucketName,
                objectName,
                3600 // 1 hour
            );
            
            console.log('✅ Upload successful!');
            
            return {
                success: true,
                objectName: objectName,
                bucket: this.bucketName,
                size: buffer.length,
                downloadUrl: downloadUrl,
                etag: uploadInfo.etag,
                metadata: {
                    userEmail: userEmail,
                    backupDate: metadata.user.backupDate,
                    postsCount: metadata.statistics.totalPosts,
                    mediaCount: metadata.statistics.totalMedia,
                    totalSize: this.formatBytes(buffer.length)
                }
            };
            
        } catch (error) {
            console.error('❌ MinIO upload failed:', error);
            throw error;
        }
    }

    /**
     * Ensure backup bucket exists
     */
    async ensureBucketExists() {
        try {
            const exists = await this.minioClient.bucketExists(this.bucketName);
            
            if (!exists) {
                console.log(`🪣 Creating MinIO bucket: ${this.bucketName}`);
                await this.minioClient.makeBucket(this.bucketName, 'us-east-1');
                console.log(`✅ Bucket created: ${this.bucketName}`);
            }
            
        } catch (error) {
            console.error(`❌ Failed to ensure bucket exists: ${error.message}`);
            throw error;
        }
    }

    /**
     * List user backups in MinIO
     */
    async listUserBackups(userId) {
        try {
            const prefix = `user-backups/${userId}/`;
            const objectsStream = this.minioClient.listObjects(this.bucketName, prefix, true);
            
            const backups = [];
            
            return new Promise((resolve, reject) => {
                objectsStream.on('data', (obj) => {
                    backups.push({
                        name: obj.name,
                        size: obj.size,
                        lastModified: obj.lastModified,
                        etag: obj.etag,
                        sizeFormatted: this.formatBytes(obj.size)
                    });
                });
                
                objectsStream.on('end', () => {
                    resolve({
                        success: true,
                        backups,
                        count: backups.length
                    });
                });
                
                objectsStream.on('error', (error) => {
                    console.error('Error listing backups:', error);
                    reject(error);
                });
            });
            
        } catch (error) {
            console.error('Error listing MinIO backups:', error);
            throw error;
        }
    }

    /**
     * Get backup download URL
     */
    async getBackupDownloadUrl(objectName, expirySeconds = 3600) {
        try {
            const downloadUrl = await this.minioClient.presignedGetObject(
                this.bucketName,
                objectName,
                expirySeconds
            );
            
            return {
                success: true,
                downloadUrl,
                expiresIn: expirySeconds
            };
            
        } catch (error) {
            console.error('Error generating download URL:', error);
            throw error;
        }
    }

    /**
     * Delete backup from MinIO
     */
    async deleteBackup(objectName) {
        try {
            await this.minioClient.removeObject(this.bucketName, objectName);
            
            return {
                success: true,
                message: 'Backup deleted successfully'
            };
            
        } catch (error) {
            console.error('Error deleting backup:', error);
            throw error;
        }
    }

    /**
     * Test MinIO connection
     */
    async testConnection() {
        try {
            console.log('🔍 Testing MinIO connection...');
            
            // Test by listing buckets
            const buckets = await this.minioClient.listBuckets();
            
            console.log('✅ MinIO connection successful!');
            console.log(`📊 Found ${buckets.length} buckets`);
            
            if (buckets.length > 0) {
                buckets.forEach(bucket => {
                    console.log(`  - ${bucket.name} (created: ${bucket.creationDate.toISOString().split('T')[0]})`);
                });
            }
            
            return {
                success: true,
                buckets: buckets.length,
                endpoint: `${this.minioClient.protocol}//${this.minioClient.host}:${this.minioClient.port}`
            };
            
        } catch (error) {
            console.error('❌ MinIO connection failed:', error);
            throw error;
        }
    }

    /**
     * Format post as markdown
     */
    formatPostAsMarkdown(post) {
        return `# ${post.title}\n\n` +
            `**Status**: ${post.status}\n` +
            `**Created**: ${post.createdAt}\n` +
            `**Updated**: ${post.updatedAt}\n` +
            `**Tags**: ${post.tags?.join(', ') || 'None'}\n` +
            `**Featured**: ${post.featured ? 'Yes' : 'No'}\n\n` +
            (post.excerpt ? `## Excerpt\n${post.excerpt}\n\n` : '') +
            `## Content\n\n${post.content}\n\n` +
            (post.seoTitle ? `## SEO\n**Title**: ${post.seoTitle}\n**Description**: ${post.seoDescription || 'None'}\n**Keywords**: ${post.seoKeywords?.join(', ') || 'None'}\n` : '');
    }

    /**
     * Generate README content
     */
    generateReadme(userEmail, postsCount, mediaCount) {
        return `# Blog Data Backup\n\n` +
            `**User**: ${userEmail}\n` +
            `**Backup Date**: ${new Date().toISOString()}\n` +
            `**Total Posts**: ${postsCount}\n` +
            `**Total Media Files**: ${mediaCount}\n` +
            `**Storage**: MinIO Object Storage\n\n` +
            `## Contents\n\n` +
            `This JSON file contains:\n` +
            `- Complete backup metadata\n` +
            `- All blog posts with full content\n` +
            `- Media files information\n` +
            `- Formatted markdown versions of posts\n\n` +
            `## Usage\n\n` +
            `You can use this backup to:\n` +
            `- Restore your blog content\n` +
            `- Migrate to another platform\n` +
            `- Archive your writing\n` +
            `- Analyze your content\n\n` +
            `This backup was created by the Universal Data Management System with MinIO storage.`;
    }

    /**
     * Utility functions
     */
    sanitizeFilename(filename) {
        return filename
            .replace(/[^a-z0-9]/gi, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '')
            .toLowerCase()
            .substring(0, 50);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

export default MinIOBackupService;