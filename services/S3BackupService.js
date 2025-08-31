import { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import mongoose from 'mongoose';

/**
 * S3 Backup Service
 * Handles user data backup to AWS S3 as zip files
 */
class S3BackupService {
    constructor() {
        this.s3Client = new S3Client({
            region: process.env.AWS_REGION || 'us-east-1',
            credentials: {
                accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'test-key',
                secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'test-secret'
            }
        });
        
        this.bucketName = process.env.S3_BACKUP_BUCKET || 'user-data-backups';
        this.isLocalMode = !process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'test-key';
    }

    /**
     * Create a zip file of user's blog data and upload to S3
     */
    async backupUserBlogData(userId, userEmail) {
        try {
            console.log(`📦 Creating backup for user: ${userEmail}`);
            
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
                    status: post.status,
                    tags: post.tags,
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
                }
            };
            
            // Create backup file in memory
            const backupBuffer = await this.createBackupBuffer(backupData);
            
            // Upload to S3 (or simulate in local mode)
            const s3Key = `user-backups/${userId}/${Date.now()}-blog-backup.json`;
            
            if (this.isLocalMode) {
                return await this.simulateS3Upload(s3Key, backupBuffer, backupData);
            } else {
                return await this.uploadToS3(s3Key, backupBuffer, backupData);
            }
            
        } catch (error) {
            console.error('❌ Backup failed:', error);
            throw error;
        }
    }

    /**
     * Create backup data as JSON buffer
     */
    async createBackupBuffer(backupData) {
        // Create a comprehensive backup with formatted content
        const formattedBackup = {
            ...backupData,
            readme: this.generateReadme(backupData),
            posts_formatted: backupData.posts.map((post, index) => ({
                filename: `${index + 1}-${this.sanitizeFilename(post.title)}.md`,
                content: this.formatPostAsMarkdown(post)
            })),
            media_list: this.formatMediaList(backupData.media)
        };
        
        const backupJson = JSON.stringify(formattedBackup, null, 2);
        return Buffer.from(backupJson, 'utf8');
    }

    /**
     * Format post as markdown
     */
    formatPostAsMarkdown(post) {
        return `# ${post.title}\n\n` +
            `**Status**: ${post.status}\n` +
            `**Created**: ${post.createdAt}\n` +
            `**Updated**: ${post.updatedAt}\n` +
            `**Tags**: ${post.tags?.join(', ') || 'None'}\n\n` +
            `---\n\n${post.content}`;
    }

    /**
     * Format media list
     */
    formatMediaList(media) {
        if (!media || media.length === 0) return 'No media files';
        
        return media.map(file => 
            `- ${file.originalName} (${file.mimeType}, ${this.formatBytes(file.size)})\n` +
            `  Description: ${file.description || 'None'}\n` +
            `  URL: ${file.url}\n` +
            `  Created: ${file.createdAt}`
        ).join('\n\n');
    }

    /**
     * Generate README content
     */
    generateReadme(backupData) {
        return `# Blog Data Backup\n\n` +
            `**User**: ${backupData.user.email}\n` +
            `**Backup Date**: ${backupData.user.backupDate}\n` +
            `**Total Posts**: ${backupData.statistics.totalPosts}\n` +
            `**Total Media Files**: ${backupData.statistics.totalMedia}\n` +
            `**Total Media Size**: ${this.formatBytes(backupData.statistics.totalSize)}\n\n` +
            `## Contents\n\n` +
            `This JSON file contains:\n` +
            `- Complete backup metadata\n` +
            `- All blog posts with full content\n` +
            `- Media files information\n` +
            `- Formatted markdown versions of posts\n\n` +
            `This backup was created by the Universal Data Management System.`;
    }

    /**
     * Upload zip file to S3
     */
    async uploadToS3(key, buffer, metadata) {
        const command = new PutObjectCommand({
            Bucket: this.bucketName,
            Key: key,
            Body: buffer,
            ContentType: 'application/json',
            Metadata: {
                'user-id': metadata.user.id,
                'user-email': metadata.user.email,
                'backup-date': metadata.user.backupDate,
                'posts-count': metadata.statistics.totalPosts.toString(),
                'media-count': metadata.statistics.totalMedia.toString()
            },
            ServerSideEncryption: 'AES256'
        });

        const result = await this.s3Client.send(command);
        
        // Generate presigned URL for download (valid for 1 hour)
        const downloadUrl = await getSignedUrl(
            this.s3Client,
            new GetObjectCommand({
                Bucket: this.bucketName,
                Key: key
            }),
            { expiresIn: 3600 }
        );

        return {
            success: true,
            s3Key: key,
            bucket: this.bucketName,
            size: buffer.length,
            downloadUrl,
            metadata: {
                userEmail: metadata.user.email,
                backupDate: metadata.user.backupDate,
                postsCount: metadata.statistics.totalPosts,
                mediaCount: metadata.statistics.totalMedia,
                totalSize: this.formatBytes(buffer.length)
            }
        };
    }

    /**
     * Simulate S3 upload for local development
     */
    async simulateS3Upload(key, buffer, metadata) {
        console.log(`🔧 LOCAL MODE: Simulating S3 upload to ${key}`);
        console.log(`📊 Backup size: ${this.formatBytes(buffer.length)}`);
        console.log(`📝 Posts: ${metadata.statistics.totalPosts}`);
        console.log(`📎 Media files: ${metadata.statistics.totalMedia}`);
        
        // In local mode, we could save to local filesystem
        const fs = await import('fs/promises');
        const path = await import('path');
        
        const localBackupDir = './local-backups';
        const localFilePath = path.join(localBackupDir, key.replace(/\//g, '_'));
        
        try {
            await fs.mkdir(localBackupDir, { recursive: true });
            await fs.writeFile(localFilePath, buffer);
            console.log(`💾 Backup saved locally: ${localFilePath}`);
        } catch (error) {
            console.log(`⚠️  Could not save locally: ${error.message}`);
        }

        return {
            success: true,
            s3Key: key,
            bucket: this.bucketName,
            size: buffer.length,
            downloadUrl: `http://localhost:3005/api/download-backup?key=${encodeURIComponent(key)}`,
            localPath: localFilePath,
            isSimulated: true,
            metadata: {
                userEmail: metadata.user.email,
                backupDate: metadata.user.backupDate,
                postsCount: metadata.statistics.totalPosts,
                mediaCount: metadata.statistics.totalMedia,
                totalSize: this.formatBytes(buffer.length)
            }
        };
    }

    /**
     * List user backups
     */
    async listUserBackups(userId) {
        if (this.isLocalMode) {
            return {
                success: true,
                backups: [],
                message: 'Local mode - backups stored locally'
            };
        }

        try {
            const command = new ListObjectsV2Command({
                Bucket: this.bucketName,
                Prefix: `user-backups/${userId}/`
            });

            const result = await this.s3Client.send(command);
            
            const backups = (result.Contents || []).map(obj => ({
                key: obj.Key,
                size: obj.Size,
                lastModified: obj.LastModified,
                sizeFormatted: this.formatBytes(obj.Size)
            }));

            return {
                success: true,
                backups,
                count: backups.length
            };
        } catch (error) {
            console.error('Error listing backups:', error);
            throw error;
        }
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

export default S3BackupService;