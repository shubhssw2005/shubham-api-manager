import mongoose from 'mongoose';

/**
 * Simple Backup Service
 * Creates JSON backups of user data (can be extended to S3 later)
 */
class SimpleBackupService {
    constructor() {
        this.isLocalMode = true; // Always local for now
    }

    /**
     * Create a backup of user's blog data
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
            
            // Save locally and return result
            return await this.saveBackupLocally(userId, userEmail, backupBuffer, backupData);
            
        } catch (error) {
            console.error('❌ Backup failed:', error);
            throw error;
        }
    }

    /**
     * Save backup locally
     */
    async saveBackupLocally(userId, userEmail, buffer, metadata) {
        const backupKey = `user-backups/${userId}/${Date.now()}-blog-backup.json`;
        
        console.log(`🔧 LOCAL MODE: Creating backup for ${userEmail}`);
        console.log(`📊 Backup size: ${this.formatBytes(buffer.length)}`);
        console.log(`📝 Posts: ${metadata.statistics.totalPosts}`);
        console.log(`📎 Media files: ${metadata.statistics.totalMedia}`);
        
        // Save to local filesystem
        try {
            const fs = await import('fs/promises');
            const path = await import('path');
            
            const localBackupDir = './local-backups';
            const localFilePath = path.join(localBackupDir, backupKey.replace(/\//g, '_'));
            
            await fs.mkdir(localBackupDir, { recursive: true });
            await fs.writeFile(localFilePath, buffer);
            console.log(`💾 Backup saved locally: ${localFilePath}`);
            
            return {
                success: true,
                backupKey: backupKey,
                localPath: localFilePath,
                size: buffer.length,
                downloadUrl: `http://localhost:3005/api/download-backup?key=${encodeURIComponent(backupKey)}`,
                isSimulated: true,
                metadata: {
                    userEmail: userEmail,
                    backupDate: metadata.user.backupDate,
                    postsCount: metadata.statistics.totalPosts,
                    mediaCount: metadata.statistics.totalMedia,
                    totalSize: this.formatBytes(buffer.length)
                }
            };
            
        } catch (error) {
            console.log(`⚠️  Could not save locally: ${error.message}`);
            
            // Return in-memory backup info even if file save fails
            return {
                success: true,
                backupKey: backupKey,
                size: buffer.length,
                downloadUrl: null,
                isSimulated: true,
                inMemoryOnly: true,
                metadata: {
                    userEmail: userEmail,
                    backupDate: metadata.user.backupDate,
                    postsCount: metadata.statistics.totalPosts,
                    mediaCount: metadata.statistics.totalMedia,
                    totalSize: this.formatBytes(buffer.length)
                }
            };
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
            `**Total Media Files**: ${mediaCount}\n\n` +
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
            `This backup was created by the Universal Data Management System.`;
    }

    /**
     * List user backups (local mode)
     */
    async listUserBackups(userId) {
        try {
            const fs = await import('fs/promises');
            const path = await import('path');
            
            const localBackupDir = './local-backups';
            const userPrefix = `user-backups_${userId}_`;
            
            try {
                const files = await fs.readdir(localBackupDir);
                const userBackups = files
                    .filter(file => file.startsWith(userPrefix))
                    .map(async (file) => {
                        const filePath = path.join(localBackupDir, file);
                        const stats = await fs.stat(filePath);
                        return {
                            filename: file,
                            size: stats.size,
                            created: stats.birthtime,
                            modified: stats.mtime,
                            sizeFormatted: this.formatBytes(stats.size)
                        };
                    });
                
                const backups = await Promise.all(userBackups);
                
                return {
                    success: true,
                    backups,
                    count: backups.length
                };
            } catch (error) {
                return {
                    success: true,
                    backups: [],
                    count: 0,
                    message: 'No backups directory found'
                };
            }
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

export default SimpleBackupService;