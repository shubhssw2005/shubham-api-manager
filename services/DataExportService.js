import mongoose from 'mongoose';
import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';
import AWS from 'aws-sdk';
import DataExportJob from '../models/DataExportJob.js';
import User from '../models/User.js';
import Post from '../models/Post.js';
import Media from '../models/Media.js';
import Outbox from '../models/Outbox.js';

/**
 * Data Export Service
 * Handles user data export to ZIP files with async processing
 */
class DataExportService {
    constructor() {
        this.s3 = new AWS.S3({
            region: process.env.AWS_REGION || 'us-east-1',
            accessKeyId: process.env.AWS_ACCESS_KEY_ID,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
        });
        this.exportsBucket = process.env.EXPORTS_BUCKET_NAME || 'user-data-exports';
        this.tempDir = process.env.TEMP_EXPORT_DIR || '/tmp/exports';
    }

    /**
     * Request a data export for a user
     */
    async requestExport(userId, requestedBy, options = {}) {
        try {
            // Check if there's already a pending/in_progress job
            const existingJob = await DataExportJob.findOne({
                userId,
                status: { $in: ['pending', 'in_progress'] }
            });

            if (existingJob) {
                return {
                    success: false,
                    message: 'Export already in progress',
                    jobId: existingJob._id
                };
            }

            // Create new export job
            const job = await DataExportJob.create({
                userId,
                requestedBy,
                exportType: options.exportType || 'full',
                includeMedia: options.includeMedia !== false,
                includeDeleted: options.includeDeleted || false
            });

            // Queue the job for processing (in production, use SQS/Kafka)
            setImmediate(() => this.processExportJob(job._id));

            return {
                success: true,
                message: 'Export job created successfully',
                jobId: job._id,
                estimatedTime: '5-15 minutes'
            };

        } catch (error) {
            console.error('Error requesting export:', error);
            throw error;
        }
    }

    /**
     * Process the export job asynchronously
     */
    async processExportJob(jobId) {
        let job;
        try {
            job = await DataExportJob.findById(jobId);
            if (!job) {
                console.error(`Export job ${jobId} not found`);
                return;
            }

            // Update status to in_progress
            await DataExportJob.updateOne(
                { _id: jobId },
                { 
                    $set: { 
                        status: 'in_progress',
                        progress: 0
                    }
                }
            );

            console.log(`Starting export job ${jobId} for user ${job.userId}`);

            // Create temp directory for this export
            const exportDir = path.join(this.tempDir, jobId.toString());
            await fs.mkdir(exportDir, { recursive: true });

            // Step 1: Export user profile
            await this.exportUserProfile(job.userId, exportDir);
            await this.updateProgress(jobId, 10);

            // Step 2: Export posts
            const postsCount = await this.exportUserPosts(job.userId, exportDir, job.includeDeleted);
            await this.updateProgress(jobId, 30);

            // Step 3: Export media files
            let mediaCount = 0;
            let totalMediaSize = 0;
            if (job.includeMedia) {
                const mediaResult = await this.exportUserMedia(job.userId, exportDir);
                mediaCount = mediaResult.count;
                totalMediaSize = mediaResult.totalSize;
            }
            await this.updateProgress(jobId, 60);

            // Step 4: Export activity logs (events)
            await this.exportUserActivity(job.userId, exportDir);
            await this.updateProgress(jobId, 80);

            // Step 5: Create ZIP file
            const zipPath = await this.createZipFile(exportDir, jobId.toString());
            const fileStats = await fs.stat(zipPath);
            await this.updateProgress(jobId, 90);

            // Step 6: Upload to S3 and generate signed URL
            const downloadUrl = await this.uploadToS3(zipPath, jobId.toString());
            await this.updateProgress(jobId, 100);

            // Update job with completion details
            await DataExportJob.updateOne(
                { _id: jobId },
                {
                    $set: {
                        status: 'ready',
                        progress: 100,
                        downloadUrl,
                        downloadExpiry: new Date(Date.now() + 48 * 60 * 60 * 1000), // 48 hours
                        fileSize: fileStats.size,
                        metadata: {
                            collections: [
                                { name: 'posts', count: postsCount },
                                { name: 'media', count: mediaCount }
                            ],
                            mediaFiles: {
                                count: mediaCount,
                                totalSize: totalMediaSize
                            },
                            exportedAt: new Date(),
                            zipPath
                        }
                    }
                }
            );

            // Cleanup temp directory
            await fs.rm(exportDir, { recursive: true, force: true });

            console.log(`Export job ${jobId} completed successfully`);

        } catch (error) {
            console.error(`Export job ${jobId} failed:`, error);
            
            if (job) {
                await DataExportJob.updateOne(
                    { _id: jobId },
                    {
                        $set: {
                            status: 'failed',
                            errorMessage: error.message
                        }
                    }
                );
            }
        }
    }

    /**
     * Export user profile data
     */
    async exportUserProfile(userId, exportDir) {
        const user = await User.findById(userId).lean();
        if (!user) {
            throw new Error('User not found');
        }

        // Remove sensitive data
        delete user.password;
        delete user.__v;

        const profileData = {
            exportInfo: {
                exportedAt: new Date().toISOString(),
                exportType: 'user_profile',
                userId: userId.toString()
            },
            profile: user
        };

        await fs.writeFile(
            path.join(exportDir, 'profile.json'),
            JSON.stringify(profileData, null, 2)
        );
    }

    /**
     * Export user posts
     */
    async exportUserPosts(userId, exportDir, includeDeleted = false) {
        const filter = { author: userId };
        if (!includeDeleted) {
            filter.$or = [
                { isDeleted: { $exists: false } },
                { isDeleted: false }
            ];
        }

        const posts = await Post.find(filter)
            .populate('mediaIds', 'filename originalName url mimeType size')
            .lean();

        const postsData = {
            exportInfo: {
                exportedAt: new Date().toISOString(),
                exportType: 'user_posts',
                includeDeleted,
                totalCount: posts.length
            },
            posts: posts.map(post => ({
                ...post,
                _id: post._id.toString(),
                author: post.author.toString(),
                createdAt: post.createdAt,
                updatedAt: post.updatedAt
            }))
        };

        await fs.writeFile(
            path.join(exportDir, 'posts.json'),
            JSON.stringify(postsData, null, 2)
        );

        return posts.length;
    }

    /**
     * Export user media files
     */
    async exportUserMedia(userId, exportDir) {
        const mediaFiles = await Media.find({ uploadedBy: userId }).lean();
        
        if (mediaFiles.length === 0) {
            return { count: 0, totalSize: 0 };
        }

        // Create media directory
        const mediaDir = path.join(exportDir, 'media');
        await fs.mkdir(mediaDir, { recursive: true });

        let totalSize = 0;
        const mediaMetadata = [];

        for (const media of mediaFiles) {
            try {
                // Copy file from uploads directory
                const sourcePath = media.path;
                const fileName = `${media._id}_${media.originalName}`;
                const destPath = path.join(mediaDir, fileName);

                if (await this.fileExists(sourcePath)) {
                    await fs.copyFile(sourcePath, destPath);
                    totalSize += media.size || 0;
                }

                mediaMetadata.push({
                    id: media._id.toString(),
                    originalName: media.originalName,
                    filename: media.filename,
                    mimeType: media.mimeType,
                    size: media.size,
                    uploadedAt: media.createdAt,
                    description: media.description,
                    tags: media.tags
                });

            } catch (error) {
                console.error(`Failed to copy media file ${media._id}:`, error);
            }
        }

        // Write media metadata
        await fs.writeFile(
            path.join(exportDir, 'media_metadata.json'),
            JSON.stringify({
                exportInfo: {
                    exportedAt: new Date().toISOString(),
                    exportType: 'user_media',
                    totalCount: mediaFiles.length,
                    totalSize
                },
                media: mediaMetadata
            }, null, 2)
        );

        return { count: mediaFiles.length, totalSize };
    }

    /**
     * Export user activity logs
     */
    async exportUserActivity(userId, exportDir) {
        // Get events where user was the actor
        const userEvents = await Outbox.find({
            $or: [
                { 'payload.data.author': userId },
                { 'payload.data.createdBy': userId },
                { 'payload.deletedBy': userId }
            ]
        }).lean();

        const activityData = {
            exportInfo: {
                exportedAt: new Date().toISOString(),
                exportType: 'user_activity',
                totalCount: userEvents.length
            },
            events: userEvents.map(event => ({
                id: event._id.toString(),
                aggregate: event.aggregate,
                eventType: event.eventType,
                timestamp: event.createdAt,
                payload: event.payload
            }))
        };

        await fs.writeFile(
            path.join(exportDir, 'activity.json'),
            JSON.stringify(activityData, null, 2)
        );
    }

    /**
     * Create ZIP file from export directory
     */
    async createZipFile(exportDir, jobId) {
        const zipPath = path.join(this.tempDir, `${jobId}.zip`);
        
        try {
            // Use system zip command for better compression
            execSync(`cd "${exportDir}" && zip -r "${zipPath}" .`, { 
                stdio: 'pipe',
                timeout: 300000 // 5 minutes timeout
            });
            
            return zipPath;
        } catch (error) {
            console.error('ZIP creation failed:', error);
            throw new Error('Failed to create ZIP file');
        }
    }

    /**
     * Upload ZIP to S3 and return signed URL (or local path for testing)
     */
    async uploadToS3(zipPath, jobId) {
        try {
            // If S3 is not configured, return local file path for testing
            if (!process.env.AWS_ACCESS_KEY_ID || process.env.ENABLE_S3_ARCHIVE === 'false') {
                console.log('S3 not configured, using local storage for testing');
                
                // Move ZIP to a permanent location
                const permanentPath = path.join(this.tempDir, 'downloads', `${jobId}.zip`);
                await fs.mkdir(path.dirname(permanentPath), { recursive: true });
                await fs.rename(zipPath, permanentPath);
                
                // Return local download URL
                return `/api/data-export/download/${jobId}`;
            }

            const key = `exports/${jobId}.zip`;
            const fileBuffer = await fs.readFile(zipPath);

            await this.s3.upload({
                Bucket: this.exportsBucket,
                Key: key,
                Body: fileBuffer,
                ContentType: 'application/zip',
                Metadata: {
                    'job-id': jobId,
                    'created-at': new Date().toISOString()
                }
            }).promise();

            // Generate signed URL valid for 48 hours
            const signedUrl = this.s3.getSignedUrl('getObject', {
                Bucket: this.exportsBucket,
                Key: key,
                Expires: 48 * 60 * 60 // 48 hours
            });

            // Cleanup local ZIP file
            await fs.unlink(zipPath);

            return signedUrl;

        } catch (error) {
            console.error('S3 upload failed:', error);
            throw new Error('Failed to upload export file');
        }
    }

    /**
     * Get export job status
     */
    async getJobStatus(jobId) {
        const job = await DataExportJob.findById(jobId)
            .populate('userId', 'name email')
            .populate('requestedBy', 'name email');

        if (!job) {
            throw new Error('Export job not found');
        }

        return {
            id: job._id,
            status: job.status,
            progress: job.progress,
            exportType: job.exportType,
            includeMedia: job.includeMedia,
            downloadUrl: job.downloadUrl,
            downloadExpiry: job.downloadExpiry,
            fileSize: job.fileSize,
            errorMessage: job.errorMessage,
            metadata: job.metadata,
            createdAt: job.createdAt,
            updatedAt: job.updatedAt
        };
    }

    /**
     * Get user's export history
     */
    async getUserExportHistory(userId, limit = 10) {
        const jobs = await DataExportJob.find({ userId })
            .sort({ createdAt: -1 })
            .limit(limit)
            .select('-metadata -errorMessage');

        return jobs;
    }

    /**
     * Helper methods
     */
    async updateProgress(jobId, progress) {
        await DataExportJob.updateOne(
            { _id: jobId },
            { $set: { progress } }
        );
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Cleanup expired exports
     */
    async cleanupExpiredExports() {
        const expiredJobs = await DataExportJob.find({
            status: 'ready',
            downloadExpiry: { $lt: new Date() }
        });

        for (const job of expiredJobs) {
            try {
                // Delete from S3
                if (job.metadata?.zipPath) {
                    const key = `exports/${job._id}.zip`;
                    await this.s3.deleteObject({
                        Bucket: this.exportsBucket,
                        Key: key
                    }).promise();
                }

                // Update job status
                await DataExportJob.updateOne(
                    { _id: job._id },
                    { $set: { status: 'expired', downloadUrl: null } }
                );

            } catch (error) {
                console.error(`Failed to cleanup export ${job._id}:`, error);
            }
        }

        return expiredJobs.length;
    }
}

export default DataExportService;