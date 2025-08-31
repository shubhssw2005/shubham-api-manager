import dbConnect from '../../lib/dbConnect.js';
import { requireApprovedUser } from '../../middleware/auth.js';
import MinIOBackupService from '../../services/MinIOBackupService.js';

/**
 * Blog Backup API
 * Creates zip backup of user's blog data and stores in MinIO/S3
 */
export default async function handler(req, res) {
  try {
    // Initialize database connections (Scylla/Foundation) if needed elsewhere
    await dbConnect();

    const user = await requireApprovedUser(req, res);
    if (!user) return;

    switch (req.method) {
      case 'POST':
        return await createBackup(req, res, user);
      case 'GET':
        return await listBackups(req, res, user);
      default:
        return res.status(405).json({
          success: false,
          message: `Method ${req.method} not allowed`,
        });
    }
  } catch (error) {
    console.error('Backup API error:', error);
    return res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message,
    });
  }
}

async function createBackup(req, res, user) {
  try {
    console.log(`🚀 Starting backup for user: ${user.email}`);

    // Create backup using MinIO service
    const backupService = new MinIOBackupService();
    const backupResult = await backupService.backupUserBlogData(
      user._id,
      user.email
    );

    // Return success (logging to DB intentionally skipped to remove mongoose dependency)
    return res.status(201).json({
      success: true,
      message: 'Blog backup created successfully',
      data: {
        backupId: backupResult.objectName || backupResult.s3Key,
        downloadUrl: backupResult.downloadUrl,
        size: backupResult.metadata?.totalSize ?? backupResult.size,
        postsIncluded: backupResult.metadata?.postsCount ?? 0,
        mediaFilesIncluded: backupResult.metadata?.mediaCount ?? 0,
        bucket: backupResult.bucket,
        etag: backupResult.etag,
        createdAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 3600000).toISOString(), // 1 hour
        storageType: 'MinIO',
      },
    });
  } catch (error) {
    console.error('Backup creation failed:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to create backup',
      error: error.message,
    });
  }
}

async function listBackups(req, res, user) {
  try {
    // Use the same MinIO service to list backups
    const backupService = new MinIOBackupService();
    const backupsResult = await backupService.listUserBackups(user._id);

    return res.status(200).json({
      success: true,
      data: {
        backups: backupsResult.backups,
        totalBackups: backupsResult.count || 0,
        user: {
          id: user._id,
          email: user.email,
        },
      },
    });
  } catch (error) {
    console.error('List backups failed:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to list backups',
      error: error.message,
    });
  }
}