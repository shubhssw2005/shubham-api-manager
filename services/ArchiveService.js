import mongoose from 'mongoose';
import AWS from 'aws-sdk';

/**
 * Universal Archive Service
 * Handles archiving of soft-deleted documents to cold storage
 */
class ArchiveService {
    constructor() {
        this.s3 = new AWS.S3({
            region: process.env.AWS_REGION || 'us-east-1',
            accessKeyId: process.env.AWS_ACCESS_KEY_ID,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
        });
        this.bucketName = process.env.ARCHIVE_BUCKET_NAME || 'data-archive-bucket';
    }

    /**
     * Archive documents from any collection
     */
    async archiveCollection(modelName, daysOld = 90, batchSize = 1000) {
        const Model = mongoose.model(modelName);
        const ArchiveModel = this.getArchiveModel(modelName);
        
        const cutoffDate = new Date(Date.now() - daysOld * 24 * 60 * 60 * 1000);
        
        console.log(`Starting archive process for ${modelName}...`);
        
        let totalArchived = 0;
        let hasMore = true;
        
        while (hasMore) {
            const session = await mongoose.startSession();
            
            try {
                session.startTransaction();
                
                // Find documents to archive
                const docsToArchive = await Model.find({
                    isDeleted: true,
                    deletedAt: { $lte: cutoffDate }
                })
                .limit(batchSize)
                .lean()
                .session(session);
                
                if (docsToArchive.length === 0) {
                    hasMore = false;
                    break;
                }
                
                // Prepare archive documents
                const archiveDocs = docsToArchive.map(doc => ({
                    originalId: doc._id,
                    ...doc,
                    originalCreatedAt: doc.createdAt,
                    originalUpdatedAt: doc.updatedAt,
                    archivedAt: new Date(),
                    archiveReason: 'retention_policy'
                }));
                
                // Insert into archive collection
                await ArchiveModel.insertMany(archiveDocs, { session });
                
                // Export to S3 (optional)
                if (process.env.ENABLE_S3_ARCHIVE === 'true') {
                    await this.exportToS3(modelName, archiveDocs);
                }
                
                // Hard delete from main collection
                const idsToDelete = docsToArchive.map(doc => doc._id);
                await Model.deleteMany({ _id: { $in: idsToDelete } }, { session });
                
                await session.commitTransaction();
                
                totalArchived += docsToArchive.length;
                console.log(`Archived ${docsToArchive.length} ${modelName} documents`);
                
            } catch (error) {
                await session.abortTransaction();
                console.error(`Archive error for ${modelName}:`, error);
                throw error;
            } finally {
                session.endSession();
            }
        }
        
        console.log(`Archive complete for ${modelName}. Total archived: ${totalArchived}`);
        return totalArchived;
    }

    /**
     * Export archived data to S3 in Parquet format
     */
    async exportToS3(modelName, documents) {
        const date = new Date().toISOString().split('T')[0];
        const key = `archives/${modelName}/dt=${date}/${Date.now()}.json.gz`;
        
        const compressed = await this.compressData(JSON.stringify(documents));
        
        const params = {
            Bucket: this.bucketName,
            Key: key,
            Body: compressed,
            ContentType: 'application/gzip',
            StorageClass: 'GLACIER', // Direct to cold storage
            Metadata: {
                'model': modelName,
                'archive-date': date,
                'document-count': documents.length.toString()
            }
        };
        
        try {
            await this.s3.upload(params).promise();
            console.log(`Exported ${documents.length} ${modelName} documents to S3: ${key}`);
        } catch (error) {
            console.error(`S3 export error for ${modelName}:`, error);
            throw error;
        }
    }

    /**
     * Get or create archive model for any collection
     */
    getArchiveModel(modelName) {
        const archiveModelName = `${modelName}Archive`;
        
        try {
            return mongoose.model(archiveModelName);
        } catch (error) {
            // Create dynamic archive schema
            const originalModel = mongoose.model(modelName);
            const originalSchema = originalModel.schema;
            
            const archiveSchema = new mongoose.Schema({
                originalId: {
                    type: mongoose.Schema.Types.ObjectId,
                    required: true,
                    index: true
                },
                archivedAt: {
                    type: Date,
                    default: Date.now,
                    index: true
                },
                archiveReason: {
                    type: String,
                    default: 'retention_policy'
                }
            });
            
            // Copy all fields from original schema
            archiveSchema.add(originalSchema.obj);
            
            // Add archive-specific indexes
            archiveSchema.index({ archivedAt: 1 });
            archiveSchema.index({ originalId: 1, archivedAt: 1 });
            
            return mongoose.model(archiveModelName, archiveSchema);
        }
    }

    /**
     * Compress data for storage
     */
    async compressData(data) {
        const zlib = await import('zlib');
        return new Promise((resolve, reject) => {
            zlib.gzip(data, (err, compressed) => {
                if (err) reject(err);
                else resolve(compressed);
            });
        });
    }

    /**
     * Schedule archive job for all models
     */
    async scheduleArchiveJobs() {
        const models = mongoose.modelNames().filter(name => 
            !name.endsWith('Archive') && 
            !['Outbox', 'User', 'Role'].includes(name)
        );
        
        const results = {};
        
        for (const modelName of models) {
            try {
                const archived = await this.archiveCollection(modelName);
                results[modelName] = { success: true, archived };
            } catch (error) {
                results[modelName] = { success: false, error: error.message };
            }
        }
        
        return results;
    }

    /**
     * Restore from archive (emergency recovery)
     */
    async restoreFromArchive(modelName, originalId) {
        const ArchiveModel = this.getArchiveModel(modelName);
        const Model = mongoose.model(modelName);
        
        const archivedDoc = await ArchiveModel.findOne({ originalId });
        if (!archivedDoc) {
            throw new Error(`Document ${originalId} not found in ${modelName} archive`);
        }
        
        const restoreData = archivedDoc.toObject();
        delete restoreData._id;
        delete restoreData.archivedAt;
        delete restoreData.archiveReason;
        delete restoreData.originalId;
        
        restoreData._id = originalId;
        restoreData.isDeleted = false;
        restoreData.deletedAt = null;
        restoreData.deletedBy = null;
        restoreData.tombstoneReason = null;
        
        const restoredDoc = await Model.create(restoreData);
        return restoredDoc;
    }
}

export default ArchiveService;