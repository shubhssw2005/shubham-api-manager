#!/usr/bin/env node

import mongoose from 'mongoose';
import ArchiveService from '../services/ArchiveService.js';
import OutboxWorker from '../workers/OutboxWorker.js';
import ModelFactory from '../lib/ModelFactory.js';
import { modelsConfig } from '../config/models.js';

/**
 * Universal Data Management CLI
 * Handles archiving, cleanup, and maintenance across all models
 */

const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/your-database');
        console.log('✅ MongoDB connected');
    } catch (error) {
        console.error('❌ MongoDB connection failed:', error);
        process.exit(1);
    }
};

const initializeModels = () => {
    console.log('🔧 Initializing models...');
    const { models, repositories } = ModelFactory.createModelsFromConfig(modelsConfig);
    console.log(`✅ Initialized ${Object.keys(models).length} models`);
    return { models, repositories };
};

const commands = {
    // Archive old soft-deleted documents
    async archive(modelName = null, daysOld = 90) {
        console.log(`🗄️  Starting archive process (${daysOld} days old)...`);
        const archiveService = new ArchiveService();
        
        if (modelName) {
            const archived = await archiveService.archiveCollection(modelName, daysOld);
            console.log(`✅ Archived ${archived} ${modelName} documents`);
        } else {
            const results = await archiveService.scheduleArchiveJobs();
            console.log('✅ Archive results:', results);
        }
    },

    // Process outbox events
    async processOutbox() {
        console.log('📤 Processing outbox events...');
        const worker = new OutboxWorker();
        await worker.processBatch();
        const stats = await worker.getStats();
        console.log('✅ Outbox stats:', stats);
    },

    // Start outbox worker daemon
    async startWorker() {
        console.log('🚀 Starting outbox worker daemon...');
        const worker = new OutboxWorker();
        
        // Graceful shutdown
        process.on('SIGINT', () => {
            console.log('🛑 Shutting down worker...');
            worker.stop();
            process.exit(0);
        });
        
        await worker.start();
    },

    // Get model statistics
    async stats(modelName = null) {
        console.log('📊 Generating statistics...');
        
        if (modelName) {
            const stats = await ModelFactory.getModelStats(modelName);
            console.log(`✅ ${modelName} stats:`, stats);
        } else {
            const health = await ModelFactory.healthCheck();
            console.log('✅ All models health:', health);
        }
    },

    // Cleanup old outbox events
    async cleanupOutbox(daysOld = 30) {
        console.log(`🧹 Cleaning up outbox events older than ${daysOld} days...`);
        const cutoffDate = new Date(Date.now() - daysOld * 24 * 60 * 60 * 1000);
        
        const Outbox = mongoose.model('Outbox');
        const result = await Outbox.deleteMany({
            processed: true,
            processedAt: { $lte: cutoffDate }
        });
        
        console.log(`✅ Cleaned up ${result.deletedCount} outbox events`);
    },

    // Restore document from archive
    async restore(modelName, originalId) {
        console.log(`🔄 Restoring ${modelName} document ${originalId}...`);
        const archiveService = new ArchiveService();
        
        try {
            const restored = await archiveService.restoreFromArchive(modelName, originalId);
            console.log('✅ Document restored:', restored._id);
        } catch (error) {
            console.error('❌ Restore failed:', error.message);
        }
    },

    // Replay events for debugging
    async replay(aggregate, fromDate, toDate) {
        console.log(`🔄 Replaying events for ${aggregate} from ${fromDate} to ${toDate}...`);
        const worker = new OutboxWorker();
        await worker.replayEvents(aggregate, fromDate, toDate);
        console.log('✅ Replay completed');
    },

    // Create indexes for all models
    async createIndexes() {
        console.log('🔍 Creating indexes for all models...');
        const modelNames = mongoose.modelNames();
        
        for (const modelName of modelNames) {
            try {
                const Model = mongoose.model(modelName);
                await Model.createIndexes();
                console.log(`✅ Created indexes for ${modelName}`);
            } catch (error) {
                console.error(`❌ Failed to create indexes for ${modelName}:`, error.message);
            }
        }
    },

    // Validate data integrity
    async validate(modelName = null) {
        console.log('🔍 Validating data integrity...');
        const modelsToCheck = modelName ? [modelName] : mongoose.modelNames();
        
        for (const name of modelsToCheck) {
            try {
                const Model = mongoose.model(name);
                const issues = [];
                
                // Check for orphaned references
                const docs = await Model.find({}).limit(1000);
                for (const doc of docs) {
                    // Add your validation logic here
                    // Example: check if referenced documents exist
                }
                
                if (issues.length > 0) {
                    console.log(`⚠️  ${name} has ${issues.length} integrity issues`);
                } else {
                    console.log(`✅ ${name} integrity check passed`);
                }
            } catch (error) {
                console.error(`❌ Validation failed for ${name}:`, error.message);
            }
        }
    },

    // Export data for backup
    async export(modelName, outputPath = './backup') {
        console.log(`📦 Exporting ${modelName} data to ${outputPath}...`);
        const fs = await import('fs/promises');
        const path = await import('path');
        
        try {
            await fs.mkdir(outputPath, { recursive: true });
            
            const Model = mongoose.model(modelName);
            const data = await Model.find({}).lean();
            
            const filename = path.join(outputPath, `${modelName}-${Date.now()}.json`);
            await fs.writeFile(filename, JSON.stringify(data, null, 2));
            
            console.log(`✅ Exported ${data.length} ${modelName} documents to ${filename}`);
        } catch (error) {
            console.error(`❌ Export failed:`, error.message);
        }
    },

    // Cleanup expired exports
    async cleanupExports() {
        console.log('🧹 Cleaning up expired data exports...');
        const { default: DataExportService } = await import('../services/DataExportService.js');
        const exportService = new DataExportService();
        
        const cleaned = await exportService.cleanupExpiredExports();
        console.log(`✅ Cleaned up ${cleaned} expired exports`);
    },

    // Show export statistics
    async exportStatus() {
        console.log('📊 Data export statistics...');
        const { default: DataExportJob } = await import('../models/DataExportJob.js');
        
        const stats = await DataExportJob.aggregate([
            {
                $group: {
                    _id: '$status',
                    count: { $sum: 1 },
                    avgFileSize: { $avg: '$fileSize' },
                    totalSize: { $sum: '$fileSize' }
                }
            }
        ]);

        const recentJobs = await DataExportJob.find({})
            .sort({ createdAt: -1 })
            .limit(10)
            .populate('userId', 'name email')
            .select('status progress exportType fileSize createdAt');

        console.log('✅ Export statistics:');
        console.log('Status breakdown:', stats);
        console.log('\nRecent jobs:');
        recentJobs.forEach(job => {
            console.log(`- ${job._id}: ${job.status} (${job.progress}%) - ${job.exportType} - ${job.fileSize ? (job.fileSize / 1024 / 1024).toFixed(2) + 'MB' : 'N/A'}`);
        });
    },

    // Show help
    help() {
        console.log(`
🛠️  Universal Data Management CLI

Usage: node scripts/data-management.js <command> [options]

Commands:
  archive [model] [days]     Archive soft-deleted documents (default: 90 days)
  processOutbox             Process pending outbox events
  startWorker               Start outbox worker daemon
  stats [model]             Show model statistics
  cleanupOutbox [days]      Cleanup old outbox events (default: 30 days)
  restore <model> <id>      Restore document from archive
  replay <aggregate> <from> <to>  Replay events for debugging
  createIndexes             Create indexes for all models
  validate [model]          Validate data integrity
  export <model> [path]     Export model data for backup
  cleanupExports            Cleanup expired data exports
  exportStatus              Show data export statistics
  help                      Show this help message

Examples:
  node scripts/data-management.js archive Post 30
  node scripts/data-management.js stats
  node scripts/data-management.js restore Product 60f1b2c3d4e5f6789abcdef0
  node scripts/data-management.js startWorker
        `);
    }
};

// Main execution
const main = async () => {
    const [,, command, ...args] = process.argv;
    
    if (!command || !commands[command]) {
        commands.help();
        process.exit(1);
    }
    
    try {
        await connectDB();
        initializeModels();
        await commands[command](...args);
    } catch (error) {
        console.error('❌ Command failed:', error);
        process.exit(1);
    } finally {
        await mongoose.disconnect();
        console.log('👋 Disconnected from MongoDB');
    }
};

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export default commands;