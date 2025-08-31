import mongoose from 'mongoose';
import BaseRepository from '../lib/BaseRepository.js';
import ModelFactory from '../lib/ModelFactory.js';

/**
 * Universal Data Management Middleware
 * Provides consistent CRUD operations across all models
 */

export const createUniversalCRUD = (modelName) => {
    return {
        // GET /api/{model} - List with pagination, filtering, search
        async list(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const {
                    page = 1,
                    limit = 20,
                    sort = '-createdAt',
                    search,
                    ...filters
                } = req.query;

                // Remove empty filters
                Object.keys(filters).forEach(key => {
                    if (filters[key] === '' || filters[key] === undefined) {
                        delete filters[key];
                    }
                });

                let result;
                if (search) {
                    result = await repository.search(search, filters, {
                        page: parseInt(page),
                        limit: parseInt(limit),
                        sort: parseSortString(sort)
                    });
                } else {
                    result = await repository.findWithPagination(filters, {
                        page: parseInt(page),
                        limit: parseInt(limit),
                        sort: parseSortString(sort)
                    });
                }

                res.status(200).json({
                    success: true,
                    data: result.docs,
                    pagination: {
                        page: result.page,
                        pages: result.pages,
                        total: result.total,
                        hasNext: result.hasNext,
                        hasPrev: result.hasPrev
                    }
                });
            } catch (error) {
                console.error(`Error listing ${modelName}:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error retrieving ${modelName.toLowerCase()}s`,
                    error: error.message
                });
            }
        },

        // GET /api/{model}/{id} - Get single document
        async getById(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const doc = await repository.findActiveById(req.params.id);
                
                if (!doc) {
                    return res.status(404).json({
                        success: false,
                        message: `${modelName} not found`
                    });
                }

                res.status(200).json({
                    success: true,
                    data: doc
                });
            } catch (error) {
                console.error(`Error getting ${modelName}:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error retrieving ${modelName.toLowerCase()}`,
                    error: error.message
                });
            }
        },

        // POST /api/{model} - Create new document
        async create(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const userId = req.user?.id || req.user?._id;
                const doc = await repository.create(req.body, userId);

                res.status(201).json({
                    success: true,
                    data: doc,
                    message: `${modelName} created successfully`
                });
            } catch (error) {
                console.error(`Error creating ${modelName}:`, error);
                
                if (error.name === 'ValidationError') {
                    return res.status(400).json({
                        success: false,
                        message: 'Validation error',
                        errors: Object.values(error.errors).map(e => e.message)
                    });
                }

                if (error.code === 11000) {
                    return res.status(409).json({
                        success: false,
                        message: 'Duplicate entry',
                        error: 'A record with this information already exists'
                    });
                }

                res.status(500).json({
                    success: false,
                    message: `Error creating ${modelName.toLowerCase()}`,
                    error: error.message
                });
            }
        },

        // PUT /api/{model}/{id} - Update document
        async update(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const userId = req.user?.id || req.user?._id;
                const doc = await repository.updateById(req.params.id, req.body, userId);

                if (!doc) {
                    return res.status(404).json({
                        success: false,
                        message: `${modelName} not found`
                    });
                }

                res.status(200).json({
                    success: true,
                    data: doc,
                    message: `${modelName} updated successfully`
                });
            } catch (error) {
                console.error(`Error updating ${modelName}:`, error);
                
                if (error.name === 'ValidationError') {
                    return res.status(400).json({
                        success: false,
                        message: 'Validation error',
                        errors: Object.values(error.errors).map(e => e.message)
                    });
                }

                res.status(500).json({
                    success: false,
                    message: `Error updating ${modelName.toLowerCase()}`,
                    error: error.message
                });
            }
        },

        // DELETE /api/{model}/{id} - Soft delete document
        async softDelete(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const userId = req.user?.id || req.user?._id;
                const reason = req.body?.reason || 'user_request';
                
                const result = await repository.softDelete(req.params.id, userId, reason);

                if (result.matchedCount === 0) {
                    return res.status(404).json({
                        success: false,
                        message: `${modelName} not found`
                    });
                }

                res.status(200).json({
                    success: true,
                    message: `${modelName} deleted successfully`
                });
            } catch (error) {
                console.error(`Error deleting ${modelName}:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error deleting ${modelName.toLowerCase()}`,
                    error: error.message
                });
            }
        },

        // POST /api/{model}/{id}/restore - Restore soft deleted document
        async restore(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const result = await repository.restore(req.params.id);

                if (result.matchedCount === 0) {
                    return res.status(404).json({
                        success: false,
                        message: `${modelName} not found in deleted items`
                    });
                }

                res.status(200).json({
                    success: true,
                    message: `${modelName} restored successfully`
                });
            } catch (error) {
                console.error(`Error restoring ${modelName}:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error restoring ${modelName.toLowerCase()}`,
                    error: error.message
                });
            }
        },

        // GET /api/{model}/stats - Get model statistics
        async getStats(req, res) {
            try {
                const stats = await ModelFactory.getModelStats(modelName);
                
                res.status(200).json({
                    success: true,
                    data: stats
                });
            } catch (error) {
                console.error(`Error getting ${modelName} stats:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error retrieving ${modelName.toLowerCase()} statistics`,
                    error: error.message
                });
            }
        },

        // POST /api/{model}/bulk-delete - Bulk soft delete
        async bulkDelete(req, res) {
            try {
                const Model = mongoose.model(modelName);
                const repository = new BaseRepository(Model);
                
                const { ids, filter, reason = 'bulk_operation' } = req.body;
                const userId = req.user?.id || req.user?._id;

                let deleteFilter = {};
                if (ids && Array.isArray(ids)) {
                    deleteFilter._id = { $in: ids };
                } else if (filter) {
                    deleteFilter = filter;
                } else {
                    return res.status(400).json({
                        success: false,
                        message: 'Either ids array or filter object is required'
                    });
                }

                const result = await repository.bulkSoftDelete(deleteFilter, userId, reason);

                res.status(200).json({
                    success: true,
                    data: {
                        deletedCount: result.modifiedCount
                    },
                    message: `${result.modifiedCount} ${modelName.toLowerCase()}s deleted successfully`
                });
            } catch (error) {
                console.error(`Error bulk deleting ${modelName}:`, error);
                res.status(500).json({
                    success: false,
                    message: `Error bulk deleting ${modelName.toLowerCase()}s`,
                    error: error.message
                });
            }
        }
    };
};

// Helper function to parse sort string
const parseSortString = (sortStr) => {
    if (typeof sortStr !== 'string') return sortStr;
    
    const sortObj = {};
    const fields = sortStr.split(',');
    
    fields.forEach(field => {
        const trimmed = field.trim();
        if (trimmed.startsWith('-')) {
            sortObj[trimmed.substring(1)] = -1;
        } else {
            sortObj[trimmed] = 1;
        }
    });
    
    return sortObj;
};

export default createUniversalCRUD;