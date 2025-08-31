import { v4 as uuidv4 } from 'uuid';
import { 
    createDocument, 
    updateDocument, 
    deleteDocument, 
    findDocumentById, 
    findDocuments,
    getScyllaDB 
} from '../lib/dbConnect.js';

// Helper function to parse Next.js catch-all route params
function parseParams(params) {
    if (Array.isArray(params)) {
        return params;
    } else if (params && typeof params === 'object') {
        return Object.keys(params).sort((a, b) => parseInt(a) - parseInt(b)).map(key => params[key]);
    } else if (params) {
        return [params];
    }
    return [];
}

/**
 * Create universal CRUD operations for ScyllaDB
 */
export function createUniversalCRUD(tableName) {
    return {
        // Create a new document
        async create(req, res) {
            try {
                const data = req.body;
                
                // Validate required fields
                if (!data || typeof data !== 'object') {
                    return res.status(400).json({
                        success: false,
                        message: 'Request body is required and must be an object'
                    });
                }

                // Generate ID if not provided
                if (!data.id) {
                    data.id = uuidv4();
                }

                const document = await createDocument(tableName.toLowerCase(), data);

                return res.status(201).json({
                    success: true,
                    data: document,
                    message: `${tableName} created successfully`
                });
            } catch (error) {
                console.error(`Create ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to create ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Get document by ID
        async getById(req, res) {
            try {
                const { params } = req.query;
                const paramsArray = parseParams(params);
                const id = paramsArray[1]; // Second parameter is the ID

                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required'
                    });
                }

                const document = await findDocumentById(tableName.toLowerCase(), id);

                if (!document || document.is_deleted) {
                    return res.status(404).json({
                        success: false,
                        message: `${tableName} not found`
                    });
                }

                return res.status(200).json({
                    success: true,
                    data: document
                });
            } catch (error) {
                console.error(`Get ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to retrieve ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // List documents with pagination
        async list(req, res) {
            try {
                const { limit = 50, pageState } = req.query;
                const parsedLimit = Math.min(parseInt(limit) || 50, 100); // Max 100 items

                const result = await findDocuments(tableName.toLowerCase(), parsedLimit, pageState);

                return res.status(200).json({
                    success: true,
                    data: result.rows,
                    pagination: {
                        limit: parsedLimit,
                        pageState: result.pageState,
                        hasMore: !!result.pageState
                    }
                });
            } catch (error) {
                console.error(`List ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to list ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Update document
        async update(req, res) {
            try {
                const { params } = req.query;
                const paramsArray = parseParams(params);
                const id = paramsArray[1]; // Second parameter is the ID
                const data = req.body;

                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required'
                    });
                }

                if (!data || typeof data !== 'object') {
                    return res.status(400).json({
                        success: false,
                        message: 'Request body is required and must be an object'
                    });
                }

                // Check if document exists
                const existing = await findDocumentById(tableName.toLowerCase(), id);
                if (!existing || existing.is_deleted) {
                    return res.status(404).json({
                        success: false,
                        message: `${tableName} not found`
                    });
                }

                // Remove fields that shouldn't be updated
                const { id: _, created_at, ...updateData } = data;

                const updatedDocument = await updateDocument(tableName.toLowerCase(), id, updateData);

                return res.status(200).json({
                    success: true,
                    data: updatedDocument,
                    message: `${tableName} updated successfully`
                });
            } catch (error) {
                console.error(`Update ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to update ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Soft delete document
        async softDelete(req, res) {
            try {
                const { params } = req.query;
                const paramsArray = parseParams(params);
                const id = paramsArray[1]; // Second parameter is the ID

                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required'
                    });
                }

                // Check if document exists
                const existing = await findDocumentById(tableName.toLowerCase(), id);
                if (!existing || existing.is_deleted) {
                    return res.status(404).json({
                        success: false,
                        message: `${tableName} not found`
                    });
                }

                await deleteDocument(tableName.toLowerCase(), id);

                return res.status(200).json({
                    success: true,
                    message: `${tableName} deleted successfully`
                });
            } catch (error) {
                console.error(`Delete ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to delete ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Get statistics
        async getStats(req, res) {
            try {
                const scylla = await getScyllaDB();
                
                // Get basic count (this is a simplified version)
                const countQuery = `SELECT COUNT(*) as total FROM ${scylla.config.keyspace}.${tableName.toLowerCase()} WHERE is_deleted = false`;
                const result = await scylla.execute(countQuery);
                const total = result.rows[0]?.total || 0;

                const stats = {
                    total: parseInt(total),
                    table: tableName.toLowerCase(),
                    keyspace: scylla.config.keyspace,
                    timestamp: new Date().toISOString()
                };

                return res.status(200).json({
                    success: true,
                    data: stats
                });
            } catch (error) {
                console.error(`Get ${tableName} stats error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to get ${tableName} statistics`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Restore soft-deleted document
        async restore(req, res) {
            try {
                const { params } = req.query;
                const paramsArray = parseParams(params);
                const id = paramsArray[1]; // Second parameter is the ID

                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required'
                    });
                }

                const restored = await updateDocument(tableName.toLowerCase(), id, { 
                    is_deleted: false 
                });

                return res.status(200).json({
                    success: true,
                    data: restored,
                    message: `${tableName} restored successfully`
                });
            } catch (error) {
                console.error(`Restore ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to restore ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        },

        // Bulk delete (for admin operations)
        async bulkDelete(req, res) {
            try {
                const { ids } = req.body;

                if (!Array.isArray(ids) || ids.length === 0) {
                    return res.status(400).json({
                        success: false,
                        message: 'Array of IDs is required'
                    });
                }

                const results = [];
                for (const id of ids) {
                    try {
                        await deleteDocument(tableName.toLowerCase(), id);
                        results.push({ id, success: true });
                    } catch (error) {
                        results.push({ id, success: false, error: error.message });
                    }
                }

                const successCount = results.filter(r => r.success).length;

                return res.status(200).json({
                    success: true,
                    data: {
                        total: ids.length,
                        successful: successCount,
                        failed: ids.length - successCount,
                        results
                    },
                    message: `Bulk delete completed: ${successCount}/${ids.length} successful`
                });
            } catch (error) {
                console.error(`Bulk delete ${tableName} error:`, error);
                return res.status(500).json({
                    success: false,
                    message: `Failed to bulk delete ${tableName}`,
                    error: process.env.NODE_ENV === 'development' ? error.message : undefined
                });
            }
        }
    };
}

export default createUniversalCRUD;