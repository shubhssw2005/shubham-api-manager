import { v4 as uuidv4 } from 'uuid';

/**
 * Universal API v2 - Standalone Version
 * High-performance API without external dependencies
 * 
 * Routes:
 * GET    /api/v2/universal/health               - Health check
 * GET    /api/v2/universal/{table}              - List records
 * GET    /api/v2/universal/{table}/{id}         - Get single record
 * POST   /api/v2/universal/{table}              - Create record
 * PUT    /api/v2/universal/{table}/{id}         - Update record
 * DELETE /api/v2/universal/{table}/{id}         - Soft delete record
 * POST   /api/v2/universal/{table}/{id}/restore - Restore soft deleted record
 * DELETE /api/v2/universal/{table}/{id}/hard    - Hard delete record (permanent)
 * GET    /api/v2/universal/{table}/stats        - Get table statistics
 * POST   /api/v2/universal/{table}/bulk-delete  - Bulk soft delete
 * GET    /api/v2/universal/{table}/search       - Search records
 */

// In-memory data store for demonstration
const dataStore = {
    posts: new Map(),
    products: new Map(),
    customers: new Map(),
    orders: new Map(),
    categories: new Map()
};

// Performance metrics
const performanceMetrics = {
    requests: 0,
    totalTime: 0,
    operations: {
        create: { count: 0, totalTime: 0 },
        read: { count: 0, totalTime: 0 },
        update: { count: 0, totalTime: 0 },
        delete: { count: 0, totalTime: 0 }
    }
};

// Supported tables configuration
const SUPPORTED_TABLES = {
    posts: {
        searchFields: ['title', 'content', 'excerpt'],
        requiredFields: ['title', 'content'],
        softDelete: true
    },
    products: {
        searchFields: ['name', 'description', 'sku'],
        requiredFields: ['name', 'price'],
        softDelete: true
    },
    orders: {
        searchFields: ['order_number'],
        requiredFields: ['customer_id', 'total'],
        softDelete: true
    },
    customers: {
        searchFields: ['email', 'first_name', 'last_name'],
        requiredFields: ['email', 'first_name', 'last_name'],
        softDelete: true
    },
    categories: {
        searchFields: ['name', 'description'],
        requiredFields: ['name'],
        softDelete: true
    }
};

// Initialize with sample data
const initializeSampleData = () => {
    if (dataStore.posts.size === 0) {
        // Add sample posts
        const samplePosts = [
            {
                id: uuidv4(),
                title: "Getting Started with Ultra-Low Latency Systems",
                content: "Building high-performance systems requires careful attention to latency optimization...",
                excerpt: "Learn about ultra-low latency system design",
                status: "published",
                tags: ["performance", "cpp", "optimization"],
                author_id: uuidv4(),
                view_count: 1250,
                like_count: 45,
                featured: true,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                is_deleted: false
            },
            {
                id: uuidv4(),
                title: "ScyllaDB + FoundationDB Integration",
                content: "Learn how to integrate ScyllaDB with FoundationDB for maximum performance...",
                excerpt: "Database integration patterns",
                status: "published",
                tags: ["database", "scylladb", "foundationdb"],
                author_id: uuidv4(),
                view_count: 890,
                like_count: 32,
                featured: false,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                is_deleted: false
            }
        ];
        
        samplePosts.forEach(post => {
            dataStore.posts.set(post.id, post);
        });
        
        // Add sample products
        const sampleProducts = [
            {
                id: uuidv4(),
                name: "High-Performance Server",
                description: "Ultra-fast server for low-latency applications",
                price: 2999.99,
                sku: "HP-SERVER-001",
                status: "active",
                quantity: 50,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                is_deleted: false
            }
        ];
        
        sampleProducts.forEach(product => {
            dataStore.products.set(product.id, product);
        });
    }
};

export default async function handler(req, res) {
    const startTime = Date.now();
    
    try {
        // Initialize sample data
        initializeSampleData();
        
        const { params } = req.query;
        const [tableName, id, action] = params || [];

        // Health check endpoint
        if (tableName === 'health') {
            const health = {
                status: 'healthy',
                timestamp: new Date().toISOString(),
                databases: {
                    scylladb: { status: 'mock', latency_ms: Math.floor(Math.random() * 10) + 1 },
                    foundationdb: { status: 'mock', latency_ms: Math.floor(Math.random() * 5) + 1 }
                },
                performance: {
                    total_requests: performanceMetrics.requests,
                    avg_response_time: performanceMetrics.requests > 0 ? 
                        Math.round(performanceMetrics.totalTime / performanceMetrics.requests) : 0,
                    operations: Object.entries(performanceMetrics.operations).reduce((acc, [op, data]) => {
                        acc[op] = {
                            count: data.count,
                            avg_time_ms: data.count > 0 ? Math.round(data.totalTime / data.count) : 0
                        };
                        return acc;
                    }, {})
                },
                uptime: process.uptime(),
                memory: process.memoryUsage()
            };
            
            return res.status(200).json({
                success: true,
                data: health
            });
        }

        if (!tableName) {
            return res.status(400).json({
                success: false,
                message: 'Table name is required',
                supportedTables: Object.keys(SUPPORTED_TABLES)
            });
        }

        // Validate table name
        if (!SUPPORTED_TABLES[tableName]) {
            return res.status(404).json({
                success: false,
                message: `Table '${tableName}' not supported`,
                supportedTables: Object.keys(SUPPORTED_TABLES)
            });
        }

        // Route to appropriate handler
        let result;
        switch (req.method) {
            case 'GET':
                result = await handleGet(req, res, tableName, id, action);
                break;
            
            case 'POST':
                result = await handlePost(req, res, tableName, id, action);
                break;
            
            case 'PUT':
                result = await handlePut(req, res, tableName, id);
                break;
            
            case 'DELETE':
                result = await handleDelete(req, res, tableName, id, action);
                break;
            
            default:
                res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
                return res.status(405).json({
                    success: false,
                    message: `Method ${req.method} not allowed`
                });
        }

        // Update performance metrics
        const endTime = Date.now();
        const duration = endTime - startTime;
        performanceMetrics.requests++;
        performanceMetrics.totalTime += duration;

        return result;

    } catch (error) {
        console.error('Universal API v2 error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
}

/**
 * Handle GET requests
 */
async function handleGet(req, res, tableName, id, action) {
    const startTime = Date.now();
    
    try {
        const table = dataStore[tableName];
        
        // Statistics endpoint
        if (id === 'stats') {
            const allRecords = Array.from(table.values());
            const active = allRecords.filter(r => !r.is_deleted);
            const deleted = allRecords.filter(r => r.is_deleted);
            
            const stats = {
                tableName,
                total: allRecords.length,
                active: active.length,
                deleted: deleted.length,
                deletionRate: allRecords.length > 0 ? 
                    ((deleted.length / allRecords.length) * 100).toFixed(2) + '%' : '0%'
            };
            
            return res.status(200).json({
                success: true,
                data: stats
            });
        }

        // Search endpoint
        if (id === 'search') {
            const { q: searchTerm, limit = 50, includeDeleted = false } = req.query;
            
            if (!searchTerm) {
                return res.status(400).json({
                    success: false,
                    message: 'Search term (q) is required'
                });
            }

            const tableConfig = SUPPORTED_TABLES[tableName];
            const allRecords = Array.from(table.values());
            
            const results = allRecords.filter(record => {
                if (!includeDeleted && record.is_deleted) return false;
                
                return tableConfig.searchFields.some(field => {
                    const value = record[field];
                    return value && value.toString().toLowerCase().includes(searchTerm.toLowerCase());
                });
            }).slice(0, parseInt(limit));

            return res.status(200).json({
                success: true,
                data: results,
                meta: {
                    searchTerm,
                    total: results.length,
                    searchFields: tableConfig.searchFields
                }
            });
        }

        // Get single record
        if (id) {
            const { includeDeleted = false } = req.query;
            const record = table.get(id);
            
            if (!record || (!includeDeleted && record.is_deleted)) {
                return res.status(404).json({
                    success: false,
                    message: `Record not found: ${id}`
                });
            }

            return res.status(200).json({
                success: true,
                data: record
            });
        }

        // List records
        const {
            limit = 50,
            offset = 0,
            includeDeleted = false,
            ...filters
        } = req.query;

        let records = Array.from(table.values());
        
        // Apply filters
        if (!includeDeleted) {
            records = records.filter(r => !r.is_deleted);
        }
        
        // Apply additional filters
        Object.entries(filters).forEach(([key, value]) => {
            if (value !== undefined) {
                records = records.filter(r => r[key] == value);
            }
        });
        
        // Apply pagination
        const paginatedRecords = records.slice(parseInt(offset), parseInt(offset) + parseInt(limit));

        return res.status(200).json({
            success: true,
            data: paginatedRecords,
            meta: {
                total: records.length,
                limit: parseInt(limit),
                offset: parseInt(offset),
                filters
            }
        });

    } finally {
        const duration = Date.now() - startTime;
        performanceMetrics.operations.read.count++;
        performanceMetrics.operations.read.totalTime += duration;
    }
}

/**
 * Handle POST requests
 */
async function handlePost(req, res, tableName, id, action) {
    const startTime = Date.now();
    
    try {
        const table = dataStore[tableName];
        
        // Restore endpoint
        if (id && action === 'restore') {
            const record = table.get(id);
            if (!record) {
                return res.status(404).json({
                    success: false,
                    message: `Record not found: ${id}`
                });
            }
            
            if (!record.is_deleted) {
                return res.status(400).json({
                    success: false,
                    message: `Record is not deleted: ${id}`
                });
            }
            
            record.is_deleted = false;
            record.deleted_at = null;
            record.updated_at = new Date().toISOString();
            
            table.set(id, record);
            
            return res.status(200).json({
                success: true,
                data: record
            });
        }

        // Bulk delete endpoint
        if (id === 'bulk-delete') {
            const { ids, reason } = req.body;
            
            if (!ids || !Array.isArray(ids) || ids.length === 0) {
                return res.status(400).json({
                    success: false,
                    message: 'Array of IDs is required'
                });
            }

            const results = ids.map(recordId => {
                const record = table.get(recordId);
                if (!record) {
                    return { id: recordId, success: false, error: 'Record not found' };
                }
                
                if (record.is_deleted) {
                    return { id: recordId, success: false, error: 'Already deleted' };
                }
                
                record.is_deleted = true;
                record.deleted_at = new Date().toISOString();
                record.tombstone_reason = reason;
                table.set(recordId, record);
                
                return { id: recordId, success: true };
            });

            return res.status(200).json({
                success: true,
                data: results,
                meta: {
                    processed: results.length,
                    successful: results.filter(r => r.success).length,
                    failed: results.filter(r => !r.success).length
                }
            });
        }

        // Create new record
        if (!id) {
            const tableConfig = SUPPORTED_TABLES[tableName];
            
            // Validate required fields
            for (const field of tableConfig.requiredFields) {
                if (!req.body[field]) {
                    return res.status(400).json({
                        success: false,
                        message: `Required field missing: ${field}`,
                        requiredFields: tableConfig.requiredFields
                    });
                }
            }

            const newRecord = {
                id: uuidv4(),
                ...req.body,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                is_deleted: false
            };

            table.set(newRecord.id, newRecord);

            return res.status(201).json({
                success: true,
                data: newRecord
            });
        }

        return res.status(400).json({
            success: false,
            message: 'Invalid POST route'
        });

    } finally {
        const duration = Date.now() - startTime;
        performanceMetrics.operations.create.count++;
        performanceMetrics.operations.create.totalTime += duration;
    }
}

/**
 * Handle PUT requests
 */
async function handlePut(req, res, tableName, id) {
    const startTime = Date.now();
    
    try {
        if (!id) {
            return res.status(400).json({
                success: false,
                message: 'Record ID is required for updates'
            });
        }

        const table = dataStore[tableName];
        const record = table.get(id);
        
        if (!record) {
            return res.status(404).json({
                success: false,
                message: `Record not found: ${id}`
            });
        }

        const updatedRecord = {
            ...record,
            ...req.body,
            updated_at: new Date().toISOString()
        };

        table.set(id, updatedRecord);

        return res.status(200).json({
            success: true,
            data: updatedRecord
        });

    } finally {
        const duration = Date.now() - startTime;
        performanceMetrics.operations.update.count++;
        performanceMetrics.operations.update.totalTime += duration;
    }
}

/**
 * Handle DELETE requests
 */
async function handleDelete(req, res, tableName, id, action) {
    const startTime = Date.now();
    
    try {
        if (!id) {
            return res.status(400).json({
                success: false,
                message: 'Record ID is required for deletion'
            });
        }

        const table = dataStore[tableName];
        const record = table.get(id);
        
        if (!record) {
            return res.status(404).json({
                success: false,
                message: `Record not found: ${id}`
            });
        }

        // Hard delete (permanent)
        if (action === 'hard') {
            table.delete(id);
            return res.status(200).json({
                success: true,
                message: 'Record permanently deleted'
            });
        }

        // Soft delete (default)
        if (record.is_deleted) {
            return res.status(400).json({
                success: false,
                message: `Record already deleted: ${id}`
            });
        }

        const { reason } = req.body || {};
        record.is_deleted = true;
        record.deleted_at = new Date().toISOString();
        record.tombstone_reason = reason;
        
        table.set(id, record);

        return res.status(200).json({
            success: true,
            message: 'Record soft deleted'
        });

    } finally {
        const duration = Date.now() - startTime;
        performanceMetrics.operations.delete.count++;
        performanceMetrics.operations.delete.totalTime += duration;
    }
}