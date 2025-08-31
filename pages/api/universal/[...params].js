import dbConnect from '../../../lib/dbConnect.js';
import { requireApprovedUser } from '../../../middleware/scyllaAuth.js';
import { createUniversalCRUD } from '../../../middleware/scyllaDataManagement.js';
import { validateRequest, sanitizeInput, validateInput } from '../../../middleware/security.js';

/**
 * Universal API Route Handler for ScyllaDB + FoundationDB
 * Handles CRUD operations for any table
 * 
 * Routes:
 * GET    /api/universal/{table}           - List documents
 * GET    /api/universal/{table}/{id}      - Get single document
 * POST   /api/universal/{table}           - Create document
 * PUT    /api/universal/{table}/{id}      - Update document
 * DELETE /api/universal/{table}/{id}      - Soft delete document
 * POST   /api/universal/{table}/{id}/restore - Restore document
 * GET    /api/universal/{table}/stats     - Get statistics
 * POST   /api/universal/{table}/bulk-delete - Bulk delete
 */

// Supported tables
const SUPPORTED_TABLES = ['posts', 'users', 'media', 'sessions'];

const validateTable = (tableName) => {
    return SUPPORTED_TABLES.includes(tableName.toLowerCase());
};

export default async function handler(req, res) {
    try {
        // Apply security validation
        validateRequest(req, res, () => {});
        
        await dbConnect();

        const { params } = req.query;
        
        // Handle Next.js catch-all routes - params comes as an object with numeric keys
        let paramsArray = [];
        if (Array.isArray(params)) {
            paramsArray = params;
        } else if (params && typeof params === 'object') {
            // Convert object with numeric keys to array
            paramsArray = Object.keys(params).sort((a, b) => parseInt(a) - parseInt(b)).map(key => params[key]);
        } else if (params) {
            paramsArray = [params];
        }
        
        const [tableName, id, action] = paramsArray;

        if (!tableName) {
            return res.status(400).json({
                success: false,
                message: 'Table name is required'
            });
        }

        // Validate table name
        if (typeof tableName !== 'string' || !/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(tableName)) {
            return res.status(400).json({
                success: false,
                message: `Invalid table name format: ${tableName}`
            });
        }

        // Check if table is supported
        if (!validateTable(tableName)) {
            return res.status(404).json({
                success: false,
                message: `Table '${tableName}' not supported. Supported tables: ${SUPPORTED_TABLES.join(', ')}`
            });
        }

        // Validate ID if provided
        if (id && id !== 'stats' && id !== 'bulk-delete' && !validateInput.uuid(id)) {
            return res.status(400).json({
                success: false,
                message: 'Invalid ID format'
            });
        }

        // Get CRUD operations for this table
        const crud = createUniversalCRUD(tableName);

        // Authentication check (except for GET requests in development)
        if (req.method !== 'GET' || process.env.NODE_ENV === 'production') {
            const user = await requireApprovedUser(req, res);
            if (!user) return; // Auth middleware handles the response
            req.user = user;
        }

        // Route to appropriate handler
        switch (req.method) {
            case 'GET':
                if (id === 'stats') {
                    return await crud.getStats(req, res);
                } else if (id) {
                    return await crud.getById(req, res);
                } else {
                    return await crud.list(req, res);
                }

            case 'POST':
                if (id && action === 'restore') {
                    return await crud.restore(req, res);
                } else if (id === 'bulk-delete') {
                    return await crud.bulkDelete(req, res);
                } else if (!id) {
                    return await crud.create(req, res);
                } else {
                    return res.status(400).json({
                        success: false,
                        message: 'Invalid POST route'
                    });
                }

            case 'PUT':
                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required for updates'
                    });
                }
                return await crud.update(req, res);

            case 'DELETE':
                if (!id) {
                    return res.status(400).json({
                        success: false,
                        message: 'Document ID is required for deletion'
                    });
                }
                return await crud.softDelete(req, res);

            default:
                res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
                return res.status(405).json({
                    success: false,
                    message: `Method ${req.method} not allowed`
                });
        }

    } catch (error) {
        console.error('Universal API error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
}

// API Documentation endpoint
export const getApiDocumentation = () => {
    return {
        title: 'Universal API Documentation',
        description: 'ScyllaDB + FoundationDB CRUD API',
        database: 'ScyllaDB (Cassandra) + FoundationDB',
        tables: SUPPORTED_TABLES.map(table => ({
            name: table,
            endpoints: {
                list: `GET /api/universal/${table}`,
                get: `GET /api/universal/${table}/{id}`,
                create: `POST /api/universal/${table}`,
                update: `PUT /api/universal/${table}/{id}`,
                delete: `DELETE /api/universal/${table}/{id}`,
                restore: `POST /api/universal/${table}/{id}/restore`,
                stats: `GET /api/universal/${table}/stats`,
                bulkDelete: `POST /api/universal/${table}/bulk-delete`
            }
        }))
    };
};