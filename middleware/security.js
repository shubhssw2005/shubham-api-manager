import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import validator from 'validator';

// Rate limiting configuration
export const createRateLimit = (windowMs = 15 * 60 * 1000, max = 100) => {
    return rateLimit({
        windowMs,
        max,
        message: {
            success: false,
            message: 'Too many requests from this IP, please try again later.',
            retryAfter: Math.ceil(windowMs / 1000)
        },
        standardHeaders: true,
        legacyHeaders: false,
        handler: (req, res) => {
            res.status(429).json({
                success: false,
                message: 'Rate limit exceeded',
                retryAfter: Math.ceil(windowMs / 1000)
            });
        }
    });
};

// Input sanitization and validation
export const sanitizeInput = (input) => {
    if (typeof input === 'string') {
        // Remove potential SQL injection patterns
        const sqlPatterns = [
            /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/gi,
            /(--|\/\*|\*\/|;|'|"|`)/g,
            /(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+/gi
        ];
        
        let sanitized = input;
        sqlPatterns.forEach(pattern => {
            sanitized = sanitized.replace(pattern, '');
        });
        
        // HTML encode special characters
        sanitized = validator.escape(sanitized);
        
        return sanitized.trim();
    }
    
    if (typeof input === 'object' && input !== null) {
        const sanitized = {};
        for (const [key, value] of Object.entries(input)) {
            sanitized[sanitizeInput(key)] = sanitizeInput(value);
        }
        return sanitized;
    }
    
    return input;
};

// Validate common input types
export const validateInput = {
    email: (email) => validator.isEmail(email),
    uuid: (id) => validator.isUUID(id),
    alphanumeric: (str) => validator.isAlphanumeric(str),
    length: (str, min = 1, max = 1000) => validator.isLength(str, { min, max }),
    url: (url) => validator.isURL(url),
    json: (str) => {
        try {
            JSON.parse(str);
            return true;
        } catch {
            return false;
        }
    }
};

// Security headers middleware
export const securityHeaders = helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"],
            fontSrc: ["'self'"],
            objectSrc: ["'none'"],
            mediaSrc: ["'self'"],
            frameSrc: ["'none'"],
        },
    },
    crossOriginEmbedderPolicy: false,
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
});

// Request validation middleware
export const validateRequest = (req, res, next) => {
    try {
        // Sanitize query parameters
        if (req.query) {
            req.query = sanitizeInput(req.query);
        }
        
        // Sanitize body
        if (req.body) {
            req.body = sanitizeInput(req.body);
        }
        
        // Validate common patterns
        if (req.query.q && req.query.q.length > 0) {
            const query = req.query.q;
            
            // Check for SQL injection patterns
            const dangerousPatterns = [
                /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/gi,
                /(--|\/\*|\*\/)/g,
                /(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+/gi,
                /('|"|`).*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)/gi
            ];
            
            for (const pattern of dangerousPatterns) {
                if (pattern.test(query)) {
                    return res.status(400).json({
                        success: false,
                        message: 'Invalid query parameters detected',
                        code: 'INVALID_INPUT'
                    });
                }
            }
        }
        
        next();
    } catch (error) {
        console.error('Request validation error:', error);
        res.status(400).json({
            success: false,
            message: 'Request validation failed',
            code: 'VALIDATION_ERROR'
        });
    }
};

// CORS configuration for production
export const corsOptions = {
    origin: process.env.NODE_ENV === 'production' 
        ? process.env.ALLOWED_ORIGINS?.split(',') || ['https://yourdomain.com']
        : true,
    credentials: true,
    optionsSuccessStatus: 200,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

export default {
    createRateLimit,
    sanitizeInput,
    validateInput,
    securityHeaders,
    validateRequest,
    corsOptions
};