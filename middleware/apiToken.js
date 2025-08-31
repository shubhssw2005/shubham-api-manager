import APIToken from '../models/APIToken';
import AuditLog from '../models/AuditLog';

/**
 * Middleware to authenticate API requests using API tokens
 * @param {Object} req - Request object
 * @param {Object} res - Response object
 * @param {Function} next - Next middleware function
 */
export async function authenticateAPIToken(req, res, next) {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Missing or invalid authorization header' });
    }

    const token = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    // Find and validate the token
    const apiToken = await APIToken.findByToken(token);
    
    if (!apiToken) {
      // Log failed authentication attempt
      await AuditLog.logAction({
        action: 'login',
        resource: 'api_token',
        userId: null,
        details: { reason: 'invalid_token' },
        ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
        userAgent: req.headers['user-agent'],
        success: false,
        errorMessage: 'Invalid API token'
      });
      
      return res.status(401).json({ error: 'Invalid or expired token' });
    }

    // Check rate limiting
    const rateLimitResult = await checkRateLimit(apiToken, req);
    if (!rateLimitResult.allowed) {
      return res.status(429).json({ 
        error: 'Rate limit exceeded',
        retryAfter: rateLimitResult.retryAfter
      });
    }

    // Record token usage
    await apiToken.recordUsage(
      req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      req.headers['user-agent']
    );

    // Attach token info to request
    req.apiToken = apiToken;
    req.tokenUser = apiToken.createdBy;

    next();
  } catch (error) {
    console.error('API token authentication error:', error);
    return res.status(500).json({ error: 'Authentication failed' });
  }
}

/**
 * Middleware to check if API token has permission for specific action
 * @param {string} modelName - Model name to check permission for
 * @param {string} action - Action to check (create, read, update, delete)
 */
export function requireAPIPermission(modelName, action) {
  return (req, res, next) => {
    if (!req.apiToken) {
      return res.status(401).json({ error: 'API token required' });
    }

    if (!req.apiToken.hasPermission(modelName, action)) {
      // Log unauthorized access attempt
      AuditLog.logAction({
        action: 'unauthorized_access',
        resource: 'api_permission',
        userId: req.apiToken.createdBy._id,
        details: { 
          tokenName: req.apiToken.name,
          attemptedModel: modelName,
          attemptedAction: action
        },
        ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
        userAgent: req.headers['user-agent'],
        success: false,
        errorMessage: `Insufficient permissions for ${action} on ${modelName}`
      });

      return res.status(403).json({ 
        error: `Insufficient permissions for ${action} on ${modelName}` 
      });
    }

    next();
  };
}

/**
 * Check rate limiting for API token
 * @param {Object} apiToken - API token object
 * @param {Object} req - Request object
 * @returns {Object} Rate limit result
 */
async function checkRateLimit(apiToken, req) {
  const { requests, window } = apiToken.rateLimit;
  const windowStart = new Date(Date.now() - window * 1000);
  
  // Count requests in the current window
  const requestCount = await AuditLog.countDocuments({
    'details.tokenId': apiToken._id,
    createdAt: { $gte: windowStart },
    success: true
  });

  if (requestCount >= requests) {
    return {
      allowed: false,
      retryAfter: Math.ceil(window - (Date.now() - windowStart.getTime()) / 1000)
    };
  }

  return { allowed: true };
}

/**
 * Get API token from request headers
 * @param {Object} req - Request object
 * @returns {string|null} Token string or null
 */
export function getAPITokenFromRequest(req) {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }

  return authHeader.substring(7);
}