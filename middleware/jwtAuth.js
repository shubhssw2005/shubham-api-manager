import TokenService from '../lib/auth/TokenService.js';
import User from '../models/User.js';
import dbConnect from '../lib/dbConnect.js';

const tokenService = new TokenService();

/**
 * JWT Authentication middleware with tenant context injection
 */
export async function jwtAuth(req, res, next) {
  try {
    const token = tokenService.extractTokenFromRequest(req);
    
    if (!token) {
      return res.status(401).json({ 
        error: 'Access token required',
        code: 'MISSING_TOKEN'
      });
    }

    // Verify access token
    const decoded = await tokenService.verifyAccessToken(token);
    
    // Connect to database
    await dbConnect();
    
    // Get user details
    const user = await User.findById(decoded.userId).select('-password');
    if (!user) {
      return res.status(401).json({ 
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    // Check if user is active
    if (user.status !== 'active') {
      return res.status(403).json({ 
        error: user.status === 'pending' ? 'Account pending approval' : 'Account has been deactivated',
        code: 'ACCOUNT_INACTIVE'
      });
    }

    // Inject user and tenant context into request
    req.user = {
      id: user._id,
      email: user.email,
      name: user.name,
      role: user.role,
      status: user.status,
      permissions: decoded.permissions,
      tenantId: decoded.tenantId,
      isGrootUser: user.isGrootUser
    };

    // Inject tenant context if available
    if (decoded.tenantId) {
      req.tenant = {
        id: decoded.tenantId
      };
    }

    // Update last login time
    user.lastLoginAt = new Date();
    await user.save();

    next();
  } catch (error) {
    console.error('JWT Auth error:', error);
    
    if (error.message === 'Access token expired') {
      return res.status(401).json({ 
        error: 'Access token expired',
        code: 'TOKEN_EXPIRED'
      });
    } else if (error.message === 'Invalid access token') {
      return res.status(401).json({ 
        error: 'Invalid access token',
        code: 'INVALID_TOKEN'
      });
    } else if (error.message === 'Token has been revoked') {
      return res.status(401).json({ 
        error: 'Token has been revoked',
        code: 'TOKEN_REVOKED'
      });
    }

    return res.status(500).json({ 
      error: 'Authentication error',
      code: 'AUTH_ERROR'
    });
  }
}

/**
 * Optional JWT Authentication - doesn't fail if no token provided
 */
export async function optionalJwtAuth(req, res, next) {
  try {
    const token = tokenService.extractTokenFromRequest(req);
    
    if (!token) {
      // No token provided, continue without authentication
      req.user = null;
      req.tenant = null;
      return next();
    }

    // Try to authenticate, but don't fail if token is invalid
    try {
      const decoded = await tokenService.verifyAccessToken(token);
      
      await dbConnect();
      const user = await User.findById(decoded.userId).select('-password');
      
      if (user && user.status === 'active') {
        req.user = {
          id: user._id,
          email: user.email,
          name: user.name,
          role: user.role,
          status: user.status,
          permissions: decoded.permissions,
          tenantId: decoded.tenantId,
          isGrootUser: user.isGrootUser
        };

        if (decoded.tenantId) {
          req.tenant = {
            id: decoded.tenantId
          };
        }
      }
    } catch (authError) {
      // Authentication failed, but continue without user context
      req.user = null;
      req.tenant = null;
    }

    next();
  } catch (error) {
    console.error('Optional JWT Auth error:', error);
    req.user = null;
    req.tenant = null;
    next();
  }
}

/**
 * Refresh token middleware
 */
export async function refreshTokenAuth(req, res, next) {
  try {
    const refreshToken = req.body.refreshToken || req.cookies?.refreshToken;
    
    if (!refreshToken) {
      return res.status(401).json({ 
        error: 'Refresh token required',
        code: 'MISSING_REFRESH_TOKEN'
      });
    }

    // Verify refresh token
    const decoded = await tokenService.verifyRefreshToken(refreshToken);
    
    // Connect to database and get user
    await dbConnect();
    const user = await User.findById(decoded.userId).select('-password');
    
    if (!user) {
      return res.status(401).json({ 
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    if (user.status !== 'active') {
      return res.status(403).json({ 
        error: 'Account is not active',
        code: 'ACCOUNT_INACTIVE'
      });
    }

    // Inject user context
    req.user = user;
    req.refreshTokenData = decoded;

    next();
  } catch (error) {
    console.error('Refresh token auth error:', error);
    
    if (error.message === 'Refresh token expired') {
      return res.status(401).json({ 
        error: 'Refresh token expired',
        code: 'REFRESH_TOKEN_EXPIRED'
      });
    } else if (error.message === 'Invalid refresh token') {
      return res.status(401).json({ 
        error: 'Invalid refresh token',
        code: 'INVALID_REFRESH_TOKEN'
      });
    } else if (error.message === 'Refresh token not found or expired') {
      return res.status(401).json({ 
        error: 'Refresh token not found',
        code: 'REFRESH_TOKEN_NOT_FOUND'
      });
    }

    return res.status(500).json({ 
      error: 'Refresh token authentication error',
      code: 'REFRESH_AUTH_ERROR'
    });
  }
}

/**
 * Tenant isolation middleware - ensures user can only access their tenant's data
 */
export function tenantIsolation(req, res, next) {
  try {
    // Skip tenant isolation for superadmin
    if (req.user?.role === 'superadmin') {
      return next();
    }

    // Extract tenant ID from request (URL params, query, or body)
    const requestTenantId = req.params.tenantId || req.query.tenantId || req.body.tenantId;
    
    // If tenant ID is specified in request, verify it matches user's tenant
    if (requestTenantId && req.user?.tenantId && requestTenantId !== req.user.tenantId) {
      return res.status(403).json({ 
        error: 'Access denied: Tenant isolation violation',
        code: 'TENANT_ACCESS_DENIED'
      });
    }

    // Inject tenant filter for database queries
    if (req.user?.tenantId) {
      req.tenantFilter = { tenantId: req.user.tenantId };
    }

    next();
  } catch (error) {
    console.error('Tenant isolation error:', error);
    return res.status(500).json({ 
      error: 'Tenant isolation error',
      code: 'TENANT_ISOLATION_ERROR'
    });
  }
}

/**
 * Rate limiting per tenant
 */
export function tenantRateLimit(options = {}) {
  const { maxRequests = 1000, windowMs = 15 * 60 * 1000 } = options; // 1000 requests per 15 minutes
  
  return async (req, res, next) => {
    try {
      if (!req.user?.tenantId) {
        return next(); // Skip rate limiting if no tenant context
      }

      const key = `rate_limit:${req.user.tenantId}:${Math.floor(Date.now() / windowMs)}`;
      
      // Use Redis to track requests per tenant
      const current = await tokenService.redis.incr(key);
      
      if (current === 1) {
        await tokenService.redis.expire(key, Math.ceil(windowMs / 1000));
      }

      if (current > maxRequests) {
        return res.status(429).json({ 
          error: 'Rate limit exceeded for tenant',
          code: 'TENANT_RATE_LIMIT_EXCEEDED',
          retryAfter: Math.ceil(windowMs / 1000)
        });
      }

      // Add rate limit headers
      res.set({
        'X-RateLimit-Limit': maxRequests,
        'X-RateLimit-Remaining': Math.max(0, maxRequests - current),
        'X-RateLimit-Reset': new Date(Date.now() + windowMs).toISOString()
      });

      next();
    } catch (error) {
      console.error('Tenant rate limit error:', error);
      next(); // Continue on error to avoid blocking requests
    }
  };
}

export { tokenService };