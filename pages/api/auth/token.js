import TokenService from '../../../lib/auth/TokenService.js';
import User from '../../../models/User.js';
import dbConnect from '../../../lib/dbConnect.js';
import { refreshTokenAuth } from '../../../middleware/jwtAuth.js';

const tokenService = new TokenService();

export default async function handler(req, res) {
  if (req.method === 'POST') {
    return await handleTokenRefresh(req, res);
  } else if (req.method === 'DELETE') {
    return await handleTokenRevoke(req, res);
  } else {
    return res.status(405).json({ error: 'Method not allowed' });
  }
}

/**
 * Refresh access token using refresh token
 */
async function handleTokenRefresh(req, res) {
  try {
    // Use refresh token middleware to validate
    await new Promise((resolve, reject) => {
      refreshTokenAuth(req, res, (error) => {
        if (error) reject(error);
        else resolve();
      });
    });

    // Generate new access token
    const tokenPair = await tokenService.refreshAccessToken(
      req.body.refreshToken || req.cookies?.refreshToken,
      req.user
    );

    res.status(200).json({
      success: true,
      data: tokenPair
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    
    if (error.message.includes('expired') || error.message.includes('invalid')) {
      return res.status(401).json({
        error: error.message,
        code: 'REFRESH_TOKEN_INVALID'
      });
    }

    res.status(500).json({
      error: 'Failed to refresh token',
      code: 'REFRESH_ERROR'
    });
  }
}

/**
 * Revoke refresh token (logout)
 */
async function handleTokenRevoke(req, res) {
  try {
    const { refreshToken, revokeAll = false } = req.body;
    
    if (!refreshToken) {
      return res.status(400).json({
        error: 'Refresh token required',
        code: 'MISSING_REFRESH_TOKEN'
      });
    }

    // Verify refresh token to get user info
    const decoded = await tokenService.verifyRefreshToken(refreshToken);
    
    let revokedCount = 0;
    
    if (revokeAll) {
      // Revoke all user tokens
      revokedCount = await tokenService.revokeAllUserTokens(decoded.userId);
    } else {
      // Revoke specific token
      const success = await tokenService.revokeRefreshToken(decoded.userId, decoded.tokenId);
      revokedCount = success ? 1 : 0;
    }

    // Also blacklist the access token if provided
    const accessToken = tokenService.extractTokenFromRequest(req);
    if (accessToken) {
      await tokenService.blacklistAccessToken(accessToken);
    }

    res.status(200).json({
      success: true,
      message: `Successfully revoked ${revokedCount} token(s)`,
      revokedCount
    });
  } catch (error) {
    console.error('Token revoke error:', error);
    res.status(500).json({
      error: 'Failed to revoke token',
      code: 'REVOKE_ERROR'
    });
  }
}