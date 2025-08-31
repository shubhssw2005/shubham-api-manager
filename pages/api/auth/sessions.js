import { jwtAuth } from '../../../middleware/jwtAuth.js';
import { tokenService } from '../../../middleware/jwtAuth.js';

export default async function handler(req, res) {
  // Apply JWT authentication
  await new Promise((resolve, reject) => {
    jwtAuth(req, res, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });

  if (req.method === 'GET') {
    return await handleGetSessions(req, res);
  } else if (req.method === 'DELETE') {
    return await handleDeleteSession(req, res);
  } else {
    return res.status(405).json({ error: 'Method not allowed' });
  }
}

/**
 * Get user's active sessions
 */
async function handleGetSessions(req, res) {
  try {
    const sessions = await tokenService.getUserSessions(req.user.id);
    
    res.status(200).json({
      success: true,
      data: {
        sessions: sessions.map(session => ({
          tokenId: session.tokenId,
          createdAt: session.createdAt,
          userAgent: session.userAgent,
          ipAddress: session.ipAddress,
          isCurrent: session.tokenId === req.refreshTokenData?.tokenId
        })),
        total: sessions.length
      }
    });
  } catch (error) {
    console.error('Get sessions error:', error);
    res.status(500).json({
      error: 'Failed to get sessions',
      code: 'GET_SESSIONS_ERROR'
    });
  }
}

/**
 * Delete specific session or all sessions
 */
async function handleDeleteSession(req, res) {
  try {
    const { tokenId, all = false } = req.body;
    
    let revokedCount = 0;
    
    if (all) {
      // Revoke all user sessions
      revokedCount = await tokenService.revokeAllUserTokens(req.user.id);
    } else if (tokenId) {
      // Revoke specific session
      const success = await tokenService.revokeRefreshToken(req.user.id, tokenId);
      revokedCount = success ? 1 : 0;
    } else {
      return res.status(400).json({
        error: 'Token ID required or set all=true',
        code: 'MISSING_TOKEN_ID'
      });
    }

    res.status(200).json({
      success: true,
      message: `Successfully revoked ${revokedCount} session(s)`,
      revokedCount
    });
  } catch (error) {
    console.error('Delete session error:', error);
    res.status(500).json({
      error: 'Failed to delete session',
      code: 'DELETE_SESSION_ERROR'
    });
  }
}