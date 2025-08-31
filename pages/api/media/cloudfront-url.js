/**
 * CloudFront Signed URL Generation API
 * Generates secure signed URLs for media access through CloudFront
 */

import { getServerSession } from 'next-auth/next';
import { authOptions } from '../auth/[...nextauth].js';
import CloudFrontSignedURLService from '../../../services/CloudFrontSignedURLService.js';
import { APIError, ValidationError, AuthorizationError } from '../../../lib/errors/index.js';
import dbConnect from '../../../lib/dbConnect.js';
import Media from '../../../models/Media.js';

const signedURLService = new CloudFrontSignedURLService();

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    await dbConnect();
    
    const session = await getServerSession(req, res, authOptions);
    if (!session?.user) {
      throw new AuthorizationError('Authentication required');
    }

    const { mediaId, mediaIds, options = {} } = req.body;
    const tenantId = session.user.tenantId;

    // Validate input
    if (!mediaId && !mediaIds) {
      throw new ValidationError('Either mediaId or mediaIds is required');
    }

    if (mediaId && mediaIds) {
      throw new ValidationError('Provide either mediaId or mediaIds, not both');
    }

    // Handle single media file
    if (mediaId) {
      const signedUrl = await generateSingleSignedURL(mediaId, tenantId, options);
      return res.status(200).json({ signedUrl });
    }

    // Handle multiple media files
    if (mediaIds && Array.isArray(mediaIds)) {
      if (mediaIds.length > 100) {
        throw new ValidationError('Maximum 100 media files can be processed at once');
      }

      const signedUrls = await generateBulkSignedURLs(mediaIds, tenantId, options);
      return res.status(200).json({ signedUrls });
    }

    throw new ValidationError('Invalid request format');

  } catch (error) {
    console.error('CloudFront signed URL generation error:', error);
    
    if (error instanceof APIError) {
      return res.status(error.statusCode).json({ error: error.message });
    }
    
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Generate signed URL for a single media file
 */
async function generateSingleSignedURL(mediaId, tenantId, options) {
  // Find media file
  const media = await Media.findOne({ 
    _id: mediaId, 
    tenantId: tenantId 
  });

  if (!media) {
    throw new ValidationError('Media file not found');
  }

  // Check if media is public (no signed URL needed)
  if (media.isPublic && media.accessPolicy === 'public') {
    return `https://${process.env.CLOUDFRONT_DOMAIN}/${media.s3Key}`;
  }

  // Generate signed URL
  const resourcePath = `/${media.s3Key}`;
  const signedUrlOptions = {
    expiresIn: options.expiresIn || 3600, // 1 hour default
    ipAddress: options.ipAddress || null,
    dateGreaterThan: options.dateGreaterThan || null
  };

  const signedUrl = signedURLService.generateSignedURL(resourcePath, signedUrlOptions);

  // Log access for audit
  console.log(`Generated signed URL for media ${mediaId} by tenant ${tenantId}`);

  return {
    mediaId,
    signedUrl,
    expiresAt: new Date(Date.now() + (signedUrlOptions.expiresIn * 1000)),
    accessPolicy: media.accessPolicy
  };
}

/**
 * Generate signed URLs for multiple media files
 */
async function generateBulkSignedURLs(mediaIds, tenantId, options) {
  // Find all media files
  const mediaFiles = await Media.find({ 
    _id: { $in: mediaIds }, 
    tenantId: tenantId 
  });

  if (mediaFiles.length === 0) {
    throw new ValidationError('No media files found');
  }

  const results = [];
  const signedUrlOptions = {
    expiresIn: options.expiresIn || 3600,
    ipAddress: options.ipAddress || null,
    dateGreaterThan: options.dateGreaterThan || null
  };

  for (const media of mediaFiles) {
    try {
      let signedUrl;

      // Check if media is public
      if (media.isPublic && media.accessPolicy === 'public') {
        signedUrl = `https://${process.env.CLOUDFRONT_DOMAIN}/${media.s3Key}`;
      } else {
        const resourcePath = `/${media.s3Key}`;
        signedUrl = signedURLService.generateSignedURL(resourcePath, signedUrlOptions);
      }

      results.push({
        mediaId: media._id,
        signedUrl,
        expiresAt: new Date(Date.now() + (signedUrlOptions.expiresIn * 1000)),
        accessPolicy: media.accessPolicy,
        success: true
      });
    } catch (error) {
      console.error(`Failed to generate signed URL for media ${media._id}:`, error);
      results.push({
        mediaId: media._id,
        error: error.message,
        success: false
      });
    }
  }

  // Log bulk access
  console.log(`Generated ${results.filter(r => r.success).length} signed URLs for tenant ${tenantId}`);

  return results;
}

/**
 * Generate signed URL with custom policy
 */
export async function generateCustomPolicyURL(req, res) {
  try {
    const session = await getServerSession(req, res, authOptions);
    if (!session?.user) {
      throw new AuthorizationError('Authentication required');
    }

    const { mediaId, policy } = req.body;
    const tenantId = session.user.tenantId;

    if (!mediaId || !policy) {
      throw new ValidationError('mediaId and policy are required');
    }

    // Find media file
    const media = await Media.findOne({ 
      _id: mediaId, 
      tenantId: tenantId 
    });

    if (!media) {
      throw new ValidationError('Media file not found');
    }

    // Validate policy structure
    if (!policy.Statement || !Array.isArray(policy.Statement)) {
      throw new ValidationError('Invalid policy format');
    }

    const resourcePath = `/${media.s3Key}`;
    const signedUrl = signedURLService.generateSignedURLWithCustomPolicy(
      `https://${process.env.CLOUDFRONT_DOMAIN}${resourcePath}`,
      policy
    );

    return res.status(200).json({
      mediaId,
      signedUrl,
      policy,
      accessPolicy: media.accessPolicy
    });

  } catch (error) {
    console.error('Custom policy signed URL generation error:', error);
    
    if (error instanceof APIError) {
      return res.status(error.statusCode).json({ error: error.message });
    }
    
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Validate signed URL
 */
export async function validateSignedURL(req, res) {
  try {
    const { signedUrl } = req.body;

    if (!signedUrl) {
      throw new ValidationError('signedUrl is required');
    }

    const isValid = signedURLService.isURLValid(signedUrl);
    const expiration = signedURLService.getURLExpiration(signedUrl);

    return res.status(200).json({
      isValid,
      expiration,
      timeRemaining: expiration ? Math.max(0, expiration.getTime() - Date.now()) : null
    });

  } catch (error) {
    console.error('Signed URL validation error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}