import S3EventProcessor from '../../../lib/events/S3EventProcessor.js';

/**
 * API endpoint to handle S3 event notifications from SNS
 * This endpoint receives SNS notifications when files are uploaded to S3
 */

const eventProcessor = new S3EventProcessor();

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Handle SNS subscription confirmation
    if (req.headers['x-amz-sns-message-type'] === 'SubscriptionConfirmation') {
      console.log('SNS Subscription Confirmation received');
      
      const message = JSON.parse(req.body);
      const subscribeURL = message.SubscribeURL;
      
      // In production, you would verify the signature and confirm the subscription
      console.log('Subscribe URL:', subscribeURL);
      
      return res.status(200).json({ 
        message: 'Subscription confirmation received',
        subscribeURL 
      });
    }

    // Handle SNS notification
    if (req.headers['x-amz-sns-message-type'] === 'Notification') {
      const snsMessage = JSON.parse(req.body);
      
      // Verify SNS signature (in production)
      // await verifySNSSignature(snsMessage);
      
      console.log('Processing S3 event notification:', {
        messageId: snsMessage.MessageId,
        timestamp: snsMessage.Timestamp,
        subject: snsMessage.Subject
      });
      
      // Process the S3 event
      await eventProcessor.processS3Event(snsMessage);
      
      return res.status(200).json({ 
        message: 'S3 event processed successfully',
        messageId: snsMessage.MessageId 
      });
    }

    // Handle direct S3 event (for testing)
    if (req.body.Records) {
      console.log('Processing direct S3 event');
      
      const s3Event = {
        Message: JSON.stringify(req.body)
      };
      
      await eventProcessor.processS3Event(s3Event);
      
      return res.status(200).json({ 
        message: 'Direct S3 event processed successfully' 
      });
    }

    return res.status(400).json({ 
      error: 'Invalid request format' 
    });

  } catch (error) {
    console.error('Error processing S3 event:', error);
    
    return res.status(500).json({ 
      error: 'Failed to process S3 event',
      details: error.message 
    });
  }
}

/**
 * Verify SNS message signature (implement in production)
 * @param {Object} message - SNS message
 * @returns {Promise<boolean>} Verification result
 */
async function verifySNSSignature(message) {
  // Implementation would verify the SNS signature using AWS SDK
  // For now, we'll skip verification in development
  return true;
}

// Disable body parsing to handle raw SNS messages
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
}