#!/usr/bin/env node

/**
 * Media Processor Worker Startup Script
 * Starts the media processing worker with proper error handling and graceful shutdown
 */

import MediaProcessor from '../workers/MediaProcessor.js';
import { config } from 'dotenv';

// Load environment variables
config();

// Validate required environment variables
const requiredEnvVars = [
  'AWS_REGION',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'MEDIA_PROCESSING_QUEUE_URL',
  'MEDIA_BUCKET'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
if (missingEnvVars.length > 0) {
  console.error('Missing required environment variables:', missingEnvVars);
  process.exit(1);
}

// Create media processor instance
const mediaProcessor = new MediaProcessor();

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Received SIGTERM, initiating graceful shutdown...');
  await mediaProcessor.shutdown();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('Received SIGINT, initiating graceful shutdown...');
  await mediaProcessor.shutdown();
  process.exit(0);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  mediaProcessor.shutdown().then(() => {
    process.exit(1);
  });
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  mediaProcessor.shutdown().then(() => {
    process.exit(1);
  });
});

// Start the media processor
async function start() {
  try {
    console.log('Starting Media Processor Worker...');
    console.log('Configuration:', {
      region: process.env.AWS_REGION,
      bucket: process.env.MEDIA_BUCKET,
      queueUrl: process.env.MEDIA_PROCESSING_QUEUE_URL,
      concurrency: process.env.MEDIA_PROCESSOR_CONCURRENCY || 5,
      maxRetries: process.env.MAX_PROCESSING_RETRIES || 3
    });
    
    await mediaProcessor.start();
  } catch (error) {
    console.error('Failed to start media processor:', error);
    process.exit(1);
  }
}

// Start the worker
start();