/**
 * Integration tests for C++ and Node.js system integration
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import Redis from 'ioredis';
import CppIntegrationService from '../../lib/integration/CppIntegrationService.js';
import {
  initializeCppIntegration,
  shutdownCppIntegration,
  EventTypes,
} from '../../middleware/cppIntegration.js';

describe('C++ Node.js Integration', () => {
  let integrationService;
  let redis;

  beforeAll(async () => {
    // Set up test Redis connection
    redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      db: 15, // Use test database
    });

    // Clear test database
    await redis.flushdb();
  });

  afterAll(async () => {
    if (integrationService) {
      await shutdownCppIntegration();
    }
    if (redis) {
      await redis.quit();
    }
  });

  beforeEach(async () => {
    // Clear test database before each test
    await redis.flushdb();
  });

  describe('CppIntegrationService', () => {
    it('should initialize successfully', async () => {
      integrationService = new CppIntegrationService({
        redisHost: process.env.REDIS_HOST || 'localhost',
        redisPort: process.env.REDIS_PORT || 6379,
        redisDb: 15,
        cppSystemUrl: 'http://localhost:8080',
        enableHealthCheck: false, // Disable for tests
        instanceId: 'test-nodejs',
      });

      const result = await integrationService.initialize();
      expect(result).toBe(true);
      expect(integrationService.isConnected).toBe(true);
    });

    it('should sync session data to Redis', async () => {
      const sessionId = 'test-session-123';
      const sessionData = {
        userId: 'user-456',
        tenantId: 'tenant-789',
        attributes: { test: 'value' },
        createdAt: Date.now(),
        expiresAt: Date.now() + 3600000,
        isAuthenticated: true,
        userRole: 'user',
        permissions: ['read', 'write'],
      };

      const result = await integrationService.syncSession(sessionId, sessionData);
      expect(result).toBe(true);

      // Verify session was stored in Redis
      const storedSession = await redis.get('nodejs:session:' + sessionId);
      expect(storedSession).toBeTruthy();

      const parsed = JSON.parse(storedSession);
      expect(parsed.user_id).toBe(sessionData.userId);
      expect(parsed.tenant_id).toBe(sessionData.tenantId);
      expect(parsed.is_authenticated).toBe(true);
    });

    it('should retrieve session data from Redis', async () => {
      const sessionId = 'test-session-456';
      const sessionData = {
        userId: 'user-789',
        tenantId: 'tenant-123',
        attributes: { key: 'value' },
        createdAt: Date.now(),
        expiresAt: Date.now() + 3600000,
        isAuthenticated: true,
        userRole: 'admin',
        permissions: ['read', 'write', 'admin'],
      };

      // Sync session first
      await integrationService.syncSession(sessionId, sessionData);

      // Retrieve session
      const retrieved = await integrationService.getSession(sessionId);
      expect(retrieved).toBeTruthy();
      expect(retrieved.userId).toBe(sessionData.userId);
      expect(retrieved.tenantId).toBe(sessionData.tenantId);
      expect(retrieved.userRole).toBe(sessionData.userRole);
      expect(retrieved.isAuthenticated).toBe(true);
    });

    it('should delete session from Redis', async () => {
      const sessionId = 'test-session-789';
      const sessionData = {
        userId: 'user-123',
        tenantId: 'tenant-456',
        attributes: {},
        createdAt: Date.now(),
        expiresAt: Date.now() + 3600000,
        isAuthenticated: true,
        userRole: 'user',
        permissions: ['read'],
      };

      // Sync session first
      await integrationService.syncSession(sessionId, sessionData);

      // Verify session exists
      let retrieved = await integrationService.getSession(sessionId);
      expect(retrieved).toBeTruthy();

      // Delete session
      const result = await integrationService.deleteSession(sessionId);
      expect(result).toBe(true);

      // Verify session is gone
      retrieved = await integrationService.getSession(sessionId);
      expect(retrieved).toBeNull();
    });

    it('should publish events to Redis', async () => {
      const eventPromise = new Promise((resolve) => {
        const subscriber = new Redis({
          host: process.env.REDIS_HOST || 'localhost',
          port: process.env.REDIS_PORT || 6379,
          db: 1, // Events database
        });

        subscriber.psubscribe('events:*');
        subscriber.on('pmessage', (pattern, channel, message) => {
          const event = JSON.parse(message);
          subscriber.quit();
          resolve(event);
        });
      });

      // Publish event
      const eventId = await integrationService.publishEvent(
        EventTypes.USER_ACTION,
        'test.channel',
        {
          user_id: 'test-user',
          action: 'test-action',
          resource: '/test/resource',
        }
      );

      expect(eventId).toBeTruthy();

      // Wait for event to be received
      const receivedEvent = await eventPromise;
      expect(receivedEvent.id).toBe(eventId);
      expect(receivedEvent.type).toBe(EventTypes.USER_ACTION);
      expect(receivedEvent.channel).toBe('test.channel');
      expect(receivedEvent.source_system).toBe('test-nodejs');

      const payload = JSON.parse(receivedEvent.payload);
      expect(payload.user_id).toBe('test-user');
      expect(payload.action).toBe('test-action');
    });

    it('should handle routing decisions correctly', async () => {
      // Mock requests
      const cppRequest = {
        path: '/api/cache/get',
        method: 'GET',
        url: '/api/cache/get?key=test',
      };

      const nodejsRequest = {
        path: '/api/auth/login',
        method: 'POST',
        url: '/api/auth/login',
      };

      const unknownRequest = {
        path: '/api/unknown/endpoint',
        method: 'GET',
        url: '/api/unknown/endpoint',
      };

      // Set C++ system as healthy
      integrationService.cppSystemHealthy = true;

      expect(integrationService.shouldRouteToCpp(cppRequest)).toBe(true);
      expect(integrationService.shouldRouteToCpp(nodejsRequest)).toBe(false);
      expect(integrationService.shouldRouteToCpp(unknownRequest)).toBe(false);

      // Set C++ system as unhealthy
      integrationService.cppSystemHealthy = false;

      expect(integrationService.shouldRouteToCpp(cppRequest)).toBe(false);
      expect(integrationService.shouldRouteToCpp(nodejsRequest)).toBe(false);
      expect(integrationService.shouldRouteToCpp(unknownRequest)).toBe(false);
    });

    it('should collect and report statistics', async () => {
      // Perform some operations
      await integrationService.syncSession('test-session', {
        userId: 'test-user',
        tenantId: 'test-tenant',
        attributes: {},
        createdAt: Date.now(),
        expiresAt: Date.now() + 3600000,
        isAuthenticated: true,
        userRole: 'user',
        permissions: [],
      });

      await integrationService.publishEvent(
        EventTypes.SYSTEM_EVENT,
        'test.stats',
        { message: 'test event' }
      );

      const stats = integrationService.getStats();
      expect(stats).toBeTruthy();
      expect(stats.isConnected).toBe(true);
      expect(stats.sessionsSynced).toBeGreaterThan(0);
      expect(stats.eventsPublished).toBeGreaterThan(0);
    });
  });

  describe('Integration Middleware', () => {
    it('should initialize integration service through middleware', async () => {
      const service = await initializeCppIntegration({
        redisHost: process.env.REDIS_HOST || 'localhost',
        redisPort: process.env.REDIS_PORT || 6379,
        redisDb: 15,
        cppSystemUrl: 'http://localhost:8080',
        enableHealthCheck: false,
        instanceId: 'test-middleware',
      });

      expect(service).toBeTruthy();
      expect(service.isConnected).toBe(true);

      // Clean up
      await shutdownCppIntegration();
    });
  });

  describe('Event Handling', () => {
    it('should handle incoming events from C++', (done) => {
      integrationService.on('user-action', (event) => {
        expect(event.eventId).toBeTruthy();
        expect(event.payload.user_id).toBe('cpp-user');
        expect(event.payload.action).toBe('cpp-action');
        done();
      });

      // Simulate incoming event from C++
      const mockEvent = {
        id: 'cpp-event-123',
        type: 0, // USER_ACTION
        source_system: 'cpp-system',
        target_system: 'test-nodejs',
        channel: 'user.actions',
        payload: JSON.stringify({
          user_id: 'cpp-user',
          action: 'cpp-action',
          resource: '/cpp/resource',
        }),
        metadata: {},
        timestamp: Math.floor(Date.now() / 1000),
      };

      integrationService.handleIncomingEvent(
        'events:*',
        'events:user.actions',
        JSON.stringify(mockEvent)
      );
    });

    it('should ignore events from same system', () => {
      let eventReceived = false;

      integrationService.on('user-action', () => {
        eventReceived = true;
      });

      // Simulate event from same system
      const mockEvent = {
        id: 'self-event-123',
        type: 0, // USER_ACTION
        source_system: 'test-nodejs', // Same as our instance
        target_system: '',
        channel: 'user.actions',
        payload: JSON.stringify({
          user_id: 'self-user',
          action: 'self-action',
        }),
        metadata: {},
        timestamp: Math.floor(Date.now() / 1000),
      };

      integrationService.handleIncomingEvent(
        'events:*',
        'events:user.actions',
        JSON.stringify(mockEvent)
      );

      // Wait a bit and check that event was not processed
      setTimeout(() => {
        expect(eventReceived).toBe(false);
      }, 100);
    });
  });
});