/**
 * Multi-Layer Cache System Usage Examples
 * 
 * This file demonstrates how to use the comprehensive cache system
 * with L1 (memory) and L2 (Redis) caching, invalidation, and warming.
 */

import { initializeCacheService, getCacheService } from '../middleware/cache.js';

// Initialize the cache service
const cacheService = initializeCacheService({
  // Memory cache settings
  memoryTTL: 300,        // 5 minutes
  memoryMaxKeys: 10000,  // Max 10k keys in memory
  
  // Redis cache settings
  redisTTL: 3600,        // 1 hour
  redisKeyPrefix: 'app:cache:',
  
  // Invalidation strategies
  invalidationStrategies: {
    immediate: true,     // Invalidate immediately on events
    delayed: false,      // Don't batch invalidations
    ttlBased: true,      // Use TTL-based invalidation
    tagBased: true       // Support tag-based invalidation
  },
  
  // Warming strategies
  warmingStrategies: {
    startup: true,       // Warm cache on startup
    scheduled: true,     // Periodic warming
    predictive: false,   // No ML-based predictions yet
    onDemand: true       // Support on-demand warming
  }
});

/**
 * Example 1: Basic Cache Operations
 */
async function basicCacheOperations() {
  console.log('=== Basic Cache Operations ===');
  
  // Set a value in cache
  await cacheService.set('user:123', { 
    id: '123', 
    name: 'John Doe', 
    email: 'john@example.com' 
  }, 600); // 10 minutes TTL
  
  // Get value from cache
  const user = await cacheService.get('user:123');
  console.log('Retrieved user:', user);
  
  // Delete from cache
  await cacheService.del('user:123');
  
  // Verify deletion
  const deletedUser = await cacheService.get('user:123');
  console.log('After deletion:', deletedUser); // Should be null
}

/**
 * Example 2: Request Coalescing (Thundering Herd Prevention)
 */
async function requestCoalescingExample() {
  console.log('\n=== Request Coalescing Example ===');
  
  let fetchCount = 0;
  const expensiveOperation = async () => {
    fetchCount++;
    console.log(`Expensive operation called (${fetchCount})`);
    
    // Simulate slow database query
    await new Promise(resolve => setTimeout(resolve, 100));
    
    return {
      data: 'expensive result',
      timestamp: Date.now(),
      fetchCount
    };
  };
  
  // Make 5 concurrent requests for the same data
  const promises = Array(5).fill().map(() => 
    cacheService.getOrSet('expensive:data', expensiveOperation, 300)
  );
  
  const results = await Promise.all(promises);
  
  console.log('All results identical:', results.every(r => r.fetchCount === 1));
  console.log('Fetch function called only once:', fetchCount === 1);
}

/**
 * Example 3: Tag-Based Cache Invalidation
 */
async function tagBasedInvalidationExample() {
  console.log('\n=== Tag-Based Invalidation Example ===');
  
  // Set cache entries with tags
  await cacheService.setWithTags('post:1', { 
    id: '1', 
    title: 'First Post',
    authorId: 'user123',
    categoryId: 'tech'
  }, 600, ['posts', 'user:user123', 'category:tech']);
  
  await cacheService.setWithTags('post:2', { 
    id: '2', 
    title: 'Second Post',
    authorId: 'user123',
    categoryId: 'news'
  }, 600, ['posts', 'user:user123', 'category:news']);
  
  await cacheService.setWithTags('post:3', { 
    id: '3', 
    title: 'Third Post',
    authorId: 'user456',
    categoryId: 'tech'
  }, 600, ['posts', 'user:user456', 'category:tech']);
  
  // Verify all posts are cached
  console.log('Post 1:', await cacheService.get('post:1'));
  console.log('Post 2:', await cacheService.get('post:2'));
  console.log('Post 3:', await cacheService.get('post:3'));
  
  // Invalidate all posts by user123
  await cacheService.invalidateByTags(['user:user123']);
  
  // Check what's left
  console.log('\nAfter invalidating user123 posts:');
  console.log('Post 1:', await cacheService.get('post:1')); // null
  console.log('Post 2:', await cacheService.get('post:2')); // null
  console.log('Post 3:', await cacheService.get('post:3')); // still there
}

/**
 * Example 4: Event-Driven Invalidation
 */
async function eventDrivenInvalidationExample() {
  console.log('\n=== Event-Driven Invalidation Example ===');
  
  const tenantId = 'tenant123';
  const postId = 'post456';
  
  // Set up some cached data
  await cacheService.set(`posts:${tenantId}:${postId}`, {
    id: postId,
    title: 'Original Title',
    content: 'Original content'
  }, 600);
  
  await cacheService.set(`posts:${tenantId}:list:recent`, [
    { id: postId, title: 'Original Title' }
  ], 600);
  
  await cacheService.set(`posts:${tenantId}:count`, 5, 600);
  
  console.log('Before update:');
  console.log('Post:', await cacheService.get(`posts:${tenantId}:${postId}`));
  console.log('Recent list:', await cacheService.get(`posts:${tenantId}:list:recent`));
  console.log('Count:', await cacheService.get(`posts:${tenantId}:count`));
  
  // Emit a post update event
  cacheService.emitModelEvent('post:updated', {
    id: postId,
    tenantId: tenantId,
    title: 'Updated Title',
    content: 'Updated content',
    authorId: 'author789'
  });
  
  // Wait for invalidation to process
  await new Promise(resolve => setTimeout(resolve, 100));
  
  console.log('\nAfter update event:');
  console.log('Post:', await cacheService.get(`posts:${tenantId}:${postId}`));
  console.log('Recent list:', await cacheService.get(`posts:${tenantId}:list:recent`));
  console.log('Count:', await cacheService.get(`posts:${tenantId}:count`));
}

/**
 * Example 5: Pattern-Based Invalidation
 */
async function patternBasedInvalidationExample() {
  console.log('\n=== Pattern-Based Invalidation Example ===');
  
  // Set up test data
  await cacheService.set('session:user123:web', { sessionId: 'web123' }, 300);
  await cacheService.set('session:user123:mobile', { sessionId: 'mobile123' }, 300);
  await cacheService.set('session:user456:web', { sessionId: 'web456' }, 300);
  await cacheService.set('profile:user123', { name: 'John' }, 300);
  
  console.log('Before invalidation:');
  console.log('User123 web session:', await cacheService.get('session:user123:web'));
  console.log('User123 mobile session:', await cacheService.get('session:user123:mobile'));
  console.log('User456 web session:', await cacheService.get('session:user456:web'));
  console.log('User123 profile:', await cacheService.get('profile:user123'));
  
  // Invalidate all sessions for user123
  await cacheService.invalidate('session:user123:*');
  
  console.log('\nAfter invalidating user123 sessions:');
  console.log('User123 web session:', await cacheService.get('session:user123:web'));
  console.log('User123 mobile session:', await cacheService.get('session:user123:mobile'));
  console.log('User456 web session:', await cacheService.get('session:user456:web'));
  console.log('User123 profile:', await cacheService.get('profile:user123'));
}

/**
 * Example 6: Cache Warming
 */
async function cacheWarmingExample() {
  console.log('\n=== Cache Warming Example ===');
  
  const tenantId = 'tenant789';
  
  // Warm tenant cache
  const warmedCount = await cacheService.warmTenant(tenantId, 'high');
  console.log(`Warmed ${warmedCount} cache entries for tenant ${tenantId}`);
  
  // Warm specific content type
  const contentWarmedCount = await cacheService.warmContentType('posts', 20, 'medium');
  console.log(`Warmed ${contentWarmedCount} post entries`);
}

/**
 * Example 7: Cache Statistics and Health Check
 */
async function statisticsAndHealthExample() {
  console.log('\n=== Statistics and Health Check ===');
  
  // Get cache statistics
  const stats = cacheService.getStats();
  console.log('Cache Statistics:');
  console.log('L1 Cache:', {
    hits: stats.cache.l1.hits,
    misses: stats.cache.l1.misses,
    hitRate: (stats.cache.l1.hitRate * 100).toFixed(2) + '%',
    keys: stats.cache.l1.keys
  });
  
  console.log('L2 Cache:', {
    hits: stats.cache.l2.hits,
    misses: stats.cache.l2.misses,
    hitRate: (stats.cache.l2.hitRate * 100).toFixed(2) + '%',
    sets: stats.cache.l2.sets
  });
  
  console.log('Coalescing:', {
    coalescedRequests: stats.cache.coalescing.coalescedRequests,
    pendingRequests: stats.cache.coalescing.pendingRequests
  });
  
  // Health check
  const health = await cacheService.healthCheck();
  console.log('\nHealth Check:', {
    healthy: health.healthy,
    timestamp: new Date(health.timestamp).toISOString()
  });
}

/**
 * Run all examples
 */
async function runExamples() {
  try {
    await basicCacheOperations();
    await requestCoalescingExample();
    await tagBasedInvalidationExample();
    await eventDrivenInvalidationExample();
    await patternBasedInvalidationExample();
    await cacheWarmingExample();
    await statisticsAndHealthExample();
    
    console.log('\n=== All Examples Completed Successfully ===');
  } catch (error) {
    console.error('Error running examples:', error);
  } finally {
    // Clean up
    await cacheService.close();
  }
}

// Export for use in other files
export {
  basicCacheOperations,
  requestCoalescingExample,
  tagBasedInvalidationExample,
  eventDrivenInvalidationExample,
  patternBasedInvalidationExample,
  cacheWarmingExample,
  statisticsAndHealthExample,
  runExamples
};

// Run examples if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExamples();
}