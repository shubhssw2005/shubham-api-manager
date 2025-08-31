import UltraDistributedDB from '../../../lib/ultra-distributed-db.js';

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ message: 'Method not allowed' });
    }

    try {
        await UltraDistributedDB.connect();
        
        const { posts, batchSize, source } = req.body;

        if (!posts || !Array.isArray(posts)) {
            return res.status(400).json({ message: 'Invalid posts data' });
        }

        console.log(`🚀 Ultra-fast batch processing: ${posts.length} posts from ${source || 'unknown'}`);

        // Prepare posts for ultra-distributed insertion
        const postsToInsert = posts.map(post => ({
            title: post.title,
            slug: post.slug,
            content: post.content,
            excerpt: post.excerpt,
            tags: post.tags || [],
            status: post.status || 'published',
            featured: post.featured || false,
            author_id: post.author,
            author_email: post.authorEmail,
            author_name: post.authorName || 'Test User',
            view_count: Math.floor(Math.random() * 1000),
            like_count: Math.floor(Math.random() * 100),
            metadata: {
                ...post.metadata,
                batchProcessed: 'true',
                batchTimestamp: new Date().toISOString(),
                source: source || 'ultra-distributed-api',
                database_strategy: 'scylladb_primary_foundationdb_replica'
            }
        }));

        // Ultra-fast batch insert with distributed strategy
        const results = await UltraDistributedDB.createPostsBatch(postsToInsert);

        console.log(`✅ Ultra-fast batch completed: ${results.length} posts distributed across ScyllaDB + FoundationDB`);

        res.status(200).json({
            success: true,
            insertedCount: results.length,
            batchSize: posts.length,
            source: source,
            timestamp: new Date(),
            database: 'ultra-distributed',
            strategy: 'scylladb_primary_foundationdb_replica',
            performance: 'ultra-fast'
        });

    } catch (error) {
        console.error('❌ Batch insert error:', error);
        
        // Ultra-distributed database has redundancy for high availability
        console.error('❌ Ultra-distributed batch insert error:', error);

        res.status(500).json({ 
            success: false,
            message: 'Ultra-distributed batch insert failed',
            error: error.message,
            database: 'ultra-distributed',
            strategy: 'scylladb_foundationdb'
        });
    }
}