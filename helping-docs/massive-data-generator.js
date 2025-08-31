import { MongoClient } from 'mongodb';
import fs from 'fs';

const uri = "mongodb+srv://shubhamsw2005:oPDpxYsFvbdJvMi6@sellerauth.d3v2srv.mongodb.net/";

class MassiveDataGenerator {
    constructor() {
        this.client = new MongoClient(uri);
        this.db = null;
        this.users = [];
        this.posts = [];
        this.stats = {
            usersProcessed: 0,
            postsCreated: 0,
            postsUpdated: 0,
            postsSoftDeleted: 0,
            startTime: null,
            endTime: null
        };
    }

    async connect() {
        await this.client.connect();
        this.db = this.client.db();
        console.log('✅ Connected to MongoDB');
    }

    async disconnect() {
        await this.client.close();
        console.log('🔌 Disconnected from MongoDB');
    }

    generateSlug(title, userId, postIndex) {
        return title
            .toLowerCase()
            .replace(/[^a-z0-9\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .trim('-') + `-${userId}-${postIndex}`;
    }

    generatePostContent(userId, postIndex) {
        const topics = [
            'Technology and Innovation',
            'Software Development Best Practices',
            'Machine Learning and AI',
            'Web Development Trends',
            'Database Optimization',
            'Cloud Computing Solutions',
            'Cybersecurity Insights',
            'Mobile App Development',
            'DevOps and Automation',
            'Data Science and Analytics'
        ];

        const topic = topics[postIndex % topics.length];
        const title = `${topic} - Post ${postIndex + 1} by User ${userId}`;
        
        return {
            title: title,
            slug: this.generateSlug(title, userId, postIndex),
            content: `This is a comprehensive blog post about ${topic.toLowerCase()}. 
            
Post Number: ${postIndex + 1}
Author User ID: ${userId}
Created for massive data testing and performance analysis.

Content includes:
- Detailed technical analysis
- Real-world examples and case studies
- Best practices and recommendations
- Performance benchmarks and metrics
- Future trends and predictions

${Array.from({length: 50}, (_, i) => 
    `Sentence ${i + 1}: This is detailed content about ${topic.toLowerCase()} that provides valuable insights for developers and technical professionals.`
).join(' ')}

This post demonstrates the relationship between users and posts in our system, with proper foreign key relationships and soft delete capabilities.`,
            
            excerpt: `A comprehensive guide to ${topic.toLowerCase()} with practical examples and insights.`,
            
            tags: [
                topic.toLowerCase().replace(/\s+/g, '-'),
                'technical',
                'development',
                'best-practices',
                `user-${userId}`,
                `post-${postIndex + 1}`
            ],
            
            status: postIndex % 10 === 0 ? 'draft' : 'published',
            featured: postIndex % 25 === 0,
            
            metadata: {
                postNumber: postIndex + 1,
                authorUserId: userId,
                wordCount: Math.floor(Math.random() * 1000) + 500,
                readingTime: Math.floor(Math.random() * 10) + 2,
                category: topic,
                difficulty: ['beginner', 'intermediate', 'advanced'][postIndex % 3],
                estimatedViews: Math.floor(Math.random() * 10000),
                socialShares: Math.floor(Math.random() * 500)
            }
        };
    }

    async getAllUsers() {
        console.log('📊 Fetching all users...');
        this.users = await this.db.collection('users').find({
            email: { $regex: /test|demo|perf|realdata|api/i }
        }).toArray();
        
        console.log(`✅ Found ${this.users.length} test users`);
        return this.users;
    }

    async createPostsForUser(user, postsPerUser = 1000) {
        console.log(`📝 Creating ${postsPerUser} posts for user: ${user.email}`);
        
        const posts = [];
        const batchSize = 100; // Insert in batches for better performance
        
        for (let i = 0; i < postsPerUser; i += batchSize) {
            const batch = [];
            const endIndex = Math.min(i + batchSize, postsPerUser);
            
            for (let j = i; j < endIndex; j++) {
                const postData = this.generatePostContent(user._id, j);
                
                batch.push({
                    ...postData,
                    author: user._id, // Foreign key relationship
                    authorEmail: user.email, // Denormalized for easier querying
                    authorName: user.name,
                    createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000), // Random date within last 30 days
                    updatedAt: new Date(),
                    isDeleted: false, // For soft delete functionality
                    deletedAt: null,
                    deletedBy: null,
                    version: 1
                });
            }
            
            // Insert batch
            const result = await this.db.collection('posts').insertMany(batch);
            posts.push(...Object.values(result.insertedIds));
            this.stats.postsCreated += batch.length;
            
            // Progress indicator
            if ((i + batchSize) % 500 === 0 || endIndex === postsPerUser) {
                console.log(`   📈 Progress: ${endIndex}/${postsPerUser} posts created for ${user.email}`);
            }
        }
        
        this.stats.usersProcessed++;
        return posts;
    }

    async cleanupExistingPosts() {
        console.log('🧹 Cleaning up existing posts to avoid conflicts...');
        const deleteResult = await this.db.collection('posts').deleteMany({
            authorEmail: { $regex: /test|demo|perf|realdata|api/i }
        });
        console.log(`✅ Removed ${deleteResult.deletedCount} existing test posts`);
    }

    async createAllPosts() {
        console.log('\n🚀 STARTING MASSIVE POST CREATION');
        console.log('==================================');
        
        // Clean up existing posts first
        await this.cleanupExistingPosts();
        
        this.stats.startTime = new Date();
        
        for (const user of this.users) {
            await this.createPostsForUser(user, 1000);
            
            // Show overall progress
            console.log(`✅ Completed user ${this.stats.usersProcessed}/${this.users.length} - Total posts: ${this.stats.postsCreated}`);
        }
        
        this.stats.endTime = new Date();
        console.log('\n🎉 POST CREATION COMPLETED!');
    }

    async performCRUDOperations() {
        console.log('\n🔧 PERFORMING CRUD OPERATIONS');
        console.log('=============================');
        
        // READ - Get some posts
        console.log('📖 READ: Fetching sample posts...');
        const samplePosts = await this.db.collection('posts').find({})
            .limit(10)
            .toArray();
        console.log(`✅ Retrieved ${samplePosts.length} sample posts`);
        
        // UPDATE - Update some posts
        console.log('✏️  UPDATE: Updating posts...');
        const updateResult = await this.db.collection('posts').updateMany(
            { status: 'draft' },
            { 
                $set: { 
                    status: 'published',
                    updatedAt: new Date(),
                    'metadata.lastModified': new Date()
                },
                $inc: { version: 1 }
            }
        );
        this.stats.postsUpdated = updateResult.modifiedCount;
        console.log(`✅ Updated ${updateResult.modifiedCount} posts from draft to published`);
        
        // SOFT DELETE - Soft delete some posts (NOT removing from database)
        console.log('🗑️  SOFT DELETE: Performing soft deletes...');
        
        // Get some posts to soft delete
        const postsToDelete = await this.db.collection('posts').find({
            isDeleted: false
        }).limit(1000).toArray();
        
        const softDeleteResult = await this.db.collection('posts').updateMany(
            { _id: { $in: postsToDelete.map(p => p._id) } },
            {
                $set: {
                    isDeleted: true,
                    deletedAt: new Date(),
                    deletedBy: 'system_test',
                    deletedReason: 'Testing soft delete functionality'
                },
                $inc: { version: 1 }
            }
        );
        
        this.stats.postsSoftDeleted = softDeleteResult.modifiedCount;
        console.log(`✅ Soft deleted ${softDeleteResult.modifiedCount} posts (still in database)`);
        
        // Verify soft deletes
        const activePostsCount = await this.db.collection('posts').countDocuments({ isDeleted: false });
        const deletedPostsCount = await this.db.collection('posts').countDocuments({ isDeleted: true });
        
        console.log(`📊 Active posts: ${activePostsCount}`);
        console.log(`📊 Soft deleted posts: ${deletedPostsCount}`);
        console.log('✅ Soft delete verification: Posts still exist in database but marked as deleted');
    }

    async generateStatistics() {
        console.log('\n📊 GENERATING COMPREHENSIVE STATISTICS');
        console.log('=====================================');
        
        const stats = {
            users: {
                total: await this.db.collection('users').countDocuments(),
                testUsers: await this.db.collection('users').countDocuments({
                    email: { $regex: /test|demo|perf|realdata|api/i }
                })
            },
            posts: {
                total: await this.db.collection('posts').countDocuments(),
                active: await this.db.collection('posts').countDocuments({ isDeleted: false }),
                softDeleted: await this.db.collection('posts').countDocuments({ isDeleted: true }),
                published: await this.db.collection('posts').countDocuments({ status: 'published', isDeleted: false }),
                draft: await this.db.collection('posts').countDocuments({ status: 'draft', isDeleted: false }),
                featured: await this.db.collection('posts').countDocuments({ featured: true, isDeleted: false })
            },
            relationships: {
                usersWithPosts: await this.db.collection('posts').distinct('author').then(authors => authors.length),
                avgPostsPerUser: 0
            },
            performance: {
                totalTimeSeconds: (this.stats.endTime - this.stats.startTime) / 1000,
                postsPerSecond: 0
            }
        };
        
        stats.relationships.avgPostsPerUser = Math.round(stats.posts.total / stats.relationships.usersWithPosts);
        stats.performance.postsPerSecond = Math.round(this.stats.postsCreated / stats.performance.totalTimeSeconds);
        
        // User-Post relationship analysis
        const userPostCounts = await this.db.collection('posts').aggregate([
            { $match: { isDeleted: false } },
            { $group: { _id: '$author', postCount: { $sum: 1 } } },
            { $sort: { postCount: -1 } },
            { $limit: 10 }
        ]).toArray();
        
        console.log('👥 USER STATISTICS:');
        console.log(`   Total Users: ${stats.users.total}`);
        console.log(`   Test Users: ${stats.users.testUsers}`);
        
        console.log('\n📝 POST STATISTICS:');
        console.log(`   Total Posts: ${stats.posts.total}`);
        console.log(`   Active Posts: ${stats.posts.active}`);
        console.log(`   Soft Deleted Posts: ${stats.posts.softDeleted}`);
        console.log(`   Published Posts: ${stats.posts.published}`);
        console.log(`   Draft Posts: ${stats.posts.draft}`);
        console.log(`   Featured Posts: ${stats.posts.featured}`);
        
        console.log('\n🔗 RELATIONSHIPS:');
        console.log(`   Users with Posts: ${stats.relationships.usersWithPosts}`);
        console.log(`   Average Posts per User: ${stats.relationships.avgPostsPerUser}`);
        
        console.log('\n⚡ PERFORMANCE:');
        console.log(`   Total Creation Time: ${stats.performance.totalTimeSeconds} seconds`);
        console.log(`   Posts Created per Second: ${stats.performance.postsPerSecond}`);
        console.log(`   Posts Created: ${this.stats.postsCreated}`);
        console.log(`   Posts Updated: ${this.stats.postsUpdated}`);
        console.log(`   Posts Soft Deleted: ${this.stats.postsSoftDeleted}`);
        
        console.log('\n🏆 TOP USERS BY POST COUNT:');
        for (let i = 0; i < Math.min(5, userPostCounts.length); i++) {
            const userPost = userPostCounts[i];
            const user = await this.db.collection('users').findOne({ _id: userPost._id });
            console.log(`   ${i + 1}. ${user?.email || 'Unknown'}: ${userPost.postCount} posts`);
        }
        
        return stats;
    }

    async createMongoDBCompassGuide() {
        const guide = `
# 🧭 MONGODB COMPASS VIEWING GUIDE
================================

## 📊 Database Overview
- **Database**: Default database in your MongoDB Atlas cluster
- **Total Collections**: users, posts, media, outboxes, and more
- **Total Records**: 76,000+ posts + 76 users + media files

## 🔍 How to View Data in MongoDB Compass

### 1. Connect to Your Database
- Connection String: mongodb+srv://shubhamsw2005:oPDpxYsFvbdJvMi6@sellerauth.d3v2srv.mongodb.net/
- Use MongoDB Compass desktop application

### 2. Key Collections to Explore

#### 👥 USERS Collection
- **Filter**: { "email": { "$regex": "test|demo|perf|realdata|api", "$options": "i" } }
- **Count**: 76 test users
- **Fields**: _id, email, name, password (hashed), createdAt, status, role

#### 📝 POSTS Collection  
- **Total Records**: ~76,000 posts
- **Filter for Active Posts**: { "isDeleted": false }
- **Filter for Soft Deleted**: { "isDeleted": true }
- **Filter by User**: { "author": ObjectId("USER_ID_HERE") }

### 3. Useful Queries for MongoDB Compass

#### View Posts by Specific User:
\`\`\`json
{ "authorEmail": "test@example.com" }
\`\`\`

#### View Only Active Posts:
\`\`\`json
{ "isDeleted": false }
\`\`\`

#### View Soft Deleted Posts:
\`\`\`json
{ "isDeleted": true }
\`\`\`

#### View Featured Posts:
\`\`\`json
{ "featured": true, "isDeleted": false }
\`\`\`

#### View Posts by Status:
\`\`\`json
{ "status": "published", "isDeleted": false }
\`\`\`

### 4. Aggregation Queries

#### Count Posts per User:
\`\`\`json
[
  { "$match": { "isDeleted": false } },
  { "$group": { "_id": "$authorEmail", "count": { "$sum": 1 } } },
  { "$sort": { "count": -1 } }
]
\`\`\`

#### Posts Created by Date:
\`\`\`json
[
  { "$match": { "isDeleted": false } },
  { "$group": { 
      "_id": { "$dateToString": { "format": "%Y-%m-%d", "date": "$createdAt" } },
      "count": { "$sum": 1 }
  }},
  { "$sort": { "_id": -1 } }
]
\`\`\`

### 5. Relationship Verification

#### Check User-Post Relationships:
1. Go to USERS collection
2. Copy a user's _id (ObjectId)
3. Go to POSTS collection  
4. Filter: { "author": ObjectId("PASTE_USER_ID_HERE") }
5. You should see exactly 1000 posts for that user

### 6. Soft Delete Verification

#### Verify Soft Deletes Work:
1. Filter: { "isDeleted": true }
2. Check these posts have:
   - isDeleted: true
   - deletedAt: timestamp
   - deletedBy: "system_test"
   - deletedReason: "Testing soft delete functionality"
3. Verify posts are NOT physically removed from database

## 📊 Expected Data Counts
- **Total Users**: 79 (76 test users + 3 original)
- **Total Posts**: ~76,000 (1000 per test user)
- **Active Posts**: ~75,000 (after soft deletes)
- **Soft Deleted Posts**: ~1,000
- **User-Post Relationships**: Each test user has exactly 1000 posts

## 🎯 Key Features Demonstrated
✅ **Foreign Key Relationships**: author field links to users._id
✅ **Soft Deletes**: Posts marked as deleted but not removed
✅ **CRUD Operations**: Create, Read, Update, Soft Delete
✅ **Data Integrity**: All relationships properly maintained
✅ **Performance**: 76,000 records created efficiently
✅ **Audit Trail**: Version tracking and timestamps

Your data is now ready for exploration in MongoDB Compass! 🚀
`;

        fs.writeFileSync('MongoDB_Compass_Guide.md', guide);
        console.log('📋 Created MongoDB_Compass_Guide.md for viewing instructions');
    }
}

async function main() {
    const generator = new MassiveDataGenerator();
    
    try {
        console.log('🚀 MASSIVE DATA GENERATION & CRUD TESTING');
        console.log('=========================================');
        console.log('Creating 1000 posts for each of 76 users (76,000 total posts!)');
        console.log('With proper relationships and soft delete functionality');
        console.log('');
        
        await generator.connect();
        
        // Get all test users
        await generator.getAllUsers();
        
        if (generator.users.length === 0) {
            console.log('❌ No test users found. Please run the user creation tests first.');
            return;
        }
        
        // Create massive amount of posts
        await generator.createAllPosts();
        
        // Perform CRUD operations
        await generator.performCRUDOperations();
        
        // Generate comprehensive statistics
        const stats = await generator.generateStatistics();
        
        // Create MongoDB Compass guide
        await generator.createMongoDBCompassGuide();
        
        console.log('\n🎉 MASSIVE DATA GENERATION COMPLETED!');
        console.log('====================================');
        console.log(`✅ Created ${generator.stats.postsCreated} posts with proper user relationships`);
        console.log(`✅ Updated ${generator.stats.postsUpdated} posts`);
        console.log(`✅ Soft deleted ${generator.stats.postsSoftDeleted} posts (still in database)`);
        console.log('✅ All data is now visible in MongoDB Compass');
        console.log('✅ Check MongoDB_Compass_Guide.md for viewing instructions');
        
    } catch (error) {
        console.error('❌ Error:', error);
    } finally {
        await generator.disconnect();
    }
}

main();