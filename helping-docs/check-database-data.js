import { MongoClient } from 'mongodb';

async function checkDatabaseData() {
    const uri = "mongodb+srv://shubhamsw2005:oPDpxYsFvbdJvMi6@sellerauth.d3v2srv.mongodb.net/";
    const client = new MongoClient(uri);

    try {
        console.log('🔍 CHECKING DATABASE FOR REAL DATA SENT DURING TESTS');
        console.log('====================================================');
        
        await client.connect();
        console.log('✅ Connected to MongoDB');
        
        const db = client.db(); // Use default database
        
        // List all collections
        const collections = await db.listCollections().toArray();
        console.log('\n📊 Available Collections:');
        collections.forEach(col => {
            console.log(`   • ${col.name}`);
        });
        
        // Check Users collection (where our test data should be)
        console.log('\n👤 USERS COLLECTION DATA:');
        console.log('=========================');
        
        const users = db.collection('users');
        const userCount = await users.countDocuments();
        console.log(`📊 Total users in database: ${userCount}`);
        
        if (userCount > 0) {
            console.log('\n📋 Recent Users (created during our tests):');
            const recentUsers = await users.find({})
                .sort({ createdAt: -1 })
                .limit(10)
                .toArray();
            
            recentUsers.forEach((user, index) => {
                console.log(`\n${index + 1}. User ID: ${user._id}`);
                console.log(`   📧 Email: ${user.email}`);
                console.log(`   👤 Name: ${user.name}`);
                console.log(`   📅 Created: ${user.createdAt}`);
                console.log(`   🔒 Status: ${user.status}`);
                console.log(`   🎭 Role: ${user.role}`);
                
                // Check if this was from our test
                if (user.email.includes('test') || user.email.includes('demo') || user.email.includes('perf') || user.email.includes('realdata')) {
                    console.log(`   🧪 TEST DATA: ✅ This was created during our API tests!`);
                }
            });
        }
        
        // Check Posts collection
        console.log('\n📝 POSTS COLLECTION DATA:');
        console.log('=========================');
        
        const posts = db.collection('posts');
        const postCount = await posts.countDocuments();
        console.log(`📊 Total posts in database: ${postCount}`);
        
        if (postCount > 0) {
            console.log('\n📋 Recent Posts:');
            const recentPosts = await posts.find({})
                .sort({ createdAt: -1 })
                .limit(5)
                .toArray();
            
            recentPosts.forEach((post, index) => {
                console.log(`\n${index + 1}. Post ID: ${post._id}`);
                console.log(`   📰 Title: ${post.title}`);
                console.log(`   👤 Author: ${post.author}`);
                console.log(`   📅 Created: ${post.createdAt}`);
                console.log(`   📊 Content Length: ${post.content ? post.content.length : 0} characters`);
                
                if (post.content && post.content.length > 1000) {
                    console.log(`   📦 LARGE CONTENT: This appears to be test data!`);
                }
            });
        }
        
        // Check for any other collections that might contain our test data
        console.log('\n🔍 CHECKING OTHER COLLECTIONS:');
        console.log('==============================');
        
        for (const collection of collections) {
            if (!['users', 'posts'].includes(collection.name)) {
                const col = db.collection(collection.name);
                const count = await col.countDocuments();
                console.log(`📊 ${collection.name}: ${count} documents`);
                
                if (count > 0 && count < 20) {
                    // Show sample data for small collections
                    const samples = await col.find({}).limit(3).toArray();
                    samples.forEach((doc, index) => {
                        console.log(`   ${index + 1}. ${JSON.stringify(doc, null, 2).substring(0, 200)}...`);
                    });
                }
            }
        }
        
        // Summary of test data found
        console.log('\n🎯 TEST DATA SUMMARY:');
        console.log('====================');
        
        const testUsers = await users.find({
            $or: [
                { email: { $regex: /test|demo|perf|realdata|api/i } },
                { name: { $regex: /test|demo|perf|real|api/i } }
            ]
        }).toArray();
        
        console.log(`✅ Found ${testUsers.length} users created during our API tests`);
        
        if (testUsers.length > 0) {
            console.log('\n📋 Test Users Details:');
            testUsers.forEach((user, index) => {
                console.log(`${index + 1}. ${user.email} (${user.name}) - Created: ${user.createdAt}`);
            });
        }
        
        // Check for recent activity (last hour)
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
        const recentUsers = await users.find({
            createdAt: { $gte: oneHourAgo }
        }).toArray();
        
        console.log(`\n⏰ Users created in the last hour: ${recentUsers.length}`);
        if (recentUsers.length > 0) {
            console.log('   These are likely from our recent API tests!');
            recentUsers.forEach(user => {
                console.log(`   • ${user.email} - ${user.createdAt}`);
            });
        }
        
    } catch (error) {
        console.error('❌ Error checking database:', error.message);
    } finally {
        await client.close();
        console.log('\n🔌 Database connection closed');
    }
}

checkDatabaseData();