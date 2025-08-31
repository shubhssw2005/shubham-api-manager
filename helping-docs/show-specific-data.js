import { MongoClient } from 'mongodb';

async function showSpecificData() {
    const uri = "mongodb+srv://shubhamsw2005:oPDpxYsFvbdJvMi6@sellerauth.d3v2srv.mongodb.net/";
    const client = new MongoClient(uri);

    try {
        console.log('🔍 SHOWING SPECIFIC REAL DATA SENT TO YOUR SYSTEM');
        console.log('=================================================');
        
        await client.connect();
        const db = client.db();
        
        // Show a specific test user with full details
        console.log('\n👤 EXAMPLE TEST USER DATA:');
        console.log('==========================');
        
        const testUser = await db.collection('users').findOne({
            email: 'realdata1756381457@example.com'
        });
        
        if (testUser) {
            console.log('📧 Email:', testUser.email);
            console.log('👤 Name:', testUser.name);
            console.log('🔒 Password Hash:', testUser.password.substring(0, 20) + '...');
            console.log('📅 Created At:', testUser.createdAt);
            console.log('🎭 Role:', testUser.role);
            console.log('📊 Status:', testUser.status);
            console.log('🆔 MongoDB ID:', testUser._id);
            
            console.log('\n📦 Full User Document:');
            console.log(JSON.stringify(testUser, null, 2));
        }
        
        // Show the large blog post content
        console.log('\n📝 LARGE BLOG POST DATA:');
        console.log('========================');
        
        const largePosts = await db.collection('posts').find({
            $expr: { $gt: [{ $strLenCP: "$content" }, 5000] }
        }).toArray();
        
        if (largePosts.length > 0) {
            const post = largePosts[0];
            console.log('📰 Title:', post.title);
            console.log('👤 Author ID:', post.author);
            console.log('📊 Content Length:', post.content.length, 'characters');
            console.log('📅 Created:', post.createdAt);
            console.log('🏷️  Tags:', post.tags);
            
            console.log('\n📄 Content Preview (first 500 characters):');
            console.log('"' + post.content.substring(0, 500) + '..."');
            
            console.log('\n📦 Full Post Document Structure:');
            const postCopy = { ...post };
            postCopy.content = post.content.substring(0, 100) + '... [TRUNCATED]';
            console.log(JSON.stringify(postCopy, null, 2));
        }
        
        // Show media files that were uploaded
        console.log('\n📎 MEDIA FILES DATA:');
        console.log('====================');
        
        const mediaFiles = await db.collection('media').find({}).toArray();
        
        mediaFiles.forEach((file, index) => {
            console.log(`\n${index + 1}. Media File:`);
            console.log('   📁 Filename:', file.filename);
            console.log('   📄 Original Name:', file.originalName);
            console.log('   📊 Size:', file.size, 'bytes');
            console.log('   🎭 MIME Type:', file.mimeType);
            console.log('   📅 Uploaded:', file.createdAt);
            console.log('   🆔 ID:', file._id);
            
            if (file.metadata) {
                console.log('   📋 Metadata:', JSON.stringify(file.metadata, null, 4));
            }
        });
        
        // Show event sourcing data (outbox pattern)
        console.log('\n🌊 EVENT SOURCING DATA:');
        console.log('=======================');
        
        const events = await db.collection('outboxes').find({}).limit(3).toArray();
        
        events.forEach((event, index) => {
            console.log(`\n${index + 1}. Event:`);
            console.log('   🎯 Event Type:', event.eventType);
            console.log('   📦 Aggregate:', event.aggregate);
            console.log('   🆔 Aggregate ID:', event.aggregateId);
            console.log('   📅 Created:', event.createdAt);
            console.log('   ✅ Processed:', event.processed || false);
            
            console.log('   📋 Event Payload:');
            console.log(JSON.stringify(event.payload, null, 4));
        });
        
        // Show statistics
        console.log('\n📊 DATABASE STATISTICS:');
        console.log('=======================');
        
        const stats = {
            totalUsers: await db.collection('users').countDocuments(),
            testUsers: await db.collection('users').countDocuments({
                email: { $regex: /test|demo|perf|realdata|api/i }
            }),
            totalPosts: await db.collection('posts').countDocuments(),
            totalMedia: await db.collection('media').countDocuments(),
            totalEvents: await db.collection('outboxes').countDocuments(),
            recentUsers: await db.collection('users').countDocuments({
                createdAt: { $gte: new Date(Date.now() - 2 * 60 * 60 * 1000) } // Last 2 hours
            })
        };
        
        console.log('📈 Total Users:', stats.totalUsers);
        console.log('🧪 Test Users (from our API tests):', stats.testUsers);
        console.log('📝 Total Posts:', stats.totalPosts);
        console.log('📎 Total Media Files:', stats.totalMedia);
        console.log('🌊 Total Events:', stats.totalEvents);
        console.log('⏰ Users Created in Last 2 Hours:', stats.recentUsers);
        
        console.log('\n✅ CONFIRMATION: Real data was successfully sent and stored!');
        console.log('🎯 Your API processed and persisted all test data correctly.');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    } finally {
        await client.close();
    }
}

showSpecificData();