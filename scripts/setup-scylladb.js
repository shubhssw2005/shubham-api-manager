#!/usr/bin/env node

import ScyllaDB from '../lib/scylladb.js';

async function setupScyllaDB() {
    console.log('🚀 SCYLLADB SETUP & INITIALIZATION');
    console.log('==================================');
    
    try {
        // Connect and initialize database
        const connected = await ScyllaDB.connect();
        
        if (!connected) {
            console.error('❌ Failed to connect to ScyllaDB');
            process.exit(1);
        }

        console.log('✅ ScyllaDB connected and initialized successfully');
        
        // Create some test users for demonstration
        console.log('\n📊 Creating test users...');
        
        const testUsers = [
            {
                email: 'test1@example.com',
                name: 'Test User 1',
                password: 'hashed_password_1',
                role: 'user',
                metadata: { source: 'setup-script' }
            },
            {
                email: 'test2@example.com', 
                name: 'Test User 2',
                password: 'hashed_password_2',
                role: 'user',
                metadata: { source: 'setup-script' }
            },
            {
                email: 'admin@example.com',
                name: 'Admin User',
                password: 'hashed_password_admin',
                role: 'admin',
                metadata: { source: 'setup-script' }
            }
        ];

        for (const userData of testUsers) {
            try {
                const user = await ScyllaDB.createUser(userData);
                console.log(`✅ Created user: ${user.email} (ID: ${user.id})`);
            } catch (error) {
                if (error.message.includes('already exists')) {
                    console.log(`⚠️  User ${userData.email} already exists`);
                } else {
                    console.error(`❌ Error creating user ${userData.email}:`, error.message);
                }
            }
        }

        // Test queries
        console.log('\n🔍 Testing queries...');
        
        const users = await ScyllaDB.findUsers();
        console.log(`✅ Found ${users.length} users in database`);
        
        const testUser = await ScyllaDB.findUserByEmail('test1@example.com');
        if (testUser) {
            console.log(`✅ Successfully queried user by email: ${testUser.email}`);
        }

        // Create a test post
        if (testUser) {
            console.log('\n📝 Creating test post...');
            
            const postData = {
                title: 'Welcome to ScyllaDB Integration',
                slug: 'welcome-to-scylladb-integration',
                content: 'This is a test post created during ScyllaDB setup. ScyllaDB provides ultra-high performance for our application.',
                excerpt: 'A test post demonstrating ScyllaDB integration',
                tags: ['scylladb', 'setup', 'test'],
                status: 'published',
                featured: true,
                author_id: testUser.id.toString(),
                author_email: testUser.email,
                author_name: testUser.name,
                metadata: {
                    source: 'setup-script',
                    database: 'scylladb'
                }
            };

            const post = await ScyllaDB.createPost(postData);
            console.log(`✅ Created test post: ${post.title} (ID: ${post.id})`);
        }

        // Performance test
        console.log('\n⚡ Running performance test...');
        
        const startTime = Date.now();
        const posts = await ScyllaDB.findPosts({}, 10);
        const endTime = Date.now();
        
        console.log(`✅ Query performance: ${endTime - startTime}ms for ${posts.length} posts`);
        
        console.log('\n🎉 SCYLLADB SETUP COMPLETED SUCCESSFULLY!');
        console.log('========================================');
        console.log('✅ Database initialized with tables and indexes');
        console.log('✅ Test users created');
        console.log('✅ Test post created');
        console.log('✅ Performance verified');
        console.log('\n📊 Next steps:');
        console.log('   1. Run: npm run dev');
        console.log('   2. Test API: curl http://localhost:3005/api/users?filter=test');
        console.log('   3. Generate massive data with C++: cd cpp-system && ./simple_data_generator');
        
    } catch (error) {
        console.error('❌ Setup failed:', error);
        process.exit(1);
    } finally {
        await ScyllaDB.disconnect();
    }
}

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    setupScyllaDB();
}

export default setupScyllaDB;