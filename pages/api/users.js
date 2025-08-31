import UltraDistributedDB from '../../lib/ultra-distributed-db.js';

export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ message: 'Method not allowed' });
    }

    try {
        await UltraDistributedDB.connect();
        
        const { filter } = req.query;

        let filterOptions = {};
        
        // Filter for test users if requested
        if (filter === 'test') {
            filterOptions.email_regex = 'test|demo|perf|realdata|api';
        }

        const users = await UltraDistributedDB.findUsers(filterOptions);

        console.log(`📊 Found ${users.length} users with ultra-distributed database (filter: ${filter || 'none'})`);

        // Convert to expected format
        const formattedUsers = users.map(user => ({
            _id: user.id || user._id,
            email: user.email,
            name: user.name,
            createdAt: user.createdAt || user.created_at
        }));

        res.status(200).json({
            success: true,
            count: formattedUsers.length,
            users: formattedUsers,
            filter: filter || 'none',
            database: 'ultra-distributed'
        });

    } catch (error) {
        console.error('❌ Error fetching users:', error);
        res.status(500).json({ 
            success: false,
            message: 'Failed to fetch users',
            error: error.message 
        });
    }
}