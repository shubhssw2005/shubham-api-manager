import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import bcrypt from 'bcryptjs';

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ message: 'Method not allowed' });
    }

    try {
        await dbConnect();

        // Check if admin user already exists
        const existingUser = await User.findOne({ email: 'admin@example.com' });
        if (existingUser) {
            return res.status(200).json({ message: 'Admin user already exists' });
        }

        // Create admin user
        const hashedPassword = await bcrypt.hash('admin123', 10);
        const user = new User({
            name: 'Admin User',
            email: 'admin@example.com',
            password: hashedPassword,
            role: 'admin'
        });

        await user.save();
        res.status(201).json({ message: 'Admin user created successfully' });
    } catch (error) {
        console.error('Error creating admin:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}
