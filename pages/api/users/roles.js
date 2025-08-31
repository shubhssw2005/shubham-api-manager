import dbConnect from '@/lib/dbConnect';
import User from '@/models/User';
import { Role } from '@/models/Role';
import { requireApprovedUser } from '@/middleware/auth';
import { hasPermission, PERMISSION } from '@/lib/rbac';

export default async function handler(req, res) {
  await dbConnect();

  const user = await requireApprovedUser(req, res);
  if (!user) return;

  // Check if user has permission to manage users
  if (!hasPermission(user.role, PERMISSION.UPDATE)) {
    return res.status(403).json({ success: false, message: 'Insufficient permissions' });
  }

  try {
    switch (req.method) {
      case 'GET':
        return await getUsers(req, res);
      case 'PUT':
        return await updateUserRole(req, res, user);
      default:
        return res.status(405).json({ success: false, message: 'Method not allowed' });
    }
  } catch (error) {
    console.error('User roles API error:', error);
    return res.status(500).json({ success: false, message: 'Internal server error' });
  }
}

async function getUsers(req, res) {
  try {
    const { page = 1, limit = 10, search = '', role = '' } = req.query;
    
    const query = {};
    
    // Add search filter
    if (search) {
      query.$or = [
        { name: { $regex: search, $options: 'i' } },
        { email: { $regex: search, $options: 'i' } }
      ];
    }
    
    // Add role filter
    if (role) {
      query.role = role;
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);
    
    const users = await User.find(query)
      .select('-password')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    const total = await User.countDocuments(query);
    
    // Get role information for each user
    const usersWithRoleInfo = await Promise.all(
      users.map(async (user) => {
        const roleInfo = await Role.findOne({ name: user.role });
        return {
          ...user.toObject(),
          roleInfo: roleInfo ? {
            displayName: roleInfo.displayName,
            description: roleInfo.description,
            isSystemRole: roleInfo.isSystemRole
          } : null
        };
      })
    );

    return res.status(200).json({
      success: true,
      data: {
        users: usersWithRoleInfo,
        pagination: {
          page: parseInt(page),
          limit: parseInt(limit),
          total,
          pages: Math.ceil(total / parseInt(limit))
        }
      }
    });
  } catch (error) {
    console.error('Error fetching users:', error);
    return res.status(500).json({ success: false, message: 'Failed to fetch users' });
  }
}

async function updateUserRole(req, res, user) {
  try {
    const { userId, newRole } = req.body;

    if (!userId || !newRole) {
      return res.status(400).json({ 
        success: false, 
        message: 'User ID and new role are required' 
      });
    }

    // Check if the new role exists
    const roleExists = await Role.findOne({ name: newRole.toLowerCase() });
    if (!roleExists) {
      return res.status(404).json({ success: false, message: 'Role not found' });
    }

    // Find and update user
    const targetUser = await User.findById(userId);
    if (!targetUser) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    // Prevent users from changing their own role
    if (targetUser._id.toString() === user._id.toString()) {
      return res.status(403).json({ 
        success: false, 
        message: 'Cannot change your own role' 
      });
    }

    targetUser.role = newRole.toLowerCase();
    await targetUser.save();

    // Get updated user with role info
    const roleInfo = await Role.findOne({ name: targetUser.role });
    const updatedUser = {
      ...targetUser.toObject(),
      roleInfo: roleInfo ? {
        displayName: roleInfo.displayName,
        description: roleInfo.description,
        isSystemRole: roleInfo.isSystemRole
      } : null
    };

    delete updatedUser.password;

    return res.status(200).json({
      success: true,
      data: updatedUser,
      message: 'User role updated successfully'
    });
  } catch (error) {
    console.error('Error updating user role:', error);
    return res.status(500).json({ success: false, message: 'Failed to update user role' });
  }
}