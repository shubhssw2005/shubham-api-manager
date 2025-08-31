import dbConnect from '@/lib/dbConnect';
import { Role } from '@/models/Role';
import User from '@/models/User';
import { requireApprovedUser } from '@/middleware/auth';
import { hasPermission, PERMISSION } from '@/lib/rbac';

export default async function handler(req, res) {
  await dbConnect();

  const user = await requireApprovedUser(req, res);
  if (!user) return;

  // Check if user has permission to manage roles
  if (!hasPermission(user.role, PERMISSION.UPDATE)) {
    return res.status(403).json({ success: false, message: 'Insufficient permissions' });
  }

  try {
    switch (req.method) {
      case 'GET':
        return await getRoles(req, res);
      case 'POST':
        return await createRole(req, res, user);
      case 'PUT':
        return await updateRole(req, res, user);
      case 'DELETE':
        return await deleteRole(req, res, user);
      default:
        return res.status(405).json({ success: false, message: 'Method not allowed' });
    }
  } catch (error) {
    console.error('Roles API error:', error);
    return res.status(500).json({ success: false, message: 'Internal server error' });
  }
}

async function getRoles(req, res) {
  try {
    const roles = await Role.find({}).sort({ createdAt: -1 });
    
    // Get user count for each role
    const rolesWithUserCount = await Promise.all(
      roles.map(async (role) => {
        const userCount = await User.countDocuments({ role: role.name });
        return {
          ...role.toObject(),
          userCount,
          displayName: role.displayName
        };
      })
    );

    return res.status(200).json({ 
      success: true, 
      data: rolesWithUserCount 
    });
  } catch (error) {
    console.error('Error fetching roles:', error);
    return res.status(500).json({ success: false, message: 'Failed to fetch roles' });
  }
}

async function createRole(req, res, user) {
  try {
    const { name, permission, routes, description } = req.body;

    if (!name) {
      return res.status(400).json({ success: false, message: 'Role name is required' });
    }

    // Check if role already exists
    const existingRole = await Role.findOne({ name: name.toLowerCase() });
    if (existingRole) {
      return res.status(409).json({ success: false, message: 'Role already exists' });
    }

    const role = new Role({
      name: name.toLowerCase(),
      permission: permission || {
        create: false,
        read: true,
        update: false,
        delete: false
      },
      routes: routes || [],
      description: description || `${name} role`,
      isSystemRole: false
    });

    await role.save();

    return res.status(201).json({ 
      success: true, 
      data: {
        ...role.toObject(),
        displayName: role.displayName,
        userCount: 0
      }
    });
  } catch (error) {
    console.error('Error creating role:', error);
    return res.status(500).json({ success: false, message: 'Failed to create role' });
  }
}

async function updateRole(req, res, user) {
  try {
    const { id, permission, routes, description } = req.body;

    if (!id) {
      return res.status(400).json({ success: false, message: 'Role ID is required' });
    }

    const role = await Role.findById(id);
    if (!role) {
      return res.status(404).json({ success: false, message: 'Role not found' });
    }

    // Prevent modification of system roles' core permissions
    if (role.isSystemRole && permission) {
      return res.status(403).json({ 
        success: false, 
        message: 'Cannot modify permissions of system roles' 
      });
    }

    // Update role
    if (permission) role.permission = permission;
    if (routes !== undefined) role.routes = routes;
    if (description) role.description = description;

    await role.save();

    const userCount = await User.countDocuments({ role: role.name });

    return res.status(200).json({ 
      success: true, 
      data: {
        ...role.toObject(),
        displayName: role.displayName,
        userCount
      }
    });
  } catch (error) {
    console.error('Error updating role:', error);
    return res.status(500).json({ success: false, message: 'Failed to update role' });
  }
}

async function deleteRole(req, res, user) {
  try {
    const { id } = req.body;

    if (!id) {
      return res.status(400).json({ success: false, message: 'Role ID is required' });
    }

    const role = await Role.findById(id);
    if (!role) {
      return res.status(404).json({ success: false, message: 'Role not found' });
    }

    // Prevent deletion of system roles
    if (role.isSystemRole) {
      return res.status(403).json({ 
        success: false, 
        message: 'Cannot delete system roles' 
      });
    }

    // Check if role is assigned to any users
    const userCount = await User.countDocuments({ role: role.name });
    if (userCount > 0) {
      return res.status(409).json({ 
        success: false, 
        message: `Cannot delete role. ${userCount} users are assigned to this role.` 
      });
    }

    await Role.findByIdAndDelete(id);

    return res.status(200).json({ 
      success: true, 
      message: 'Role deleted successfully' 
    });
  } catch (error) {
    console.error('Error deleting role:', error);
    return res.status(500).json({ success: false, message: 'Failed to delete role' });
  }
}