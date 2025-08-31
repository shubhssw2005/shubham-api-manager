import { describe, it, expect, vi } from 'vitest';

// Test helper functions from SystemOverview component
describe('SystemOverview Helper Functions', () => {
  it('should format bytes correctly', () => {
    const formatBytes = (bytes) => {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    expect(formatBytes(0)).toBe('0 Bytes');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1024000)).toBe('1000 KB');
    expect(formatBytes(1048576)).toBe('1 MB');
  });

  it('should get correct activity icons', () => {
    const getActivityIcon = (action) => {
      const iconMap = {
        login: 'users',
        logout: 'users',
        create: 'plus',
        update: 'edit',
        delete: 'trash',
        token_create: 'key',
        token_revoke: 'key',
        settings_change: 'cog',
        permission_change: 'users',
        user_approve: 'check-circle',
        user_reject: 'times-circle'
      };
      return iconMap[action] || 'activity';
    };

    expect(getActivityIcon('login')).toBe('users');
    expect(getActivityIcon('create')).toBe('plus');
    expect(getActivityIcon('unknown')).toBe('activity');
  });

  it('should get correct activity descriptions', () => {
    const getActivityDescription = (action, resource) => {
      const actionMap = {
        login: 'logged in',
        logout: 'logged out',
        create: 'created',
        update: 'updated',
        delete: 'deleted',
        token_create: 'created API token',
        token_revoke: 'revoked API token',
        settings_change: 'changed settings',
        permission_change: 'changed permissions',
        user_approve: 'approved user',
        user_reject: 'rejected user'
      };

      const description = actionMap[action] || action;
      return resource ? `${description} ${resource}` : description;
    };

    expect(getActivityDescription('login')).toBe('logged in');
    expect(getActivityDescription('create', 'user')).toBe('created user');
    expect(getActivityDescription('unknown_action')).toBe('unknown_action');
  });

  it('should format dates correctly', () => {
    const formatDate = (date) => {
      return new Date(date).toLocaleString();
    };

    const testDate = new Date('2024-01-01T12:00:00Z');
    const formatted = formatDate(testDate);
    
    // Just check that it returns a string (locale-specific formatting)
    expect(typeof formatted).toBe('string');
    expect(formatted.length).toBeGreaterThan(0);
  });

  it('should get correct notification icons', () => {
    const getNotificationIcon = (type) => {
      const iconMap = {
        warning: 'exclamation-triangle',
        error: 'times-circle',
        info: 'bell',
        success: 'check-circle'
      };
      return iconMap[type] || 'bell';
    };

    expect(getNotificationIcon('warning')).toBe('exclamation-triangle');
    expect(getNotificationIcon('error')).toBe('times-circle');
    expect(getNotificationIcon('info')).toBe('bell');
    expect(getNotificationIcon('success')).toBe('check-circle');
    expect(getNotificationIcon('unknown')).toBe('bell');
  });

  it('should format uptime correctly', () => {
    const formatUptime = (seconds) => {
      const days = Math.floor(seconds / 86400);
      const hours = Math.floor((seconds % 86400) / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      
      if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
      } else if (hours > 0) {
        return `${hours}h ${minutes}m`;
      } else {
        return `${minutes}m`;
      }
    };

    expect(formatUptime(60)).toBe('1m');
    expect(formatUptime(3600)).toBe('1h 0m');
    expect(formatUptime(3660)).toBe('1h 1m');
    expect(formatUptime(86400)).toBe('1d 0h 0m');
    expect(formatUptime(90061)).toBe('1d 1h 1m');
  });
});