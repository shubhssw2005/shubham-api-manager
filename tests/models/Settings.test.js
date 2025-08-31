import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import mongoose from 'mongoose';
import Settings from '../../models/Settings.js';
import { initializeDefaultSettings, getSetting, setSetting } from '../../lib/settings/index.js';

describe('Settings Model', () => {
  beforeEach(async () => {
    // Connect to test database
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/test');
    }
    
    // Clear settings collection
    await Settings.deleteMany({});
  });

  afterEach(async () => {
    // Clean up
    await Settings.deleteMany({});
  });

  it('should create a setting with valid data', async () => {
    const settingData = {
      key: 'test.setting',
      value: 'test value',
      type: 'string',
      category: 'system',
      description: 'Test setting',
      isPublic: true
    };

    const setting = new Settings(settingData);
    await setting.save();

    expect(setting._id).toBeDefined();
    expect(setting.key).toBe('test.setting');
    expect(setting.formattedValue).toBe('test value');
  });

  it('should validate setting value based on type', () => {
    expect(Settings.validateValue('test', 'string')).toBe(true);
    expect(Settings.validateValue(123, 'number')).toBe(true);
    expect(Settings.validateValue(true, 'boolean')).toBe(true);
    expect(Settings.validateValue([], 'array')).toBe(true);
    expect(Settings.validateValue({}, 'object')).toBe(true);
    
    expect(Settings.validateValue(123, 'string')).toBe(false);
    expect(Settings.validateValue('test', 'number')).toBe(false);
  });

  it('should get settings by category', async () => {
    await Settings.create([
      {
        key: 'system.name',
        value: 'Test System',
        type: 'string',
        category: 'system',
        isPublic: true
      },
      {
        key: 'api.rate_limit',
        value: 1000,
        type: 'number',
        category: 'api',
        isPublic: false
      }
    ]);

    const systemSettings = await Settings.getByCategory('system');
    expect(systemSettings).toHaveLength(1);
    expect(systemSettings[0].key).toBe('system.name');

    const apiSettings = await Settings.getByCategory('api', true);
    expect(apiSettings).toHaveLength(1);
    expect(apiSettings[0].key).toBe('api.rate_limit');
  });

  it('should initialize default settings', async () => {
    await initializeDefaultSettings();
    
    const settingsCount = await Settings.countDocuments();
    expect(settingsCount).toBeGreaterThan(0);
    
    const systemName = await getSetting('system.name');
    expect(systemName).toBe('API Gateway CMS');
  });

  it('should set and get setting values', async () => {
    // First initialize a setting
    await Settings.create({
      key: 'test.editable',
      value: 'initial',
      type: 'string',
      category: 'system',
      isEditable: true
    });

    await setSetting('test.editable', 'updated');
    const value = await getSetting('test.editable');
    expect(value).toBe('updated');
  });
});