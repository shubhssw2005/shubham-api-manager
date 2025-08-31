// Test setup file
import { beforeAll, afterAll } from 'vitest';
import fs from 'fs/promises';
import path from 'path';

// Clean up test directories before and after tests
const testDir = path.join(process.cwd(), 'test-uploads');

beforeAll(async () => {
  try {
    await fs.rm(testDir, { recursive: true, force: true });
  } catch (error) {
    // Directory might not exist, ignore error
  }
});

afterAll(async () => {
  try {
    await fs.rm(testDir, { recursive: true, force: true });
  } catch (error) {
    // Directory might not exist, ignore error
  }
});