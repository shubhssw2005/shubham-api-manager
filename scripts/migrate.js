#!/usr/bin/env node

import MigrationRunner from '../lib/migrations/runner.js';

const runner = new MigrationRunner();

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'up';
  const migrationName = args[1];

  try {
    switch (command) {
      case 'up':
        if (migrationName) {
          await runner.runSpecific(migrationName, 'up');
        } else {
          await runner.up();
        }
        break;
        
      case 'down':
        if (migrationName) {
          await runner.runSpecific(migrationName, 'down');
        } else {
          await runner.down();
        }
        break;
        
      default:
        console.log('Usage: node scripts/migrate.js [up|down] [migration_name]');
        console.log('  up   - Run migrations (default)');
        console.log('  down - Rollback migrations');
        console.log('  migration_name - Run specific migration (optional)');
        process.exit(1);
    }
    
    console.log('Migration completed successfully');
    process.exit(0);
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

main();