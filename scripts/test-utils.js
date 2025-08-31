const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

let mongod = null;

async function setupTestDatabase() {
  process.env.NODE_ENV = 'test';

  if (!mongod) {
    mongod = await MongoMemoryServer.create();
    const uri = mongod.getUri();
    global.TEST_MONGODB_URI = uri;
    console.log('Created MongoDB Memory Server URI:', uri);
  }

  try {
    if (mongoose.connection.readyState !== 1) {
      await mongoose.connect(global.TEST_MONGODB_URI, {
        bufferCommands: false,
        autoIndex: true,
        maxPoolSize: 10,
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 45000,
        family: 4
      });
      console.log('Connected to in-memory MongoDB');
    }
  } catch (error) {
    console.error('MongoDB connection error:', error);
    throw error;
  }
}

async function teardownTestDatabase() {
  try {
    if (mongoose.connection.readyState !== 0) {
      await mongoose.connection.close();
    }
    if (mongod) {
      await mongod.stop();
      mongod = null;
    }
    console.log('Disconnected from in-memory MongoDB');
  } catch (error) {
    console.error('MongoDB disconnection error:', error);
    throw error;
  }
}

module.exports = {
  setupTestDatabase,
  teardownTestDatabase
};
