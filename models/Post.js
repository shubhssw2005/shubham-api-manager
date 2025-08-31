import mongoose from 'mongoose';
import ModelFactory from '../lib/ModelFactory.js';

// Create Post model using Universal Data Management System
const Post = ModelFactory.createModel('Post', {
    title: {
        type: String,
        required: [true, 'Please provide a title'],
        trim: true,
        maxlength: [200, 'Title cannot be more than 200 characters']
    },
    slug: {
        type: String,
        lowercase: true,
        trim: true
    },
    content: {
        type: String,
        required: [true, 'Please provide content']
    },
    excerpt: {
        type: String,
        maxlength: [500, 'Excerpt cannot be more than 500 characters']
    },
    mediaIds: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Media'
    }],
    status: {
        type: String,
        enum: ['draft', 'published', 'archived', 'scheduled'],
        default: 'draft',
        index: true
    },
    publishedAt: {
        type: Date,
        index: true
    },
    tags: [{
        type: String,
        trim: true,
        lowercase: true
    }],
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
        index: true
    },
    viewCount: {
        type: Number,
        default: 0
    },
    likeCount: {
        type: Number,
        default: 0
    },
    featured: {
        type: Boolean,
        default: false,
        index: true
    },
    seoTitle: String,
    seoDescription: String,
    seoKeywords: [String]
}, {
    enableSoftDelete: true,
    enableEventSourcing: true,
    textSearchFields: ['title', 'content', 'excerpt', 'tags'],
    uniqueFields: ['slug'],
    indexes: [
        { fields: { status: 1, publishedAt: -1 } },
        { fields: { author: 1, status: 1 } },
        { fields: { tags: 1, status: 1 } },
        { fields: { featured: 1, publishedAt: -1 } }
    ]
});

export default Post;
