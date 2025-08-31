import mongoose from 'mongoose';

/**
 * Universal Model Configuration
 * Define all your models here with consistent patterns
 */
export const modelsConfig = {
    // Blog/Post Models
    Post: {
        schema: {
            title: {
                type: String,
                required: [true, 'Please provide a title'],
                trim: true,
                maxlength: [200, 'Title cannot be more than 200 characters']
            },
            slug: {
                type: String,
                unique: true,
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
            categories: [{
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Category'
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
            commentCount: {
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
        },
        options: {
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
        }
    },

    // Product Models
    Product: {
        schema: {
            name: {
                type: String,
                required: [true, 'Product name is required'],
                trim: true,
                maxlength: [200, 'Name cannot be more than 200 characters']
            },
            slug: {
                type: String,
                unique: true,
                lowercase: true,
                trim: true
            },
            description: {
                type: String,
                required: [true, 'Product description is required']
            },
            shortDescription: {
                type: String,
                maxlength: [500, 'Short description cannot be more than 500 characters']
            },
            sku: {
                type: String,
                unique: true,
                uppercase: true,
                trim: true
            },
            barcode: String,
            category: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Category',
                required: true,
                index: true
            },
            brand: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Brand',
                index: true
            },
            price: {
                type: Number,
                required: [true, 'Price is required'],
                min: [0, 'Price cannot be negative']
            },
            compareAtPrice: {
                type: Number,
                min: [0, 'Compare price cannot be negative']
            },
            cost: {
                type: Number,
                min: [0, 'Cost cannot be negative']
            },
            inventory: {
                quantity: {
                    type: Number,
                    default: 0,
                    min: [0, 'Quantity cannot be negative']
                },
                trackQuantity: {
                    type: Boolean,
                    default: true
                },
                allowBackorder: {
                    type: Boolean,
                    default: false
                },
                lowStockThreshold: {
                    type: Number,
                    default: 10
                }
            },
            dimensions: {
                length: Number,
                width: Number,
                height: Number,
                weight: Number,
                unit: {
                    type: String,
                    enum: ['cm', 'in', 'mm'],
                    default: 'cm'
                }
            },
            images: [{
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Media'
            }],
            featuredImage: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Media'
            },
            status: {
                type: String,
                enum: ['draft', 'active', 'inactive', 'discontinued'],
                default: 'draft',
                index: true
            },
            visibility: {
                type: String,
                enum: ['public', 'private', 'hidden'],
                default: 'public',
                index: true
            },
            featured: {
                type: Boolean,
                default: false,
                index: true
            },
            tags: [{
                type: String,
                trim: true,
                lowercase: true
            }],
            attributes: {
                type: Map,
                of: mongoose.Schema.Types.Mixed
            },
            variants: [{
                name: String,
                sku: String,
                price: Number,
                inventory: {
                    quantity: Number,
                    trackQuantity: Boolean
                },
                attributes: {
                    type: Map,
                    of: String
                }
            }],
            seoTitle: String,
            seoDescription: String,
            seoKeywords: [String],
            rating: {
                average: {
                    type: Number,
                    default: 0,
                    min: 0,
                    max: 5
                },
                count: {
                    type: Number,
                    default: 0
                }
            },
            salesCount: {
                type: Number,
                default: 0
            },
            viewCount: {
                type: Number,
                default: 0
            }
        },
        options: {
            enableSoftDelete: true,
            enableEventSourcing: true,
            textSearchFields: ['name', 'description', 'shortDescription', 'tags', 'sku'],
            uniqueFields: ['slug', 'sku'],
            indexes: [
                { fields: { category: 1, status: 1 } },
                { fields: { brand: 1, status: 1 } },
                { fields: { price: 1, status: 1 } },
                { fields: { featured: 1, status: 1 } },
                { fields: { 'inventory.quantity': 1 } },
                { fields: { rating: -1, status: 1 } }
            ]
        }
    },

    // Category Model
    Category: {
        schema: {
            name: {
                type: String,
                required: [true, 'Category name is required'],
                trim: true,
                maxlength: [100, 'Name cannot be more than 100 characters']
            },
            slug: {
                type: String,
                unique: true,
                lowercase: true,
                trim: true
            },
            description: String,
            parent: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Category',
                index: true
            },
            image: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Media'
            },
            isActive: {
                type: Boolean,
                default: true,
                index: true
            },
            sortOrder: {
                type: Number,
                default: 0
            },
            seoTitle: String,
            seoDescription: String,
            seoKeywords: [String]
        },
        options: {
            enableSoftDelete: true,
            enableEventSourcing: true,
            textSearchFields: ['name', 'description'],
            uniqueFields: ['slug'],
            indexes: [
                { fields: { parent: 1, isActive: 1 } },
                { fields: { sortOrder: 1, isActive: 1 } }
            ]
        }
    },

    // Order Model
    Order: {
        schema: {
            orderNumber: {
                type: String,
                unique: true,
                required: true
            },
            customer: {
                type: mongoose.Schema.Types.ObjectId,
                ref: 'Customer',
                required: true,
                index: true
            },
            items: [{
                product: {
                    type: mongoose.Schema.Types.ObjectId,
                    ref: 'Product',
                    required: true
                },
                variant: {
                    type: mongoose.Schema.Types.ObjectId
                },
                quantity: {
                    type: Number,
                    required: true,
                    min: 1
                },
                price: {
                    type: Number,
                    required: true
                },
                total: {
                    type: Number,
                    required: true
                }
            }],
            subtotal: {
                type: Number,
                required: true
            },
            tax: {
                type: Number,
                default: 0
            },
            shipping: {
                type: Number,
                default: 0
            },
            discount: {
                type: Number,
                default: 0
            },
            total: {
                type: Number,
                required: true
            },
            currency: {
                type: String,
                default: 'USD',
                uppercase: true
            },
            status: {
                type: String,
                enum: ['pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled', 'refunded'],
                default: 'pending',
                index: true
            },
            paymentStatus: {
                type: String,
                enum: ['pending', 'paid', 'failed', 'refunded', 'partially_refunded'],
                default: 'pending',
                index: true
            },
            fulfillmentStatus: {
                type: String,
                enum: ['unfulfilled', 'partial', 'fulfilled'],
                default: 'unfulfilled',
                index: true
            },
            shippingAddress: {
                firstName: String,
                lastName: String,
                company: String,
                address1: String,
                address2: String,
                city: String,
                state: String,
                postalCode: String,
                country: String,
                phone: String
            },
            billingAddress: {
                firstName: String,
                lastName: String,
                company: String,
                address1: String,
                address2: String,
                city: String,
                state: String,
                postalCode: String,
                country: String,
                phone: String
            },
            paymentMethod: String,
            paymentId: String,
            shippingMethod: String,
            trackingNumber: String,
            notes: String,
            customerNotes: String
        },
        options: {
            enableSoftDelete: true,
            enableEventSourcing: true,
            textSearchFields: ['orderNumber', 'customerNotes', 'notes'],
            uniqueFields: ['orderNumber'],
            indexes: [
                { fields: { customer: 1, status: 1 } },
                { fields: { status: 1, createdAt: -1 } },
                { fields: { paymentStatus: 1, createdAt: -1 } },
                { fields: { total: -1, createdAt: -1 } }
            ]
        }
    },

    // Customer Model
    Customer: {
        schema: {
            email: {
                type: String,
                required: [true, 'Email is required'],
                unique: true,
                lowercase: true,
                trim: true
            },
            firstName: {
                type: String,
                required: [true, 'First name is required'],
                trim: true
            },
            lastName: {
                type: String,
                required: [true, 'Last name is required'],
                trim: true
            },
            phone: String,
            dateOfBirth: Date,
            gender: {
                type: String,
                enum: ['male', 'female', 'other', 'prefer_not_to_say']
            },
            addresses: [{
                type: {
                    type: String,
                    enum: ['shipping', 'billing'],
                    required: true
                },
                firstName: String,
                lastName: String,
                company: String,
                address1: String,
                address2: String,
                city: String,
                state: String,
                postalCode: String,
                country: String,
                phone: String,
                isDefault: {
                    type: Boolean,
                    default: false
                }
            }],
            isActive: {
                type: Boolean,
                default: true,
                index: true
            },
            emailVerified: {
                type: Boolean,
                default: false
            },
            phoneVerified: {
                type: Boolean,
                default: false
            },
            acceptsMarketing: {
                type: Boolean,
                default: false
            },
            tags: [String],
            notes: String,
            totalSpent: {
                type: Number,
                default: 0
            },
            orderCount: {
                type: Number,
                default: 0
            },
            lastOrderAt: Date
        },
        options: {
            enableSoftDelete: true,
            enableEventSourcing: true,
            textSearchFields: ['email', 'firstName', 'lastName', 'phone'],
            uniqueFields: ['email'],
            indexes: [
                { fields: { isActive: 1, createdAt: -1 } },
                { fields: { totalSpent: -1 } },
                { fields: { orderCount: -1 } },
                { fields: { tags: 1 } }
            ]
        }
    }
};

export default modelsConfig;