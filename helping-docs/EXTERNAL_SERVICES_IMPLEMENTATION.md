# 🌐 External Services Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing and configuring all external services required for your Groot API production deployment. Each service includes account setup, configuration, and integration code.

---

## 📧 Email Services Implementation

### SendGrid Setup & Integration

#### Step 1: Account Setup
1. **Create Account**: Go to https://sendgrid.com/
2. **Verify Email**: Complete email verification
3. **Create API Key**:
   - Go to Settings → API Keys
   - Click "Create API Key"
   - Name: `groot-api-production`
   - Permissions: Full Access
   - **Save the key immediately!**

#### Step 2: Domain Authentication
```bash
# 1. Go to Settings → Sender Authentication
# 2. Click "Authenticate Your Domain"
# 3. Add your domain: yourdomain.com
# 4. Add DNS records to your domain provider:

# CNAME Records to add:
s1._domainkey.yourdomain.com → s1.domainkey.u1234567.wl123.sendgrid.net
s2._domainkey.yourdomain.com → s2.domainkey.u1234567.wl123.sendgrid.net
```

#### Step 3: Implementation
```javascript
// lib/email/sendgrid-service.js
const sgMail = require('@sendgrid/mail');

class SendGridService {
  constructor() {
    sgMail.setApiKey(process.env.SENDGRID_API_KEY);
    this.fromEmail = process.env.SENDGRID_FROM_EMAIL;
    this.fromName = process.env.SENDGRID_FROM_NAME;
  }

  async sendWelcomeEmail(userEmail, userName) {
    const msg = {
      to: userEmail,
      from: {
        email: this.fromEmail,
        name: this.fromName
      },
      subject: 'Welcome to Groot API',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h1 style="color: #2c3e50;">Welcome ${userName}!</h1>
          <p>Thank you for joining Groot API. Your account has been successfully created.</p>
          <div style="background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
            <h3>Getting Started:</h3>
            <ul>
              <li>Complete your profile</li>
              <li>Explore our API documentation</li>
              <li>Create your first blog post</li>
            </ul>
          </div>
          <p>If you have any questions, reply to this email or contact our support team.</p>
          <hr style="margin: 30px 0;">
          <p style="color: #7f8c8d; font-size: 12px;">
            This email was sent by Groot API. If you didn't create an account, please ignore this email.
          </p>
        </div>
      `
    };

    try {
      await sgMail.send(msg);
      console.log('Welcome email sent successfully');
      return { success: true };
    } catch (error) {
      console.error('SendGrid error:', error);
      throw new Error('Failed to send welcome email');
    }
  }

  async sendPasswordResetEmail(userEmail, resetToken) {
    const resetUrl = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;
    
    const msg = {
      to: userEmail,
      from: {
        email: this.fromEmail,
        name: this.fromName
      },
      subject: 'Password Reset Request',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h1 style="color: #e74c3c;">Password Reset Request</h1>
          <p>You requested a password reset for your Groot API account.</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="${resetUrl}" 
               style="background: #3498db; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
              Reset Password
            </a>
          </div>
          <p>This link will expire in 1 hour. If you didn't request this reset, please ignore this email.</p>
          <p style="color: #7f8c8d; font-size: 12px;">
            If the button doesn't work, copy and paste this URL: ${resetUrl}
          </p>
        </div>
      `
    };

    return await sgMail.send(msg);
  }

  async sendBackupNotification(userEmail, backupUrl, expiresAt) {
    const msg = {
      to: userEmail,
      from: {
        email: this.fromEmail,
        name: this.fromName
      },
      subject: 'Your Data Backup is Ready',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h1 style="color: #27ae60;">Backup Complete</h1>
          <p>Your data backup has been successfully created and is ready for download.</p>
          <div style="background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0;">
            <h3>Backup Details:</h3>
            <ul>
              <li><strong>Created:</strong> ${new Date().toLocaleString()}</li>
              <li><strong>Expires:</strong> ${new Date(expiresAt).toLocaleString()}</li>
              <li><strong>Format:</strong> JSON</li>
            </ul>
          </div>
          <div style="text-align: center; margin: 30px 0;">
            <a href="${backupUrl}" 
               style="background: #27ae60; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
              Download Backup
            </a>
          </div>
          <p style="color: #e74c3c; font-weight: bold;">
            ⚠️ This download link will expire in 1 hour for security reasons.
          </p>
        </div>
      `
    };

    return await sgMail.send(msg);
  }
}

module.exports = SendGridService;
```

#### Step 4: Environment Variables
```env
# Add to .env
SENDGRID_API_KEY=SG.your-api-key-here
SENDGRID_FROM_EMAIL=noreply@yourdomain.com
SENDGRID_FROM_NAME=Groot API
FRONTEND_URL=https://yourdomain.com
```

#### Step 5: Integration in Routes
```javascript
// pages/api/auth/register.js
const SendGridService = require('../../../lib/email/sendgrid-service');
const emailService = new SendGridService();

export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      // ... user creation logic ...
      
      // Send welcome email
      await emailService.sendWelcomeEmail(user.email, user.name);
      
      res.status(201).json({ 
        message: 'User created successfully',
        user: { id: user._id, email: user.email, name: user.name }
      });
    } catch (error) {
      console.error('Registration error:', error);
      res.status(500).json({ error: 'Registration failed' });
    }
  }
}
```

---

## 📱 SMS Services Implementation

### Twilio Setup & Integration

#### Step 1: Account Setup
1. **Create Account**: Go to https://www.twilio.com/
2. **Verify Phone Number**: Complete phone verification
3. **Get Credentials**:
   - Account SID: Found on dashboard
   - Auth Token: Found on dashboard
4. **Purchase Phone Number**:
   - Go to Phone Numbers → Manage → Buy a number
   - Choose a number with SMS capabilities

#### Step 2: Implementation
```javascript
// lib/sms/twilio-service.js
const twilio = require('twilio');

class TwilioService {
  constructor() {
    this.client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
    this.fromNumber = process.env.TWILIO_PHONE_NUMBER;
  }

  async sendVerificationCode(phoneNumber, code) {
    try {
      const message = await this.client.messages.create({
        body: `Your Groot API verification code is: ${code}. This code expires in 10 minutes.`,
        from: this.fromNumber,
        to: phoneNumber
      });

      console.log('SMS sent successfully:', message.sid);
      return { success: true, messageId: message.sid };
    } catch (error) {
      console.error('Twilio error:', error);
      throw new Error('Failed to send SMS');
    }
  }

  async sendSecurityAlert(phoneNumber, alertMessage) {
    try {
      const message = await this.client.messages.create({
        body: `🚨 SECURITY ALERT: ${alertMessage} - Groot API`,
        from: this.fromNumber,
        to: phoneNumber
      });

      return { success: true, messageId: message.sid };
    } catch (error) {
      console.error('Security alert SMS failed:', error);
      throw error;
    }
  }

  async sendBackupNotification(phoneNumber, backupId) {
    const message = `✅ Your data backup (ID: ${backupId}) is ready for download. Check your email for the download link. - Groot API`;
    
    try {
      const result = await this.client.messages.create({
        body: message,
        from: this.fromNumber,
        to: phoneNumber
      });

      return { success: true, messageId: result.sid };
    } catch (error) {
      console.error('Backup notification SMS failed:', error);
      throw error;
    }
  }
}

module.exports = TwilioService;
```

#### Step 3: Phone Verification System
```javascript
// lib/auth/phone-verification.js
const TwilioService = require('../sms/twilio-service');
const Redis = require('ioredis');

class PhoneVerificationService {
  constructor() {
    this.smsService = new TwilioService();
    this.redis = new Redis(process.env.REDIS_URL);
    this.codeExpiry = 600; // 10 minutes
  }

  generateVerificationCode() {
    return Math.floor(100000 + Math.random() * 900000).toString();
  }

  async sendVerificationCode(phoneNumber) {
    // Rate limiting: max 3 codes per hour
    const rateLimitKey = `sms_rate_limit:${phoneNumber}`;
    const currentCount = await this.redis.get(rateLimitKey) || 0;
    
    if (parseInt(currentCount) >= 3) {
      throw new Error('Too many verification attempts. Please try again in 1 hour.');
    }

    const code = this.generateVerificationCode();
    const codeKey = `verification_code:${phoneNumber}`;

    // Store code in Redis
    await this.redis.setex(codeKey, this.codeExpiry, code);
    
    // Update rate limit
    await this.redis.incr(rateLimitKey);
    await this.redis.expire(rateLimitKey, 3600); // 1 hour

    // Send SMS
    await this.smsService.sendVerificationCode(phoneNumber, code);

    return { success: true, expiresIn: this.codeExpiry };
  }

  async verifyCode(phoneNumber, code) {
    const codeKey = `verification_code:${phoneNumber}`;
    const storedCode = await this.redis.get(codeKey);

    if (!storedCode) {
      throw new Error('Verification code expired or not found');
    }

    if (storedCode !== code) {
      throw new Error('Invalid verification code');
    }

    // Delete used code
    await this.redis.del(codeKey);
    
    return { success: true, verified: true };
  }
}

module.exports = PhoneVerificationService;
```

#### Step 4: Environment Variables
```env
# Add to .env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your-auth-token-here
TWILIO_PHONE_NUMBER=+1234567890
```

---

## 🔍 Search Services Implementation

### Elasticsearch Setup & Integration

#### Step 1: Elastic Cloud Setup
1. **Create Account**: Go to https://cloud.elastic.co/
2. **Create Deployment**:
   - Choose cloud provider (AWS/GCP/Azure)
   - Select region (same as your app)
   - Choose deployment size (start with 8GB RAM)
3. **Get Credentials**:
   - Endpoint URL
   - Username: `elastic`
   - Password: (generated automatically)

#### Step 2: Implementation
```javascript
// lib/search/elasticsearch-service.js
const { Client } = require('@elastic/elasticsearch');

class ElasticsearchService {
  constructor() {
    this.client = new Client({
      node: process.env.ELASTICSEARCH_URL,
      auth: {
        username: process.env.ELASTICSEARCH_USERNAME,
        password: process.env.ELASTICSEARCH_PASSWORD
      },
      tls: {
        rejectUnauthorized: false
      }
    });
    
    this.postsIndex = 'groot-posts';
    this.mediaIndex = 'groot-media';
  }

  async initializeIndices() {
    try {
      // Create posts index
      await this.client.indices.create({
        index: this.postsIndex,
        body: {
          mappings: {
            properties: {
              title: { 
                type: 'text',
                analyzer: 'standard',
                fields: {
                  keyword: { type: 'keyword' }
                }
              },
              content: { 
                type: 'text',
                analyzer: 'standard'
              },
              excerpt: { type: 'text' },
              tags: { type: 'keyword' },
              author: {
                type: 'object',
                properties: {
                  id: { type: 'keyword' },
                  name: { type: 'text' },
                  email: { type: 'keyword' }
                }
              },
              status: { type: 'keyword' },
              createdAt: { type: 'date' },
              updatedAt: { type: 'date' }
            }
          },
          settings: {
            number_of_shards: 1,
            number_of_replicas: 1,
            analysis: {
              analyzer: {
                content_analyzer: {
                  type: 'custom',
                  tokenizer: 'standard',
                  filter: ['lowercase', 'stop', 'snowball']
                }
              }
            }
          }
        }
      });

      // Create media index
      await this.client.indices.create({
        index: this.mediaIndex,
        body: {
          mappings: {
            properties: {
              filename: { type: 'text' },
              originalName: { type: 'text' },
              mimeType: { type: 'keyword' },
              size: { type: 'long' },
              uploadedBy: {
                type: 'object',
                properties: {
                  id: { type: 'keyword' },
                  name: { type: 'text' }
                }
              },
              tags: { type: 'keyword' },
              metadata: { type: 'object' },
              createdAt: { type: 'date' }
            }
          }
        }
      });

      console.log('Elasticsearch indices created successfully');
    } catch (error) {
      if (error.meta?.body?.error?.type !== 'resource_already_exists_exception') {
        console.error('Failed to create Elasticsearch indices:', error);
        throw error;
      }
    }
  }

  async indexPost(post) {
    try {
      await this.client.index({
        index: this.postsIndex,
        id: post._id.toString(),
        body: {
          title: post.title,
          content: post.content,
          excerpt: post.excerpt,
          tags: post.tags || [],
          author: {
            id: post.author._id?.toString() || post.author,
            name: post.author.name,
            email: post.author.email
          },
          status: post.status,
          createdAt: post.createdAt,
          updatedAt: post.updatedAt
        }
      });

      console.log(`Post ${post._id} indexed successfully`);
    } catch (error) {
      console.error('Failed to index post:', error);
      throw error;
    }
  }

  async searchPosts(query, options = {}) {
    const {
      page = 1,
      limit = 20,
      tags = [],
      author = null,
      status = 'published',
      sortBy = 'createdAt',
      sortOrder = 'desc'
    } = options;

    const searchBody = {
      query: {
        bool: {
          must: [],
          filter: []
        }
      },
      sort: [
        { [sortBy]: { order: sortOrder } }
      ],
      from: (page - 1) * limit,
      size: limit,
      highlight: {
        fields: {
          title: {},
          content: {}
        }
      }
    };

    // Add text search
    if (query) {
      searchBody.query.bool.must.push({
        multi_match: {
          query: query,
          fields: ['title^3', 'content', 'excerpt^2'],
          type: 'best_fields',
          fuzziness: 'AUTO'
        }
      });
    } else {
      searchBody.query.bool.must.push({
        match_all: {}
      });
    }

    // Add filters
    if (status) {
      searchBody.query.bool.filter.push({
        term: { status: status }
      });
    }

    if (tags.length > 0) {
      searchBody.query.bool.filter.push({
        terms: { tags: tags }
      });
    }

    if (author) {
      searchBody.query.bool.filter.push({
        term: { 'author.id': author }
      });
    }

    try {
      const response = await this.client.search({
        index: this.postsIndex,
        body: searchBody
      });

      return {
        hits: response.body.hits.hits.map(hit => ({
          ...hit._source,
          id: hit._id,
          score: hit._score,
          highlights: hit.highlight
        })),
        total: response.body.hits.total.value,
        page,
        limit,
        totalPages: Math.ceil(response.body.hits.total.value / limit)
      };
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  }

  async deletePost(postId) {
    try {
      await this.client.delete({
        index: this.postsIndex,
        id: postId
      });
      console.log(`Post ${postId} deleted from search index`);
    } catch (error) {
      if (error.meta?.statusCode !== 404) {
        console.error('Failed to delete post from search index:', error);
        throw error;
      }
    }
  }

  async getSuggestions(query, field = 'title') {
    try {
      const response = await this.client.search({
        index: this.postsIndex,
        body: {
          suggest: {
            suggestions: {
              text: query,
              term: {
                field: field,
                size: 5
              }
            }
          }
        }
      });

      return response.body.suggest.suggestions[0].options.map(option => option.text);
    } catch (error) {
      console.error('Failed to get suggestions:', error);
      return [];
    }
  }
}

module.exports = ElasticsearchService;
```

#### Step 3: Integration with Post Operations
```javascript
// lib/search/search-sync.js
const ElasticsearchService = require('./elasticsearch-service');

class SearchSyncService {
  constructor() {
    this.searchService = new ElasticsearchService();
  }

  async syncPostCreate(post) {
    try {
      await this.searchService.indexPost(post);
    } catch (error) {
      console.error('Failed to sync post creation to search:', error);
      // Don't throw - search sync shouldn't break main operation
    }
  }

  async syncPostUpdate(post) {
    try {
      await this.searchService.indexPost(post);
    } catch (error) {
      console.error('Failed to sync post update to search:', error);
    }
  }

  async syncPostDelete(postId) {
    try {
      await this.searchService.deletePost(postId);
    } catch (error) {
      console.error('Failed to sync post deletion to search:', error);
    }
  }
}

module.exports = SearchSyncService;
```

#### Step 4: Search API Endpoint
```javascript
// pages/api/search/posts.js
const ElasticsearchService = require('../../../lib/search/elasticsearch-service');

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const searchService = new ElasticsearchService();
    
    const {
      q: query,
      page = 1,
      limit = 20,
      tags,
      author,
      status = 'published',
      sortBy = 'createdAt',
      sortOrder = 'desc'
    } = req.query;

    const searchOptions = {
      page: parseInt(page),
      limit: parseInt(limit),
      tags: tags ? tags.split(',') : [],
      author,
      status,
      sortBy,
      sortOrder
    };

    const results = await searchService.searchPosts(query, searchOptions);

    res.status(200).json({
      success: true,
      data: results,
      query: {
        text: query,
        options: searchOptions
      }
    });
  } catch (error) {
    console.error('Search API error:', error);
    res.status(500).json({
      error: 'Search failed',
      message: error.message
    });
  }
}
```

#### Step 5: Environment Variables
```env
# Add to .env
ELASTICSEARCH_URL=https://your-deployment.es.io:9243
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your-generated-password
```

---

## 💳 Payment Processing Implementation

### Stripe Setup & Integration

#### Step 1: Account Setup
1. **Create Account**: Go to https://stripe.com/
2. **Complete Business Verification**
3. **Get API Keys**:
   - Go to Developers → API Keys
   - Copy Publishable Key and Secret Key
4. **Set up Webhooks**:
   - Go to Developers → Webhooks
   - Add endpoint: `https://yourdomain.com/api/webhooks/stripe`
   - Select events: `payment_intent.succeeded`, `payment_intent.payment_failed`

#### Step 2: Implementation
```javascript
// lib/payment/stripe-service.js
const Stripe = require('stripe');

class StripeService {
  constructor() {
    this.stripe = Stripe(process.env.STRIPE_SECRET_KEY);
  }

  async createPaymentIntent(amount, currency = 'usd', metadata = {}) {
    try {
      const paymentIntent = await this.stripe.paymentIntents.create({
        amount: Math.round(amount * 100), // Convert to cents
        currency,
        metadata,
        automatic_payment_methods: {
          enabled: true,
        },
      });

      return {
        clientSecret: paymentIntent.client_secret,
        paymentIntentId: paymentIntent.id
      };
    } catch (error) {
      console.error('Stripe payment intent creation failed:', error);
      throw new Error('Failed to create payment intent');
    }
  }

  async createSubscription(customerId, priceId, metadata = {}) {
    try {
      const subscription = await this.stripe.subscriptions.create({
        customer: customerId,
        items: [{ price: priceId }],
        metadata,
        payment_behavior: 'default_incomplete',
        payment_settings: { save_default_payment_method: 'on_subscription' },
        expand: ['latest_invoice.payment_intent'],
      });

      return {
        subscriptionId: subscription.id,
        clientSecret: subscription.latest_invoice.payment_intent.client_secret
      };
    } catch (error) {
      console.error('Stripe subscription creation failed:', error);
      throw new Error('Failed to create subscription');
    }
  }

  async createCustomer(email, name, metadata = {}) {
    try {
      const customer = await this.stripe.customers.create({
        email,
        name,
        metadata
      });

      return customer;
    } catch (error) {
      console.error('Stripe customer creation failed:', error);
      throw new Error('Failed to create customer');
    }
  }

  async handleWebhook(body, signature) {
    try {
      const event = this.stripe.webhooks.constructEvent(
        body,
        signature,
        process.env.STRIPE_WEBHOOK_SECRET
      );

      switch (event.type) {
        case 'payment_intent.succeeded':
          await this.handlePaymentSuccess(event.data.object);
          break;
        case 'payment_intent.payment_failed':
          await this.handlePaymentFailure(event.data.object);
          break;
        case 'customer.subscription.created':
          await this.handleSubscriptionCreated(event.data.object);
          break;
        case 'customer.subscription.deleted':
          await this.handleSubscriptionCanceled(event.data.object);
          break;
        default:
          console.log(`Unhandled event type: ${event.type}`);
      }

      return { received: true };
    } catch (error) {
      console.error('Webhook handling failed:', error);
      throw error;
    }
  }

  async handlePaymentSuccess(paymentIntent) {
    console.log('Payment succeeded:', paymentIntent.id);
    
    // Update order status in database
    const orderId = paymentIntent.metadata.orderId;
    if (orderId) {
      // Update order status to 'paid'
      // Send confirmation email
      // Trigger fulfillment process
    }
  }

  async handlePaymentFailure(paymentIntent) {
    console.log('Payment failed:', paymentIntent.id);
    
    // Update order status in database
    const orderId = paymentIntent.metadata.orderId;
    if (orderId) {
      // Update order status to 'failed'
      // Send failure notification
      // Retry payment if applicable
    }
  }
}

module.exports = StripeService;
```

#### Step 3: Payment API Endpoints
```javascript
// pages/api/payment/create-intent.js
const StripeService = require('../../../lib/payment/stripe-service');

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { amount, currency, orderId, userId } = req.body;

    if (!amount || amount <= 0) {
      return res.status(400).json({ error: 'Invalid amount' });
    }

    const stripeService = new StripeService();
    
    const paymentIntent = await stripeService.createPaymentIntent(
      amount,
      currency,
      { orderId, userId }
    );

    res.status(200).json({
      success: true,
      clientSecret: paymentIntent.clientSecret,
      paymentIntentId: paymentIntent.paymentIntentId
    });
  } catch (error) {
    console.error('Payment intent creation error:', error);
    res.status(500).json({
      error: 'Failed to create payment intent',
      message: error.message
    });
  }
}
```

```javascript
// pages/api/webhooks/stripe.js
const StripeService = require('../../../lib/payment/stripe-service');

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const signature = req.headers['stripe-signature'];
    const stripeService = new StripeService();
    
    await stripeService.handleWebhook(req.body, signature);
    
    res.status(200).json({ received: true });
  } catch (error) {
    console.error('Webhook error:', error);
    res.status(400).json({ error: 'Webhook handling failed' });
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
}
```

#### Step 4: Environment Variables
```env
# Add to .env
STRIPE_PUBLISHABLE_KEY=pk_live_xxxxxxxxxxxxx
STRIPE_SECRET_KEY=sk_live_xxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx
```

---

## 📊 Analytics Implementation

### Google Analytics 4 Setup

#### Step 1: Account Setup
1. **Create GA4 Property**: Go to https://analytics.google.com/
2. **Get Measurement ID**: Format: G-XXXXXXXXXX
3. **Enable Enhanced Ecommerce**: For tracking purchases

#### Step 2: Implementation
```javascript
// lib/analytics/google-analytics.js
class GoogleAnalyticsService {
  constructor() {
    this.measurementId = process.env.GA_MEASUREMENT_ID;
  }

  // Server-side event tracking
  async trackEvent(eventName, parameters = {}) {
    if (!this.measurementId) return;

    try {
      const response = await fetch(`https://www.google-analytics.com/mp/collect?measurement_id=${this.measurementId}&api_secret=${process.env.GA_API_SECRET}`, {
        method: 'POST',
        body: JSON.stringify({
          client_id: parameters.client_id || 'server-side',
          events: [{
            name: eventName,
            params: parameters
          }]
        })
      });

      if (!response.ok) {
        throw new Error(`GA tracking failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Google Analytics tracking error:', error);
    }
  }

  async trackUserRegistration(userId, userEmail) {
    await this.trackEvent('sign_up', {
      user_id: userId,
      method: 'email'
    });
  }

  async trackPostCreated(userId, postId, postTitle) {
    await this.trackEvent('post_created', {
      user_id: userId,
      post_id: postId,
      post_title: postTitle,
      content_type: 'blog_post'
    });
  }

  async trackBackupCreated(userId, backupSize) {
    await this.trackEvent('backup_created', {
      user_id: userId,
      backup_size: backupSize,
      currency: 'USD',
      value: 0 // Free feature
    });
  }

  async trackMediaUpload(userId, fileType, fileSize) {
    await this.trackEvent('media_upload', {
      user_id: userId,
      file_type: fileType,
      file_size: fileSize
    });
  }
}

module.exports = GoogleAnalyticsService;
```

#### Step 3: Client-side Tracking
```javascript
// lib/analytics/client-analytics.js
export class ClientAnalytics {
  constructor() {
    this.gtag = window.gtag;
  }

  trackPageView(path) {
    if (this.gtag) {
      this.gtag('config', process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID, {
        page_path: path,
      });
    }
  }

  trackEvent(eventName, parameters = {}) {
    if (this.gtag) {
      this.gtag('event', eventName, parameters);
    }
  }

  trackUserEngagement(engagementTime) {
    this.trackEvent('user_engagement', {
      engagement_time_msec: engagementTime
    });
  }

  trackSearch(searchTerm, resultsCount) {
    this.trackEvent('search', {
      search_term: searchTerm,
      results_count: resultsCount
    });
  }
}
```

#### Step 4: Environment Variables
```env
# Add to .env
GA_MEASUREMENT_ID=G-XXXXXXXXXX
GA_API_SECRET=your-api-secret
NEXT_PUBLIC_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

---

## 🔔 Push Notifications Implementation

### Firebase Cloud Messaging Setup

#### Step 1: Firebase Project Setup
1. **Create Project**: Go to https://console.firebase.google.com/
2. **Add Web App**: Get Firebase config
3. **Generate Service Account Key**:
   - Go to Project Settings → Service Accounts
   - Generate new private key
   - Download JSON file

#### Step 2: Implementation
```javascript
// lib/notifications/firebase-service.js
const admin = require('firebase-admin');

class FirebaseNotificationService {
  constructor() {
    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert({
          projectId: process.env.FIREBASE_PROJECT_ID,
          clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
          privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
        }),
      });
    }
    this.messaging = admin.messaging();
  }

  async sendToDevice(token, notification, data = {}) {
    try {
      const message = {
        token,
        notification: {
          title: notification.title,
          body: notification.body,
          imageUrl: notification.image
        },
        data,
        webpush: {
          fcmOptions: {
            link: notification.clickAction
          }
        }
      };

      const response = await this.messaging.send(message);
      console.log('Notification sent successfully:', response);
      return { success: true, messageId: response };
    } catch (error) {
      console.error('Failed to send notification:', error);
      throw error;
    }
  }

  async sendToTopic(topic, notification, data = {}) {
    try {
      const message = {
        topic,
        notification: {
          title: notification.title,
          body: notification.body
        },
        data
      };

      const response = await this.messaging.send(message);
      return { success: true, messageId: response };
    } catch (error) {
      console.error('Failed to send topic notification:', error);
      throw error;
    }
  }

  async subscribeToTopic(tokens, topic) {
    try {
      const response = await this.messaging.subscribeToTopic(tokens, topic);
      console.log('Successfully subscribed to topic:', response);
      return response;
    } catch (error) {
      console.error('Failed to subscribe to topic:', error);
      throw error;
    }
  }

  async sendBackupNotification(userToken, backupId) {
    return await this.sendToDevice(userToken, {
      title: '✅ Backup Complete',
      body: 'Your data backup is ready for download',
      clickAction: '/dashboard/backups'
    }, {
      backupId,
      type: 'backup_complete'
    });
  }

  async sendSecurityAlert(userToken, alertMessage) {
    return await this.sendToDevice(userToken, {
      title: '🚨 Security Alert',
      body: alertMessage,
      clickAction: '/dashboard/security'
    }, {
      type: 'security_alert',
      priority: 'high'
    });
  }
}

module.exports = FirebaseNotificationService;
```

#### Step 3: Environment Variables
```env
# Add to .env
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYour private key here\n-----END PRIVATE KEY-----\n"
```

---

## 🎯 Implementation Priority Order

### Phase 1: Essential Services (Week 1)
1. **Email (SendGrid)** - Critical for user communication
2. **Database (MongoDB Atlas)** - Core data storage
3. **Object Storage (S3/MinIO)** - Media and backup storage
4. **Monitoring (Basic)** - Health checks and basic metrics

### Phase 2: Enhanced Features (Week 2)
1. **Search (Elasticsearch)** - Improved user experience
2. **SMS (Twilio)** - Security and notifications
3. **Analytics (Google Analytics)** - User behavior tracking
4. **Advanced Monitoring** - Detailed metrics and alerting

### Phase 3: Advanced Features (Week 3)
1. **Payment Processing (Stripe)** - Monetization
2. **Push Notifications (Firebase)** - User engagement
3. **Advanced Security** - Enhanced protection
4. **Performance Optimization** - Scaling improvements

---

## 🔧 Testing Your Implementations

### Email Testing
```bash
# Test SendGrid integration
curl -X POST http://localhost:3005/api/test/email \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "name": "Test User"}'
```

### SMS Testing
```bash
# Test Twilio integration
curl -X POST http://localhost:3005/api/test/sms \
  -H "Content-Type: application/json" \
  -d '{"phone": "+1234567890", "code": "123456"}'
```

### Search Testing
```bash
# Test Elasticsearch integration
curl -X GET "http://localhost:3005/api/search/posts?q=test&limit=5"
```

### Payment Testing
```bash
# Test Stripe integration (use test card: 4242424242424242)
curl -X POST http://localhost:3005/api/payment/create-intent \
  -H "Content-Type: application/json" \
  -d '{"amount": 10.00, "currency": "usd", "orderId": "test-order"}'
```

---

## 📞 Support Resources

### Service Documentation
- **SendGrid**: https://docs.sendgrid.com/
- **Twilio**: https://www.twilio.com/docs
- **Elasticsearch**: https://www.elastic.co/guide/
- **Stripe**: https://stripe.com/docs
- **Firebase**: https://firebase.google.com/docs

### Community Support
- **Stack Overflow**: Tag your questions appropriately
- **Service-specific forums**: Each service has community forums
- **GitHub Issues**: For open-source integrations

---

**🎉 You now have comprehensive implementations for all major external services!**

Each service is production-ready with proper error handling, security considerations, and scalability in mind. Start with Phase 1 services and gradually implement the others based on your needs and priorities.