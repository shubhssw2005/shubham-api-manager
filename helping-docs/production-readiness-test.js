import axios from 'axios';
import { spawn } from 'child_process';
import { performance } from 'perf_hooks';
import fs from 'fs/promises';
import path from 'path';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';

class ProductionReadinessTest {
    constructor() {
        this.baseURL = 'http://localhost:3005';
        this.results = {
            overall: 'UNKNOWN',
            score: 0,
            maxScore: 100,
            tests: [],
            critical_issues: [],
            warnings: [],
            recommendations: []
        };
        this.token = null;
    }

    async generateToken() {
        const secret = process.env.JWT_SECRET || 'b802e635a669a62c06677a295dfe2f6c';
        this.token = jwt.sign({
            userId: uuidv4(),
            email: 'production-test@example.com',
            approved: true,
            iat: Math.floor(Date.now() / 1000),
            exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60)
        }, secret);
    }

    addTest(name, status, score, details = '', critical = false) {
        this.results.tests.push({
            name,
            status,
            score,
            details,
            critical,
            timestamp: new Date().toISOString()
        });

        if (status === 'PASS') {
            this.results.score += score;
        } else if (critical) {
            this.results.critical_issues.push(`${name}: ${details}`);
        } else {
            this.results.warnings.push(`${name}: ${details}`);
        }
    }

    async testAPIHealth() {
        console.log('🏥 Testing API Health...');
        try {
            const response = await axios.get(`${this.baseURL}/api/v2/universal/health`, {
                timeout: 5000
            });

            if (response.status === 200 && response.data.success) {
                const health = response.data.data;
                
                if (health.status === 'healthy') {
                    this.addTest('API Health Check', 'PASS', 10, `Response time: ${health.responseTime}ms`);
                    
                    // Check database services
                    if (health.services.scylladb?.status === 'healthy') {
                        this.addTest('ScyllaDB Health', 'PASS', 5, `Latency: ${health.services.scylladb.latency}ms`);
                    } else {
                        this.addTest('ScyllaDB Health', 'FAIL', 0, 'ScyllaDB not healthy', true);
                    }

                    if (health.services.foundationdb?.status === 'healthy') {
                        this.addTest('FoundationDB Health', 'PASS', 5, `Latency: ${health.services.foundationdb.latency}ms`);
                    } else {
                        this.addTest('FoundationDB Health', 'FAIL', 0, 'FoundationDB not healthy', true);
                    }
                } else {
                    this.addTest('API Health Check', 'FAIL', 0, `Status: ${health.status}`, true);
                }
            } else {
                this.addTest('API Health Check', 'FAIL', 0, 'Invalid response format', true);
            }
        } catch (error) {
            this.addTest('API Health Check', 'FAIL', 0, `Error: ${error.message}`, true);
        }
    }

    async testAuthentication() {
        console.log('🔐 Testing Authentication...');
        try {
            await this.generateToken();

            // Test protected endpoint without token
            try {
                await axios.post(`${this.baseURL}/api/universal/posts`, {
                    title: 'Test Post',
                    content: 'Test content'
                });
                this.addTest('Authentication Required', 'FAIL', 0, 'Endpoint accessible without token', true);
            } catch (error) {
                if (error.response?.status === 401) {
                    this.addTest('Authentication Required', 'PASS', 5, 'Properly rejects unauthenticated requests');
                } else {
                    this.addTest('Authentication Required', 'FAIL', 0, `Unexpected error: ${error.message}`);
                }
            }

            // Test with valid token
            try {
                const response = await axios.post(`${this.baseURL}/api/universal/posts`, {
                    title: 'Production Test Post',
                    content: 'This is a production readiness test post',
                    author_id: uuidv4()
                }, {
                    headers: {
                        'Authorization': `Bearer ${this.token}`,
                        'Content-Type': 'application/json'
                    }
                });

                if (response.status === 201) {
                    this.addTest('JWT Authentication', 'PASS', 5, 'Valid token accepted');
                } else {
                    this.addTest('JWT Authentication', 'FAIL', 0, `Unexpected status: ${response.status}`);
                }
            } catch (error) {
                this.addTest('JWT Authentication', 'FAIL', 0, `Token validation failed: ${error.message}`);
            }

        } catch (error) {
            this.addTest('Authentication System', 'FAIL', 0, `Setup fa