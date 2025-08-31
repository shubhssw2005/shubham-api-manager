import dbConnect, { getDatabaseHealth } from '../../../../lib/dbConnect.js';

export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({
            success: false,
            message: 'Method not allowed'
        });
    }

    const startTime = Date.now();
    const healthCheck = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: process.env.npm_package_version || '1.0.0',
        environment: process.env.NODE_ENV || 'development',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        services: {}
    };

    try {
        // Check database connections
        try {
            await dbConnect();
            const dbHealth = await getDatabaseHealth();
            
            healthCheck.services = {
                ...healthCheck.services,
                ...dbHealth
            };

            // Set overall status based on database health
            if (dbHealth.overall === 'unhealthy') {
                healthCheck.status = 'unhealthy';
            } else if (dbHealth.overall === 'degraded') {
                healthCheck.status = 'degraded';
            }
        } catch (error) {
            healthCheck.services.database = {
                status: 'unhealthy',
                error: error.message
            };
            healthCheck.status = 'unhealthy';
        }

        // Response time
        healthCheck.responseTime = Date.now() - startTime;

        // Determine overall status
        const serviceStatuses = Object.values(healthCheck.services).map(s => s.status);
        if (serviceStatuses.includes('unhealthy')) {
            healthCheck.status = 'unhealthy';
        } else if (serviceStatuses.includes('degraded')) {
            healthCheck.status = 'degraded';
        }

        const statusCode = healthCheck.status === 'healthy' ? 200 : 
                          healthCheck.status === 'degraded' ? 200 : 503;

        return res.status(statusCode).json({
            success: true,
            data: healthCheck
        });

    } catch (error) {
        console.error('Health check error:', error);
        return res.status(503).json({
            success: false,
            status: 'unhealthy',
            message: 'Health check failed',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined,
            responseTime: Date.now() - startTime
        });
    }
}