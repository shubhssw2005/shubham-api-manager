import mongoose from 'mongoose';
import dbConnect from '../../lib/dbConnect.js';
import { requireApprovedUser } from '../../middleware/auth.js';
import Outbox from '../../models/Outbox.js';

/**
 * Test Outbox Events
 */
export default async function handler(req, res) {
    try {
        await dbConnect();

        const user = await requireApprovedUser(req, res);
        if (!user) return;

        switch (req.method) {
            case 'GET':
                return await getOutboxEvents(req, res);
            default:
                return res.status(405).json({
                    success: false,
                    message: `Method ${req.method} not allowed`
                });
        }
    } catch (error) {
        console.error('Outbox test error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: error.message
        });
    }
}

async function getOutboxEvents(req, res) {
    try {
        const { aggregate, processed } = req.query;
        
        const filter = {};
        if (aggregate) filter.aggregate = aggregate;
        if (processed !== undefined) filter.processed = processed === 'true';
        
        const events = await Outbox.find(filter)
            .sort({ createdAt: -1 })
            .limit(20);
        
        const stats = await Outbox.aggregate([
            {
                $group: {
                    _id: '$aggregate',
                    total: { $sum: 1 },
                    processed: { $sum: { $cond: ['$processed', 1, 0] } },
                    pending: { $sum: { $cond: ['$processed', 0, 1] } }
                }
            }
        ]);
        
        res.status(200).json({
            success: true,
            data: {
                events,
                stats,
                totalEvents: events.length
            }
        });
        
    } catch (error) {
        throw error;
    }
}