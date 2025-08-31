import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler } from '../../../lib/errorHandler';

const tagsHandler = async (req, res) => {
  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  switch (req.method) {
    case 'GET':
      return await getTags(req, res, user);
    default:
      return res.status(405).json({
        success: false,
        error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
      });
  }
};

const getTags = async (req, res, user) => {
  try {
    const { 
      search,
      limit = 100,
      sortBy = 'count',
      sortOrder = 'desc'
    } = req.query;

    // Aggregate tags from all media files
    const pipeline = [
      // Unwind tags array
      { $unwind: '$tags' },
      
      // Group by tag and count occurrences
      {
        $group: {
          _id: '$tags',
          count: { $sum: 1 },
          mediaFiles: { $addToSet: '$_id' }
        }
      },
      
      // Reshape the output
      {
        $project: {
          _id: 0,
          name: '$_id',
          count: 1,
          mediaFiles: 1
        }
      }
    ];

    // Add search filter if provided
    if (search) {
      pipeline.push({
        $match: {
          name: { $regex: search, $options: 'i' }
        }
      });
    }

    // Add sorting
    const sortField = sortBy === 'name' ? 'name' : 'count';
    const sortDirection = sortOrder === 'asc' ? 1 : -1;
    pipeline.push({
      $sort: { [sortField]: sortDirection }
    });

    // Add limit
    pipeline.push({
      $limit: parseInt(limit)
    });

    const tags = await Media.aggregate(pipeline);

    // Get tag statistics
    const totalTags = await Media.aggregate([
      { $unwind: '$tags' },
      { $group: { _id: '$tags' } },
      { $count: 'total' }
    ]);

    const totalUniqueTagsCount = totalTags.length > 0 ? totalTags[0].total : 0;

    // Get most popular tags
    const popularTags = await Media.aggregate([
      { $unwind: '$tags' },
      { $group: { _id: '$tags', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
      { $limit: 10 },
      { $project: { _id: 0, name: '$_id', count: 1 } }
    ]);

    const response = {
      success: true,
      data: {
        tags,
        statistics: {
          totalUniqueTags: totalUniqueTagsCount,
          totalTaggedFiles: await Media.countDocuments({ tags: { $exists: true, $ne: [] } }),
          averageTagsPerFile: await getAverageTagsPerFile()
        },
        popularTags
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Get tags error:', error);
    return res.status(500).json({
      success: false,
      error: { message: 'Failed to retrieve tags', code: 'RETRIEVAL_ERROR' }
    });
  }
};

// Helper function to calculate average tags per file
const getAverageTagsPerFile = async () => {
  try {
    const result = await Media.aggregate([
      { $match: { tags: { $exists: true, $ne: [] } } },
      {
        $group: {
          _id: null,
          averageTags: { $avg: { $size: '$tags' } }
        }
      }
    ]);

    return result.length > 0 ? Math.round(result[0].averageTags * 100) / 100 : 0;
  } catch (error) {
    console.error('Error calculating average tags:', error);
    return 0;
  }
};

export default asyncHandler(tagsHandler);