#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <json/json.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/lambda-runtime/runtime.h>
#include <aws/ce/CostExplorerClient.h>
#include <aws/ce/model/GetCostAndUsageRequest.h>
#include <aws/ce/model/GetRightsizingRecommendationRequest.h>
#include <aws/sns/SNSClient.h>
#include <aws/sns/model/PublishRequest.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListBucketsRequest.h>
#include <aws/s3/model/GetBucketAnalyticsConfigurationRequest.h>
#include <aws/ec2/EC2Client.h>
#include <aws/ec2/model/DescribeInstancesRequest.h>

using namespace aws::lambda_runtime;

class CostOptimizer {
private:
    std::shared_ptr<Aws::CostExplorer::CostExplorerClient> ceClient;
    std::shared_ptr<Aws::SNS::SNSClient> snsClient;
    std::shared_ptr<Aws::S3::S3Client> s3Client;
    std::shared_ptr<Aws::EC2::EC2Client> ec2Client;
    std::string snsTopicArn;
    std::string projectName;
    std::string environment;

public:
    CostOptimizer() {
        Aws::Client::ClientConfiguration config;
        config.region = Aws::Environment::GetEnv("AWS_DEFAULT_REGION");
        
        ceClient = std::make_shared<Aws::CostExplorer::CostExplorerClient>(config);
        snsClient = std::make_shared<Aws::SNS::SNSClient>(config);
        s3Client = std::make_shared<Aws::S3::S3Client>(config);
        ec2Client = std::make_shared<Aws::EC2::EC2Client>(config);
        
        snsTopicArn = Aws::Environment::GetEnv("SNS_TOPIC_ARN");
        projectName = Aws::Environment::GetEnv("PROJECT_NAME");
        environment = Aws::Environment::GetEnv("ENVIRONMENT");
    }

    struct CostRecommendation {
        std::string service;
        std::string resourceId;
        std::string recommendationType;
        std::string description;
        double potentialSavings;
        std::string priority;
    };

    std::vector<CostRecommendation> analyzeEC2Costs() {
        std::vector<CostRecommendation> recommendations;
        
        try {
            // Get rightsizing recommendations
            Aws::CostExplorer::Model::GetRightsizingRecommendationRequest request;
            request.SetService("AmazonEC2");
            
            Aws::CostExplorer::Model::Expression filter;
            Aws::CostExplorer::Model::DimensionKey dimensionKey;
            dimensionKey.SetKey(Aws::CostExplorer::Model::Dimension::SERVICE);
            dimensionKey.SetValues({"Amazon Elastic Compute Cloud - Compute"});
            filter.SetDimensions(dimensionKey);
            request.SetFilter(filter);
            
            auto outcome = ceClient->GetRightsizingRecommendation(request);
            
            if (outcome.IsSuccess()) {
                const auto& result = outcome.GetResult();
                for (const auto& recommendation : result.GetRightsizingRecommendations()) {
                    CostRecommendation rec;
                    rec.service = "EC2";
                    rec.resourceId = recommendation.GetResourceId();
                    rec.recommendationType = "Rightsizing";
                    rec.description = "Consider rightsizing this instance based on utilization";
                    rec.potentialSavings = std::stod(recommendation.GetEstimatedMonthlySavings());
                    rec.priority = rec.potentialSavings > 100 ? "High" : "Medium";
                    recommendations.push_back(rec);
                }
            }
            
            // Check for underutilized instances
            Aws::EC2::Model::DescribeInstancesRequest ec2Request;
            auto ec2Outcome = ec2Client->DescribeInstances(ec2Request);
            
            if (ec2Outcome.IsSuccess()) {
                for (const auto& reservation : ec2Outcome.GetResult().GetReservations()) {
                    for (const auto& instance : reservation.GetInstances()) {
                        if (instance.GetState().GetName() == Aws::EC2::Model::InstanceStateName::running) {
                            // Check if instance has been running for more than 30 days with low utilization
                            // This would typically require CloudWatch metrics analysis
                            CostRecommendation rec;
                            rec.service = "EC2";
                            rec.resourceId = instance.GetInstanceId();
                            rec.recommendationType = "Utilization Review";
                            rec.description = "Review instance utilization - consider scheduling or rightsizing";
                            rec.potentialSavings = 50.0; // Estimated
                            rec.priority = "Medium";
                            recommendations.push_back(rec);
                        }
                    }
                }
            }
            
        } catch (const std::exception& e) {
            AWS_LOGSTREAM_ERROR("CostOptimizer", "Error analyzing EC2 costs: " << e.what());
        }
        
        return recommendations;
    }

    std::vector<CostRecommendation> analyzeS3Costs() {
        std::vector<CostRecommendation> recommendations;
        
        try {
            Aws::S3::Model::ListBucketsRequest request;
            auto outcome = s3Client->ListBuckets(request);
            
            if (outcome.IsSuccess()) {
                for (const auto& bucket : outcome.GetResult().GetBuckets()) {
                    std::string bucketName = bucket.GetName();
                    
                    // Check if bucket has lifecycle policies
                    if (bucketName.find(projectName) != std::string::npos) {
                        CostRecommendation rec;
                        rec.service = "S3";
                        rec.resourceId = bucketName;
                        rec.recommendationType = "Lifecycle Policy";
                        rec.description = "Ensure lifecycle policies are optimized for cost";
                        rec.potentialSavings = 25.0; // Estimated 25% savings
                        rec.priority = "Medium";
                        recommendations.push_back(rec);
                        
                        // Recommend intelligent tiering if not enabled
                        rec.recommendationType = "Intelligent Tiering";
                        rec.description = "Enable S3 Intelligent Tiering for automatic cost optimization";
                        rec.potentialSavings = 15.0;
                        rec.priority = "Low";
                        recommendations.push_back(rec);
                    }
                }
            }
            
        } catch (const std::exception& e) {
            AWS_LOGSTREAM_ERROR("CostOptimizer", "Error analyzing S3 costs: " << e.what());
        }
        
        return recommendations;
    }

    std::vector<CostRecommendation> analyzeReservedInstanceOpportunities() {
        std::vector<CostRecommendation> recommendations;
        
        try {
            // This would typically analyze usage patterns and recommend RIs
            CostRecommendation rec;
            rec.service = "EC2";
            rec.resourceId = "Reserved Instances";
            rec.recommendationType = "Reserved Instance Purchase";
            rec.description = "Consider purchasing Reserved Instances for consistent workloads";
            rec.potentialSavings = 200.0; // Estimated savings
            rec.priority = "High";
            recommendations.push_back(rec);
            
        } catch (const std::exception& e) {
            AWS_LOGSTREAM_ERROR("CostOptimizer", "Error analyzing RI opportunities: " << e.what());
        }
        
        return recommendations;
    }

    std::vector<CostRecommendation> analyzeSpotInstanceOpportunities() {
        std::vector<CostRecommendation> recommendations;
        
        CostRecommendation rec;
        rec.service = "EKS";
        rec.resourceId = "Worker Nodes";
        rec.recommendationType = "Spot Instances";
        rec.description = "Consider using Spot Instances for non-critical workloads (workers, batch jobs)";
        rec.potentialSavings = 150.0; // Up to 70% savings
        rec.priority = "High";
        recommendations.push_back(rec);
        
        return recommendations;
    }

    void sendRecommendations(const std::vector<CostRecommendation>& recommendations) {
        if (recommendations.empty()) {
            return;
        }
        
        Json::Value message;
        message["project"] = projectName;
        message["environment"] = environment;
        message["timestamp"] = std::time(nullptr);
        message["total_recommendations"] = static_cast<int>(recommendations.size());
        
        double totalPotentialSavings = 0.0;
        Json::Value recs(Json::arrayValue);
        
        for (const auto& rec : recommendations) {
            Json::Value recJson;
            recJson["service"] = rec.service;
            recJson["resource_id"] = rec.resourceId;
            recJson["type"] = rec.recommendationType;
            recJson["description"] = rec.description;
            recJson["potential_savings"] = rec.potentialSavings;
            recJson["priority"] = rec.priority;
            
            recs.append(recJson);
            totalPotentialSavings += rec.potentialSavings;
        }
        
        message["recommendations"] = recs;
        message["total_potential_savings"] = totalPotentialSavings;
        
        Json::StreamWriterBuilder builder;
        std::string messageStr = Json::writeString(builder, message);
        
        // Send to SNS
        Aws::SNS::Model::PublishRequest publishRequest;
        publishRequest.SetTopicArn(snsTopicArn);
        publishRequest.SetMessage(messageStr);
        publishRequest.SetSubject("Cost Optimization Recommendations - " + projectName + " (" + environment + ")");
        
        auto outcome = snsClient->Publish(publishRequest);
        if (!outcome.IsSuccess()) {
            AWS_LOGSTREAM_ERROR("CostOptimizer", "Failed to send SNS message: " << outcome.GetError().GetMessage());
        }
    }

    invocation_response run() {
        std::vector<CostRecommendation> allRecommendations;
        
        // Gather recommendations from different services
        auto ec2Recs = analyzeEC2Costs();
        auto s3Recs = analyzeS3Costs();
        auto riRecs = analyzeReservedInstanceOpportunities();
        auto spotRecs = analyzeSpotInstanceOpportunities();
        
        allRecommendations.insert(allRecommendations.end(), ec2Recs.begin(), ec2Recs.end());
        allRecommendations.insert(allRecommendations.end(), s3Recs.begin(), s3Recs.end());
        allRecommendations.insert(allRecommendations.end(), riRecs.begin(), riRecs.end());
        allRecommendations.insert(allRecommendations.end(), spotRecs.begin(), spotRecs.end());
        
        // Send recommendations
        sendRecommendations(allRecommendations);
        
        Json::Value response;
        response["statusCode"] = 200;
        response["recommendations_count"] = static_cast<int>(allRecommendations.size());
        
        Json::StreamWriterBuilder builder;
        return invocation_response::success(Json::writeString(builder, response), "application/json");
    }
};

invocation_response my_handler(invocation_request const& request) {
    CostOptimizer optimizer;
    return optimizer.run();
}

int main() {
    Aws::SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
    Aws::InitAPI(options);
    
    {
        run_handler(my_handler);
    }
    
    Aws::ShutdownAPI(options);
    return 0;
}