#include <gtest/gtest.h>
#include "network/http_protocol.hpp"
#include <string>
#include <chrono>

using namespace ultra::network;

class HttpProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser_ = std::make_unique<Http11Parser>();
        detector_ = std::make_unique<ProtocolDetector>();
    }
    
    std::unique_ptr<Http11Parser> parser_;
    std::unique_ptr<ProtocolDetector> detector_;
};

TEST_F(HttpProtocolTest, ParseSimpleGetRequest) {
    const char* request_data = 
        "GET /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "User-Agent: TestClient/1.0\r\n"
        "\r\n";
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::COMPLETE);
    EXPECT_EQ(request.method, HttpMethod::GET);
    EXPECT_EQ(request.path, "/api/users");
    EXPECT_EQ(request.version, HttpVersion::HTTP_1_1);
    EXPECT_EQ(request.headers.size(), 2);
    
    // Check headers
    bool found_host = false, found_user_agent = false;
    for (const auto& header : request.headers) {
        if (header.name == "Host") {
            EXPECT_EQ(header.value, "example.com");
            found_host = true;
        } else if (header.name == "User-Agent") {
            EXPECT_EQ(header.value, "TestClient/1.0");
            found_user_agent = true;
        }
    }
    EXPECT_TRUE(found_host);
    EXPECT_TRUE(found_user_agent);
}

TEST_F(HttpProtocolTest, ParsePostRequestWithBody) {
    const char* request_data = 
        "POST /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 25\r\n"
        "\r\n"
        "{\"name\":\"John Doe\",\"age\":30}";
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::COMPLETE);
    EXPECT_EQ(request.method, HttpMethod::POST);
    EXPECT_EQ(request.path, "/api/users");
    EXPECT_EQ(request.body, "{\"name\":\"John Doe\",\"age\":30}");
    EXPECT_EQ(state.content_length, 25);
}

TEST_F(HttpProtocolTest, ParseRequestWithQueryString) {
    const char* request_data = 
        "GET /search?q=test&limit=10&offset=0 HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "\r\n";
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::COMPLETE);
    EXPECT_EQ(request.path, "/search");
    EXPECT_EQ(request.query_string, "q=test&limit=10&offset=0");
}

TEST_F(HttpProtocolTest, ParseIncompleteRequest) {
    const char* request_data = 
        "GET /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n";
    // Missing final CRLF and potentially more headers
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::INCOMPLETE);
}

TEST_F(HttpProtocolTest, ParseInvalidRequest) {
    const char* request_data = 
        "INVALID_METHOD /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "\r\n";
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::ERROR);
}

TEST_F(HttpProtocolTest, MethodParsingSIMD) {
    // Test SIMD-optimized method parsing
    EXPECT_EQ(parser_->parse_method_simd("GET", 3), HttpMethod::GET);
    EXPECT_EQ(parser_->parse_method_simd("POST", 4), HttpMethod::POST);
    EXPECT_EQ(parser_->parse_method_simd("PUT", 3), HttpMethod::PUT);
    EXPECT_EQ(parser_->parse_method_simd("DELETE", 6), HttpMethod::DELETE);
    EXPECT_EQ(parser_->parse_method_simd("HEAD", 4), HttpMethod::HEAD);
    EXPECT_EQ(parser_->parse_method_simd("OPTIONS", 7), HttpMethod::OPTIONS);
    EXPECT_EQ(parser_->parse_method_simd("PATCH", 5), HttpMethod::PATCH);
    EXPECT_EQ(parser_->parse_method_simd("INVALID", 7), HttpMethod::UNKNOWN);
}

TEST_F(HttpProtocolTest, ProtocolDetection) {
    // Test HTTP/1.1 detection
    const char* http11_request = "GET /test HTTP/1.1\r\n";
    EXPECT_EQ(detector_->detect_protocol(http11_request, std::strlen(http11_request)), 
              ProtocolDetector::Protocol::HTTP_1_1);
    
    // Test HTTP/1.0 detection
    const char* http10_request = "GET /test HTTP/1.0\r\n";
    EXPECT_EQ(detector_->detect_protocol(http10_request, std::strlen(http10_request)), 
              ProtocolDetector::Protocol::HTTP_1_0);
    
    // Test HTTP/2 preface detection
    const char* http2_preface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    EXPECT_EQ(detector_->detect_protocol(http2_preface, std::strlen(http2_preface)), 
              ProtocolDetector::Protocol::HTTP_2_0);
    
    // Test unknown protocol
    const char* unknown_data = "UNKNOWN PROTOCOL DATA";
    EXPECT_EQ(detector_->detect_protocol(unknown_data, std::strlen(unknown_data)), 
              ProtocolDetector::Protocol::UNKNOWN);
}

TEST_F(HttpProtocolTest, PerformanceBenchmark) {
    const char* request_data = 
        "GET /api/v1/users/12345?include=profile,settings&format=json HTTP/1.1\r\n"
        "Host: api.example.com\r\n"
        "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36\r\n"
        "Accept: application/json,text/plain,*/*\r\n"
        "Accept-Language: en-US,en;q=0.9\r\n"
        "Accept-Encoding: gzip, deflate, br\r\n"
        "Connection: keep-alive\r\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "\r\n";
    
    const int num_iterations = 100000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        HttpRequest request;
        Http11Parser::ParseState state;
        
        auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
        EXPECT_EQ(result, Http11Parser::ParseResult::COMPLETE);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double avg_parse_time = static_cast<double>(duration.count()) / num_iterations;
    
    std::cout << "HTTP parsing performance:" << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "  Average parse time: " << avg_parse_time << " μs" << std::endl;
    std::cout << "  Requests per second: " << (1000000.0 / avg_parse_time) << std::endl;
    
    // Performance expectation: should parse at least 100k requests per second
    EXPECT_LT(avg_parse_time, 10.0); // Less than 10 microseconds per request
}

TEST_F(HttpProtocolTest, ZeroCopyParsing) {
    const char* request_data = 
        "GET /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "\r\n";
    
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser_->parse_request(request_data, std::strlen(request_data), request, state);
    
    EXPECT_EQ(result, Http11Parser::ParseResult::COMPLETE);
    
    // Verify zero-copy: string_views should point to original buffer
    EXPECT_GE(request.path.data(), request_data);
    EXPECT_LT(request.path.data(), request_data + std::strlen(request_data));
    
    for (const auto& header : request.headers) {
        EXPECT_GE(header.name.data(), request_data);
        EXPECT_LT(header.name.data(), request_data + std::strlen(request_data));
        EXPECT_GE(header.value.data(), request_data);
        EXPECT_LT(header.value.data(), request_data + std::strlen(request_data));
    }
}

// HTTP/2 Parser Tests
class Http2ParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser_ = std::make_unique<Http2Parser>();
    }
    
    std::unique_ptr<Http2Parser> parser_;
};

TEST_F(Http2ParserTest, ParseConnectionPreface) {
    const char* preface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    
    auto result = parser_->parse_connection_preface(preface, std::strlen(preface));
    EXPECT_EQ(result, Http2Parser::ParseResult::COMPLETE);
}

TEST_F(Http2ParserTest, ParseInvalidPreface) {
    const char* invalid_preface = "GET / HTTP/1.1\r\n\r\n";
    
    auto result = parser_->parse_connection_preface(invalid_preface, std::strlen(invalid_preface));
    EXPECT_EQ(result, Http2Parser::ParseResult::CONNECTION_ERROR);
}

TEST_F(Http2ParserTest, ParseIncompletePreface) {
    const char* incomplete_preface = "PRI * HTTP/2.0\r\n";
    
    auto result = parser_->parse_connection_preface(incomplete_preface, std::strlen(incomplete_preface));
    EXPECT_EQ(result, Http2Parser::ParseResult::INCOMPLETE);
}