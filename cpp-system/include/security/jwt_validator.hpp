#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#include <unordered_map>
#include <openssl/rsa.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <json/json.h>

namespace ultra_cpp::security {

enum class JWTAlgorithm {
    RS256,  // RSA with SHA-256
    RS384,  // RSA with SHA-384
    RS512,  // RSA with SHA-512
    ES256,  // ECDSA with SHA-256
    ES384,  // ECDSA with SHA-384
    ES512   // ECDSA with SHA-512
};

struct JWTHeader {
    JWTAlgorithm algorithm;
    std::string key_id;
    std::string type;
};

struct JWTPayload {
    std::string issuer;
    std::string subject;
    std::string audience;
    std::chrono::system_clock::time_point expiration;
    std::chrono::system_clock::time_point not_before;
    std::chrono::system_clock::time_point issued_at;
    std::string jwt_id;
    std::unordered_map<std::string, Json::Value> custom_claims;
};

struct JWTToken {
    JWTHeader header;
    JWTPayload payload;
    std::string signature;
    std::string raw_token;
};

class JWTValidator {
public:
    struct Config {
        std::string issuer;
        std::string audience;
        std::chrono::seconds clock_skew_tolerance{300}; // 5 minutes
        bool verify_expiration = true;
        bool verify_not_before = true;
        bool verify_issued_at = true;
    };

    explicit JWTValidator(const Config& config);
    ~JWTValidator();

    // Key management
    bool add_rsa_public_key(const std::string& key_id, const std::string& pem_key);
    bool add_ecdsa_public_key(const std::string& key_id, const std::string& pem_key);
    bool load_jwks(const std::string& jwks_json);
    void remove_key(const std::string& key_id);

    // Token validation
    std::optional<JWTToken> validate_token(const std::string& token);
    bool is_token_valid(const std::string& token);

    // Performance metrics
    struct ValidationStats {
        std::atomic<uint64_t> tokens_validated{0};
        std::atomic<uint64_t> validation_failures{0};
        std::atomic<uint64_t> signature_verifications{0};
        std::atomic<uint64_t> avg_validation_time_ns{0};
    };

    ValidationStats get_stats() const { return stats_; }

private:
    Config config_;
    std::unordered_map<std::string, EVP_PKEY*> public_keys_;
    mutable ValidationStats stats_;

    // Internal validation methods
    std::optional<JWTHeader> parse_header(const std::string& header_b64);
    std::optional<JWTPayload> parse_payload(const std::string& payload_b64);
    bool verify_signature(const JWTHeader& header, 
                         const std::string& message,
                         const std::string& signature_b64);
    bool verify_claims(const JWTPayload& payload);
    
    // Utility methods
    std::string base64_url_decode(const std::string& input);
    JWTAlgorithm string_to_algorithm(const std::string& alg);
    EVP_PKEY* load_rsa_key_from_pem(const std::string& pem);
    EVP_PKEY* load_ecdsa_key_from_pem(const std::string& pem);
};

} // namespace ultra_cpp::security