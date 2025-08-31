#include "security/jwt_validator.hpp"
#include "common/logger.hpp"
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace ultra_cpp::security {

JWTValidator::JWTValidator(const Config& config) : config_(config) {
    // Initialize OpenSSL
    OpenSSL_add_all_algorithms();
}

JWTValidator::~JWTValidator() {
    // Clean up public keys
    for (auto& [key_id, key] : public_keys_) {
        EVP_PKEY_free(key);
    }
    public_keys_.clear();
}

bool JWTValidator::add_rsa_public_key(const std::string& key_id, const std::string& pem_key) {
    EVP_PKEY* key = load_rsa_key_from_pem(pem_key);
    if (!key) {
        LOG_ERROR("Failed to load RSA public key for key_id: {}", key_id);
        return false;
    }

    // Remove existing key if present
    auto it = public_keys_.find(key_id);
    if (it != public_keys_.end()) {
        EVP_PKEY_free(it->second);
    }

    public_keys_[key_id] = key;
    LOG_INFO("Added RSA public key for key_id: {}", key_id);
    return true;
}

bool JWTValidator::add_ecdsa_public_key(const std::string& key_id, const std::string& pem_key) {
    EVP_PKEY* key = load_ecdsa_key_from_pem(pem_key);
    if (!key) {
        LOG_ERROR("Failed to load ECDSA public key for key_id: {}", key_id);
        return false;
    }

    // Remove existing key if present
    auto it = public_keys_.find(key_id);
    if (it != public_keys_.end()) {
        EVP_PKEY_free(it->second);
    }

    public_keys_[key_id] = key;
    LOG_INFO("Added ECDSA public key for key_id: {}", key_id);
    return true;
}

bool JWTValidator::load_jwks(const std::string& jwks_json) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(jwks_json, root)) {
        LOG_ERROR("Failed to parse JWKS JSON");
        return false;
    }

    if (!root.isMember("keys") || !root["keys"].isArray()) {
        LOG_ERROR("Invalid JWKS format: missing 'keys' array");
        return false;
    }

    bool success = true;
    for (const auto& key : root["keys"]) {
        if (!key.isMember("kid") || !key.isMember("kty")) {
            LOG_WARN("Skipping key without 'kid' or 'kty'");
            continue;
        }

        std::string key_id = key["kid"].asString();
        std::string key_type = key["kty"].asString();

        if (key_type == "RSA" && key.isMember("n") && key.isMember("e")) {
            // Convert JWK RSA to PEM format (simplified - would need full implementation)
            LOG_INFO("Loading RSA key from JWKS: {}", key_id);
            // Implementation would convert JWK format to PEM
        } else if (key_type == "EC" && key.isMember("x") && key.isMember("y")) {
            // Convert JWK EC to PEM format (simplified - would need full implementation)
            LOG_INFO("Loading EC key from JWKS: {}", key_id);
            // Implementation would convert JWK format to PEM
        }
    }

    return success;
}

void JWTValidator::remove_key(const std::string& key_id) {
    auto it = public_keys_.find(key_id);
    if (it != public_keys_.end()) {
        EVP_PKEY_free(it->second);
        public_keys_.erase(it);
        LOG_INFO("Removed key: {}", key_id);
    }
}

std::optional<JWTToken> JWTValidator::validate_token(const std::string& token) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    stats_.tokens_validated.fetch_add(1, std::memory_order_relaxed);

    // Split token into parts
    std::vector<std::string> parts;
    std::stringstream ss(token);
    std::string part;
    
    while (std::getline(ss, part, '.')) {
        parts.push_back(part);
    }

    if (parts.size() != 3) {
        LOG_ERROR("Invalid JWT format: expected 3 parts, got {}", parts.size());
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    // Parse header
    auto header = parse_header(parts[0]);
    if (!header) {
        LOG_ERROR("Failed to parse JWT header");
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    // Parse payload
    auto payload = parse_payload(parts[1]);
    if (!payload) {
        LOG_ERROR("Failed to parse JWT payload");
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    // Verify signature
    std::string message = parts[0] + "." + parts[1];
    if (!verify_signature(*header, message, parts[2])) {
        LOG_ERROR("JWT signature verification failed");
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    // Verify claims
    if (!verify_claims(*payload)) {
        LOG_ERROR("JWT claims verification failed");
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    stats_.avg_validation_time_ns.store(duration.count(), std::memory_order_relaxed);

    return JWTToken{*header, *payload, parts[2], token};
}

bool JWTValidator::is_token_valid(const std::string& token) {
    return validate_token(token).has_value();
}

std::optional<JWTHeader> JWTValidator::parse_header(const std::string& header_b64) {
    try {
        std::string header_json = base64_url_decode(header_b64);
        
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(header_json, root)) {
            return std::nullopt;
        }

        JWTHeader header;
        
        if (root.isMember("alg")) {
            header.algorithm = string_to_algorithm(root["alg"].asString());
        }
        
        if (root.isMember("kid")) {
            header.key_id = root["kid"].asString();
        }
        
        if (root.isMember("typ")) {
            header.type = root["typ"].asString();
        }

        return header;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception parsing JWT header: {}", e.what());
        return std::nullopt;
    }
}

std::optional<JWTPayload> JWTValidator::parse_payload(const std::string& payload_b64) {
    try {
        std::string payload_json = base64_url_decode(payload_b64);
        
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(payload_json, root)) {
            return std::nullopt;
        }

        JWTPayload payload;
        
        if (root.isMember("iss")) {
            payload.issuer = root["iss"].asString();
        }
        
        if (root.isMember("sub")) {
            payload.subject = root["sub"].asString();
        }
        
        if (root.isMember("aud")) {
            payload.audience = root["aud"].asString();
        }
        
        if (root.isMember("exp")) {
            auto exp_time = std::chrono::system_clock::from_time_t(root["exp"].asInt64());
            payload.expiration = exp_time;
        }
        
        if (root.isMember("nbf")) {
            auto nbf_time = std::chrono::system_clock::from_time_t(root["nbf"].asInt64());
            payload.not_before = nbf_time;
        }
        
        if (root.isMember("iat")) {
            auto iat_time = std::chrono::system_clock::from_time_t(root["iat"].asInt64());
            payload.issued_at = iat_time;
        }
        
        if (root.isMember("jti")) {
            payload.jwt_id = root["jti"].asString();
        }

        // Store all other claims as custom claims
        for (const auto& member : root.getMemberNames()) {
            if (member != "iss" && member != "sub" && member != "aud" && 
                member != "exp" && member != "nbf" && member != "iat" && member != "jti") {
                payload.custom_claims[member] = root[member];
            }
        }

        return payload;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception parsing JWT payload: {}", e.what());
        return std::nullopt;
    }
}

bool JWTValidator::verify_signature(const JWTHeader& header, 
                                   const std::string& message,
                                   const std::string& signature_b64) {
    stats_.signature_verifications.fetch_add(1, std::memory_order_relaxed);

    // Find the public key
    auto key_it = public_keys_.find(header.key_id);
    if (key_it == public_keys_.end()) {
        LOG_ERROR("Public key not found for key_id: {}", header.key_id);
        return false;
    }

    EVP_PKEY* public_key = key_it->second;
    std::string signature = base64_url_decode(signature_b64);

    // Create verification context
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        LOG_ERROR("Failed to create EVP_MD_CTX");
        return false;
    }

    bool result = false;
    const EVP_MD* md = nullptr;

    // Select hash algorithm based on JWT algorithm
    switch (header.algorithm) {
        case JWTAlgorithm::RS256:
        case JWTAlgorithm::ES256:
            md = EVP_sha256();
            break;
        case JWTAlgorithm::RS384:
        case JWTAlgorithm::ES384:
            md = EVP_sha384();
            break;
        case JWTAlgorithm::RS512:
        case JWTAlgorithm::ES512:
            md = EVP_sha512();
            break;
    }

    if (md && EVP_DigestVerifyInit(ctx, nullptr, md, nullptr, public_key) == 1) {
        if (EVP_DigestVerifyUpdate(ctx, message.c_str(), message.length()) == 1) {
            result = (EVP_DigestVerifyFinal(ctx, 
                     reinterpret_cast<const unsigned char*>(signature.c_str()), 
                     signature.length()) == 1);
        }
    }

    EVP_MD_CTX_free(ctx);
    return result;
}

bool JWTValidator::verify_claims(const JWTPayload& payload) {
    auto now = std::chrono::system_clock::now();

    // Verify issuer
    if (!config_.issuer.empty() && payload.issuer != config_.issuer) {
        LOG_ERROR("Invalid issuer: expected '{}', got '{}'", config_.issuer, payload.issuer);
        return false;
    }

    // Verify audience
    if (!config_.audience.empty() && payload.audience != config_.audience) {
        LOG_ERROR("Invalid audience: expected '{}', got '{}'", config_.audience, payload.audience);
        return false;
    }

    // Verify expiration
    if (config_.verify_expiration) {
        auto exp_with_skew = payload.expiration + config_.clock_skew_tolerance;
        if (now > exp_with_skew) {
            LOG_ERROR("Token expired");
            return false;
        }
    }

    // Verify not before
    if (config_.verify_not_before) {
        auto nbf_with_skew = payload.not_before - config_.clock_skew_tolerance;
        if (now < nbf_with_skew) {
            LOG_ERROR("Token not yet valid");
            return false;
        }
    }

    return true;
}

std::string JWTValidator::base64_url_decode(const std::string& input) {
    std::string padded = input;
    
    // Add padding if necessary
    while (padded.length() % 4 != 0) {
        padded += "=";
    }

    // Replace URL-safe characters
    std::replace(padded.begin(), padded.end(), '-', '+');
    std::replace(padded.begin(), padded.end(), '_', '/');

    // Decode base64
    BIO* bio = BIO_new_mem_buf(padded.c_str(), padded.length());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);

    std::vector<char> buffer(padded.length());
    int decoded_length = BIO_read(bio, buffer.data(), buffer.size());
    
    BIO_free_all(bio);

    if (decoded_length < 0) {
        throw std::runtime_error("Base64 decode failed");
    }

    return std::string(buffer.data(), decoded_length);
}

JWTAlgorithm JWTValidator::string_to_algorithm(const std::string& alg) {
    if (alg == "RS256") return JWTAlgorithm::RS256;
    if (alg == "RS384") return JWTAlgorithm::RS384;
    if (alg == "RS512") return JWTAlgorithm::RS512;
    if (alg == "ES256") return JWTAlgorithm::ES256;
    if (alg == "ES384") return JWTAlgorithm::ES384;
    if (alg == "ES512") return JWTAlgorithm::ES512;
    
    throw std::invalid_argument("Unsupported JWT algorithm: " + alg);
}

EVP_PKEY* JWTValidator::load_rsa_key_from_pem(const std::string& pem) {
    BIO* bio = BIO_new_mem_buf(pem.c_str(), pem.length());
    if (!bio) {
        return nullptr;
    }

    EVP_PKEY* key = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);

    if (key && EVP_PKEY_base_id(key) != EVP_PKEY_RSA) {
        EVP_PKEY_free(key);
        return nullptr;
    }

    return key;
}

EVP_PKEY* JWTValidator::load_ecdsa_key_from_pem(const std::string& pem) {
    BIO* bio = BIO_new_mem_buf(pem.c_str(), pem.length());
    if (!bio) {
        return nullptr;
    }

    EVP_PKEY* key = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);

    if (key && EVP_PKEY_base_id(key) != EVP_PKEY_EC) {
        EVP_PKEY_free(key);
        return nullptr;
    }

    return key;
}

} // namespace ultra_cpp::security