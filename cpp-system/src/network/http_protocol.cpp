#include "network/http_protocol.hpp"
#ifdef ULTRA_LOGGER_AVAILABLE
    #include "common/logger.hpp"
#else
    #include "common/simple_logger.hpp"
#endif
#include <algorithm>
#include <cstring>
#include <cctype>

// SIMD support detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ULTRA_SIMD_X86_AVAILABLE 1
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ULTRA_SIMD_ARM_AVAILABLE 1
    #include <arm_neon.h>
#else
    #define ULTRA_SIMD_FALLBACK 1
#endif

namespace ultra::network {

// HTTP method string mappings
static const std::pair<const char*, HttpMethod> HTTP_METHODS[] = {
    {"GET", HttpMethod::GET},
    {"POST", HttpMethod::POST},
    {"PUT", HttpMethod::PUT},
    {"DELETE", HttpMethod::DELETE},
    {"HEAD", HttpMethod::HEAD},
    {"OPTIONS", HttpMethod::OPTIONS},
    {"PATCH", HttpMethod::PATCH},
    {"CONNECT", HttpMethod::CONNECT},
    {"TRACE", HttpMethod::TRACE}
};

// HTTP/1.1 Parser Implementation
Http11Parser::Http11Parser() = default;
Http11Parser::~Http11Parser() = default;

Http11Parser::ParseResult Http11Parser::parse_request(const char* data, size_t length,
                                                    HttpRequest& request, ParseState& state) noexcept {
    if (!data || length == 0) {
        return ParseResult::ERROR;
    }
    
    state.buffer_start = data;
    state.buffer_end = data + length;
    state.current_pos = data;
    
    try {
        // Parse request line: METHOD PATH HTTP/VERSION\r\n
        const char* line_end = find_crlf_simd(state.current_pos, state.buffer_end);
        if (line_end == state.buffer_end) {
            return ParseResult::INCOMPLETE;
        }
        
        // Parse method
        const char* method_end = find_space_simd(state.current_pos, line_end);
        if (method_end == line_end) {
            return ParseResult::ERROR;
        }
        
        request.method = parse_method_simd(state.current_pos, method_end - state.current_pos);
        if (request.method == HttpMethod::UNKNOWN) {
            return ParseResult::ERROR;
        }
        
        state.current_pos = method_end + 1;
        
        // Parse path
        const char* path_end = find_space_simd(state.current_pos, line_end);
        if (path_end == line_end) {
            return ParseResult::ERROR;
        }
        
        // Extract path and query string
        const char* query_start = std::find(state.current_pos, path_end, '?');
        if (query_start != path_end) {
            request.path = std::string_view(state.current_pos, query_start - state.current_pos);
            request.query_string = std::string_view(query_start + 1, path_end - query_start - 1);
        } else {
            request.path = std::string_view(state.current_pos, path_end - state.current_pos);
        }
        
        state.current_pos = path_end + 1;
        
        // Parse HTTP version
        if (std::strncmp(state.current_pos, "HTTP/1.1", 8) == 0) {
            request.version = HttpVersion::HTTP_1_1;
        } else if (std::strncmp(state.current_pos, "HTTP/1.0", 8) == 0) {
            request.version = HttpVersion::HTTP_1_0;
        } else if (std::strncmp(state.current_pos, "HTTP/2.0", 8) == 0) {
            request.version = HttpVersion::HTTP_2_0;
        } else {
            return ParseResult::ERROR;
        }
        
        state.current_pos = line_end + 2; // Skip CRLF
        
        // Parse headers
        request.headers.clear();
        if (!parse_headers_simd(state.current_pos, state.buffer_end, request.headers)) {
            return ParseResult::INCOMPLETE;
        }
        
        // Find end of headers
        const char* headers_end = state.current_pos;
        while (headers_end < state.buffer_end) {
            const char* header_line_end = find_crlf_simd(headers_end, state.buffer_end);
            if (header_line_end == state.buffer_end) {
                return ParseResult::INCOMPLETE;
            }
            
            if (header_line_end == headers_end) {
                // Empty line - end of headers
                headers_end += 2;
                break;
            }
            
            headers_end = header_line_end + 2;
        }
        
        state.current_pos = headers_end;
        state.headers_complete = true;
        
        // Check for content-length or chunked encoding
        state.content_length = 0;
        state.chunked_encoding = false;
        
        for (const auto& header : request.headers) {
            if (header_name_equals_simd(header.name.data(), header.name.length(), "content-length", 14)) {
                state.content_length = parse_content_length(header.value);
            } else if (header_name_equals_simd(header.name.data(), header.name.length(), "transfer-encoding", 17)) {
                if (header.value.find("chunked") != std::string_view::npos) {
                    state.chunked_encoding = true;
                }
            }
        }
        
        // Parse body if present
        if (state.content_length > 0) {
            size_t remaining = state.buffer_end - state.current_pos;
            if (remaining < state.content_length) {
                return ParseResult::INCOMPLETE;
            }
            request.body = std::string_view(state.current_pos, state.content_length);
            state.current_pos += state.content_length;
        } else if (state.chunked_encoding) {
            // TODO: Implement chunked encoding parsing
            request.body = std::string_view(state.current_pos, state.buffer_end - state.current_pos);
        }
        
        request.received_at = std::chrono::high_resolution_clock::now();
        state.bytes_parsed = state.current_pos - state.buffer_start;
        
        return ParseResult::COMPLETE;
        
    } catch (const std::exception& e) {
        ULTRA_LOG_ERROR("HTTP parsing error: {}", e.what());
        return ParseResult::ERROR;
    }
}

HttpMethod Http11Parser::parse_method_simd(const char* data, size_t length) noexcept {
    if (!data || length == 0 || length > 8) {
        return HttpMethod::UNKNOWN;
    }
    
    // Fast path for common methods using SIMD comparison
    if (length == 3) {
        // Check for GET
        if (data[0] == 'G' && data[1] == 'E' && data[2] == 'T') {
            return HttpMethod::GET;
        }
        // Check for PUT
        if (data[0] == 'P' && data[1] == 'U' && data[2] == 'T') {
            return HttpMethod::PUT;
        }
    } else if (length == 4) {
        // Check for POST, HEAD
        u32 method_int;
        std::memcpy(&method_int, data, 4);
        
        if (method_int == 0x54534F50) { // "POST" in little-endian
            return HttpMethod::POST;
        }
        if (method_int == 0x44414548) { // "HEAD" in little-endian
            return HttpMethod::HEAD;
        }
    }
    
    // Fallback to string comparison for other methods
    for (const auto& [method_str, method_enum] : HTTP_METHODS) {
        if (std::strncmp(data, method_str, length) == 0 && 
            std::strlen(method_str) == length) {
            return method_enum;
        }
    }
    
    return HttpMethod::UNKNOWN;
}

bool Http11Parser::parse_headers_simd(const char* start, const char* end,
                                    std::vector<HttpHeader>& headers) noexcept {
    const char* pos = start;
    
    while (pos < end) {
        // Find end of header line
        const char* line_end = find_crlf_simd(pos, end);
        if (line_end == end) {
            return false; // Incomplete
        }
        
        // Empty line indicates end of headers
        if (line_end == pos) {
            return true;
        }
        
        // Find colon separator
        const char* colon = find_char_simd(pos, line_end, ':');
        if (colon == line_end) {
            return false; // Invalid header
        }
        
        // Extract header name (trim whitespace)
        std::string_view name(pos, colon - pos);
        while (!name.empty() && std::isspace(name.back())) {
            name.remove_suffix(1);
        }
        
        // Extract header value (skip colon and leading whitespace)
        const char* value_start = colon + 1;
        while (value_start < line_end && std::isspace(*value_start)) {
            ++value_start;
        }
        
        std::string_view value(value_start, line_end - value_start);
        while (!value.empty() && std::isspace(value.back())) {
            value.remove_suffix(1);
        }
        
        headers.emplace_back(HttpHeader{name, value});
        pos = line_end + 2; // Skip CRLF
    }
    
    return false; // Incomplete headers
}

const char* Http11Parser::find_crlf_simd(const char* start, const char* end) noexcept {
    const char* pos = start;
    
#ifdef ULTRA_SIMD_X86_AVAILABLE
    // Use x86 SIMD to find CRLF sequence quickly
    while (pos + 16 <= end) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
        __m128i cr = _mm_set1_epi8('\r');
        __m128i cmp = _mm_cmpeq_epi8(chunk, cr);
        
        int mask = _mm_movemask_epi8(cmp);
        if (mask != 0) {
            int offset = __builtin_ctz(mask);
            if (pos + offset + 1 < end && pos[offset + 1] == '\n') {
                return pos + offset;
            }
        }
        pos += 16;
    }
#elif defined(ULTRA_SIMD_ARM_AVAILABLE)
    // Use ARM NEON to find CRLF sequence quickly
    while (pos + 16 <= end) {
        uint8x16_t chunk = vld1q_u8(reinterpret_cast<const uint8_t*>(pos));
        uint8x16_t cr = vdupq_n_u8('\r');
        uint8x16_t cmp = vceqq_u8(chunk, cr);
        
        uint64x2_t cmp64 = vreinterpretq_u64_u8(cmp);
        uint64_t mask = vgetq_lane_u64(cmp64, 0) | vgetq_lane_u64(cmp64, 1);
        
        if (mask != 0) {
            // Find first set bit (simplified)
            for (int i = 0; i < 16; ++i) {
                if (pos[i] == '\r' && pos + i + 1 < end && pos[i + 1] == '\n') {
                    return pos + i;
                }
            }
        }
        pos += 16;
    }
#endif
    
    // Fallback: Handle remaining bytes or when SIMD not available
    while (pos + 1 < end) {
        if (*pos == '\r' && *(pos + 1) == '\n') {
            return pos;
        }
        ++pos;
    }
    
    return end;
}

const char* Http11Parser::find_char_simd(const char* start, const char* end, char target) noexcept {
    const char* pos = start;
    
#ifdef ULTRA_SIMD_X86_AVAILABLE
    // Use x86 SIMD to find character quickly
    while (pos + 16 <= end) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
        __m128i target_vec = _mm_set1_epi8(target);
        __m128i cmp = _mm_cmpeq_epi8(chunk, target_vec);
        
        int mask = _mm_movemask_epi8(cmp);
        if (mask != 0) {
            return pos + __builtin_ctz(mask);
        }
        pos += 16;
    }
#elif defined(ULTRA_SIMD_ARM_AVAILABLE)
    // Use ARM NEON to find character quickly
    while (pos + 16 <= end) {
        uint8x16_t chunk = vld1q_u8(reinterpret_cast<const uint8_t*>(pos));
        uint8x16_t target_vec = vdupq_n_u8(target);
        uint8x16_t cmp = vceqq_u8(chunk, target_vec);
        
        uint64x2_t cmp64 = vreinterpretq_u64_u8(cmp);
        uint64_t mask = vgetq_lane_u64(cmp64, 0) | vgetq_lane_u64(cmp64, 1);
        
        if (mask != 0) {
            // Find first occurrence
            for (int i = 0; i < 16; ++i) {
                if (pos[i] == target) {
                    return pos + i;
                }
            }
        }
        pos += 16;
    }
#endif
    
    // Fallback: Handle remaining bytes or when SIMD not available
    while (pos < end && *pos != target) {
        ++pos;
    }
    
    return pos;
}

const char* Http11Parser::find_space_simd(const char* start, const char* end) noexcept {
    return find_char_simd(start, end, ' ');
}

bool Http11Parser::header_name_equals_simd(const char* name1, size_t len1,
                                         const char* name2, size_t len2) noexcept {
    if (len1 != len2) {
        return false;
    }
    
    size_t i = 0;
    
#ifdef ULTRA_SIMD_X86_AVAILABLE
    // Case-insensitive comparison using x86 SIMD when possible
    while (i + 16 <= len1) {
        __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(name1 + i));
        __m128i chunk2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(name2 + i));
        
        // Convert to lowercase
        __m128i mask_upper = _mm_cmpgt_epi8(chunk1, _mm_set1_epi8('A' - 1));
        __m128i mask_lower = _mm_cmplt_epi8(chunk1, _mm_set1_epi8('Z' + 1));
        __m128i mask = _mm_and_si128(mask_upper, mask_lower);
        __m128i to_lower = _mm_and_si128(mask, _mm_set1_epi8(0x20));
        chunk1 = _mm_or_si128(chunk1, to_lower);
        
        mask_upper = _mm_cmpgt_epi8(chunk2, _mm_set1_epi8('A' - 1));
        mask_lower = _mm_cmplt_epi8(chunk2, _mm_set1_epi8('Z' + 1));
        mask = _mm_and_si128(mask_upper, mask_lower);
        to_lower = _mm_and_si128(mask, _mm_set1_epi8(0x20));
        chunk2 = _mm_or_si128(chunk2, to_lower);
        
        __m128i cmp = _mm_cmpeq_epi8(chunk1, chunk2);
        if (_mm_movemask_epi8(cmp) != 0xFFFF) {
            return false;
        }
        
        i += 16;
    }
#elif defined(ULTRA_SIMD_ARM_AVAILABLE)
    // Case-insensitive comparison using ARM NEON
    while (i + 16 <= len1) {
        uint8x16_t chunk1 = vld1q_u8(reinterpret_cast<const uint8_t*>(name1 + i));
        uint8x16_t chunk2 = vld1q_u8(reinterpret_cast<const uint8_t*>(name2 + i));
        
        // Convert to lowercase (simplified)
        uint8x16_t upper_a = vdupq_n_u8('A');
        uint8x16_t upper_z = vdupq_n_u8('Z');
        uint8x16_t to_lower = vdupq_n_u8(0x20);
        
        uint8x16_t mask1 = vandq_u8(vcgeq_u8(chunk1, upper_a), vcleq_u8(chunk1, upper_z));
        uint8x16_t mask2 = vandq_u8(vcgeq_u8(chunk2, upper_a), vcleq_u8(chunk2, upper_z));
        
        chunk1 = vorrq_u8(chunk1, vandq_u8(mask1, to_lower));
        chunk2 = vorrq_u8(chunk2, vandq_u8(mask2, to_lower));
        
        uint8x16_t cmp = vceqq_u8(chunk1, chunk2);
        uint64x2_t cmp64 = vreinterpretq_u64_u8(cmp);
        
        if ((vgetq_lane_u64(cmp64, 0) & vgetq_lane_u64(cmp64, 1)) != 0xFFFFFFFFFFFFFFFFULL) {
            return false;
        }
        
        i += 16;
    }
#endif
    
    // Handle remaining bytes
    while (i < len1) {
        char c1 = std::tolower(name1[i]);
        char c2 = std::tolower(name2[i]);
        if (c1 != c2) {
            return false;
        }
        ++i;
    }
    
    return true;
}

size_t Http11Parser::parse_content_length(std::string_view value) noexcept {
    size_t result = 0;
    for (char c : value) {
        if (c >= '0' && c <= '9') {
            result = result * 10 + (c - '0');
        } else if (!std::isspace(c)) {
            return 0; // Invalid content-length
        }
    }
    return result;
}

// HTTP/2 Parser Implementation
Http2Parser::Http2Parser() = default;
Http2Parser::~Http2Parser() = default;

Http2Parser::ParseResult Http2Parser::parse_connection_preface(const char* data, size_t length) noexcept {
    static constexpr const char* HTTP2_PREFACE = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    static constexpr size_t HTTP2_PREFACE_LENGTH = 24;
    
    if (length < HTTP2_PREFACE_LENGTH) {
        return ParseResult::INCOMPLETE;
    }
    
    if (std::memcmp(data, HTTP2_PREFACE, HTTP2_PREFACE_LENGTH) == 0) {
        connection_state_.connection_preface_received = true;
        return ParseResult::COMPLETE;
    }
    
    return ParseResult::CONNECTION_ERROR;
}

Http2Parser::ParseResult Http2Parser::parse_frame(const char* data, size_t length,
                                                Http2Frame& frame, size_t& bytes_consumed) noexcept {
    if (length < 9) { // Minimum frame header size
        return ParseResult::INCOMPLETE;
    }
    
    // Parse frame header (9 bytes)
    const u8* header = reinterpret_cast<const u8*>(data);
    
    // Length (24 bits)
    frame.length = (header[0] << 16) | (header[1] << 8) | header[2];
    
    // Type (8 bits)
    frame.type = static_cast<Http2FrameType>(header[3]);
    
    // Flags (8 bits)
    frame.flags = static_cast<Http2FrameFlags>(header[4]);
    
    // Stream ID (31 bits, R bit reserved)
    frame.stream_id = ((header[5] & 0x7F) << 24) | (header[6] << 16) | (header[7] << 8) | header[8];
    
    // Check if we have the complete frame
    if (length < 9 + frame.length) {
        return ParseResult::INCOMPLETE;
    }
    
    // Validate frame
    if (!validate_frame_header(frame)) {
        return ParseResult::CONNECTION_ERROR;
    }
    
    // Set payload pointer
    frame.payload = reinterpret_cast<const u8*>(data + 9);
    bytes_consumed = 9 + frame.length;
    
    return ParseResult::COMPLETE;
}

bool Http2Parser::validate_frame_header(const Http2Frame& frame) noexcept {
    // Check maximum frame size
    if (frame.length > connection_state_.max_frame_size) {
        return false;
    }
    
    // Validate frame type
    if (static_cast<u8>(frame.type) > static_cast<u8>(Http2FrameType::CONTINUATION)) {
        return false;
    }
    
    // Stream ID validation
    if (frame.type == Http2FrameType::SETTINGS || 
        frame.type == Http2FrameType::PING || 
        frame.type == Http2FrameType::GOAWAY) {
        // These frames must have stream ID 0
        if (frame.stream_id != 0) {
            return false;
        }
    } else {
        // Other frames must have non-zero stream ID
        if (frame.stream_id == 0) {
            return false;
        }
    }
    
    return true;
}

// Protocol Detector Implementation
ProtocolDetector::Protocol ProtocolDetector::detect_protocol(const char* data, size_t length) noexcept {
    if (!data || length == 0) {
        return Protocol::UNKNOWN;
    }
    
    // Check for HTTP/2 connection preface
    if (is_http2_preface(data, length)) {
        return Protocol::HTTP_2_0;
    }
    
    // Check for HTTP/1.x request
    if (is_http1_request(data, length)) {
        // Look for HTTP version in the request line
        const char* version_pos = std::strstr(data, "HTTP/");
        if (version_pos && version_pos < data + std::min(length, size_t(100))) {
            if (std::strncmp(version_pos, "HTTP/1.1", 8) == 0) {
                return Protocol::HTTP_1_1;
            } else if (std::strncmp(version_pos, "HTTP/1.0", 8) == 0) {
                return Protocol::HTTP_1_0;
            }
        }
        return Protocol::HTTP_1_1; // Default to HTTP/1.1
    }
    
    return Protocol::UNKNOWN;
}

bool ProtocolDetector::is_http2_preface(const char* data, size_t length) noexcept {
    if (length < HTTP2_PREFACE_LENGTH) {
        return false;
    }
    
    return std::memcmp(data, HTTP2_PREFACE, HTTP2_PREFACE_LENGTH) == 0;
}

bool ProtocolDetector::is_http1_request(const char* data, size_t length) noexcept {
    if (length < 4) {
        return false;
    }
    
    // Check for common HTTP methods
    static const char* methods[] = {"GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ", "PATCH "};
    
    for (const char* method : methods) {
        size_t method_len = std::strlen(method);
        if (length >= method_len && std::strncmp(data, method, method_len) == 0) {
            return true;
        }
    }
    
    return false;
}

} // namespace ultra::network