#pragma once

#include "common/types.hpp"
#include <string_view>
#include <vector>
#include <memory>

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

// HTTP/1.1 and HTTP/2 protocol definitions
enum class HttpVersion {
    HTTP_1_0,
    HTTP_1_1,
    HTTP_2_0
};

enum class HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
    OPTIONS,
    PATCH,
    CONNECT,
    TRACE,
    UNKNOWN
};

struct HttpHeader {
    std::string_view name;
    std::string_view value;
};

struct HttpRequest {
    HttpMethod method;
    HttpVersion version;
    std::string_view path;
    std::string_view query_string;
    std::vector<HttpHeader> headers;
    std::string_view body;
    timestamp_t received_at;
    u32 stream_id = 0; // For HTTP/2
};

struct HttpResponse {
    HttpVersion version;
    u16 status_code = 200;
    std::vector<HttpHeader> headers;
    std::string_view body;
    timestamp_t sent_at;
    u32 stream_id = 0; // For HTTP/2
};

// Zero-copy HTTP/1.1 parser with SIMD optimization
class Http11Parser {
public:
    enum class ParseResult {
        COMPLETE,
        INCOMPLETE,
        ERROR,
        NEED_MORE_DATA
    };
    
    struct ParseState {
        const char* buffer_start;
        const char* buffer_end;
        const char* current_pos;
        size_t bytes_parsed = 0;
        bool headers_complete = false;
        size_t content_length = 0;
        bool chunked_encoding = false;
    };
    
    Http11Parser();
    ~Http11Parser();
    
    // Zero-copy parsing - no data copying, only views into original buffer
    ParseResult parse_request(const char* data, size_t length, 
                            HttpRequest& request, ParseState& state) noexcept;
    
    ParseResult parse_response(const char* data, size_t length,
                             HttpResponse& response, ParseState& state) noexcept;
    
    // Fast method parsing using SIMD
    HttpMethod parse_method_simd(const char* data, size_t length) noexcept;
    
    // Fast header parsing with zero-copy
    bool parse_headers_simd(const char* start, const char* end,
                          std::vector<HttpHeader>& headers) noexcept;
    
    // Optimized status line parsing
    bool parse_status_line(const char* start, const char* end,
                         HttpVersion& version, u16& status_code) noexcept;
    
private:
    // SIMD-optimized string search functions
    const char* find_crlf_simd(const char* start, const char* end) noexcept;
    const char* find_char_simd(const char* start, const char* end, char target) noexcept;
    const char* find_space_simd(const char* start, const char* end) noexcept;
    
    // Fast case-insensitive header name comparison
    bool header_name_equals_simd(const char* name1, size_t len1,
                               const char* name2, size_t len2) noexcept;
    
    // Parse content-length header value
    size_t parse_content_length(std::string_view value) noexcept;
    
    // Check for chunked transfer encoding
    bool is_chunked_encoding(const std::vector<HttpHeader>& headers) noexcept;
};

// HTTP/2 frame types
enum class Http2FrameType : u8 {
    DATA = 0x0,
    HEADERS = 0x1,
    PRIORITY = 0x2,
    RST_STREAM = 0x3,
    SETTINGS = 0x4,
    PUSH_PROMISE = 0x5,
    PING = 0x6,
    GOAWAY = 0x7,
    WINDOW_UPDATE = 0x8,
    CONTINUATION = 0x9
};

// HTTP/2 frame flags
enum class Http2FrameFlags : u8 {
    NONE = 0x0,
    END_STREAM = 0x1,
    END_HEADERS = 0x4,
    PADDED = 0x8,
    PRIORITY = 0x20
};

struct Http2Frame {
    u32 length : 24;
    Http2FrameType type;
    Http2FrameFlags flags;
    u32 stream_id;
    const u8* payload;
};

// Zero-copy HTTP/2 parser with binary frame processing
class Http2Parser {
public:
    enum class ParseResult {
        COMPLETE,
        INCOMPLETE,
        ERROR,
        NEED_MORE_DATA,
        CONNECTION_ERROR,
        STREAM_ERROR
    };
    
    struct ConnectionState {
        bool connection_preface_received = false;
        u32 max_frame_size = 16384;
        u32 max_header_list_size = 8192;
        u32 initial_window_size = 65535;
        bool server_push_enabled = true;
    };
    
    struct StreamState {
        u32 stream_id;
        enum State {
            IDLE,
            RESERVED_LOCAL,
            RESERVED_REMOTE,
            OPEN,
            HALF_CLOSED_LOCAL,
            HALF_CLOSED_REMOTE,
            CLOSED
        } state = IDLE;
        
        u32 window_size = 65535;
        bool headers_complete = false;
        std::vector<HttpHeader> headers;
        std::string_view body;
    };
    
    Http2Parser();
    ~Http2Parser();
    
    // Parse HTTP/2 connection preface
    ParseResult parse_connection_preface(const char* data, size_t length) noexcept;
    
    // Parse HTTP/2 frames with zero-copy
    ParseResult parse_frame(const char* data, size_t length,
                          Http2Frame& frame, size_t& bytes_consumed) noexcept;
    
    // Process specific frame types
    ParseResult process_headers_frame(const Http2Frame& frame,
                                    StreamState& stream) noexcept;
    
    ParseResult process_data_frame(const Http2Frame& frame,
                                 StreamState& stream) noexcept;
    
    ParseResult process_settings_frame(const Http2Frame& frame,
                                     ConnectionState& conn_state) noexcept;
    
    // HPACK header compression/decompression
    bool decompress_headers(const u8* data, size_t length,
                          std::vector<HttpHeader>& headers) noexcept;
    
    bool compress_headers(const std::vector<HttpHeader>& headers,
                        std::vector<u8>& compressed_data) noexcept;
    
    // Stream management
    StreamState* get_stream(u32 stream_id) noexcept;
    StreamState* create_stream(u32 stream_id) noexcept;
    void close_stream(u32 stream_id) noexcept;
    
private:
    ConnectionState connection_state_;
    std::unordered_map<u32, std::unique_ptr<StreamState>> streams_;
    
    // HPACK dynamic table for header compression
    struct HpackEntry {
        std::string name;
        std::string value;
        size_t size() const { return name.size() + value.size() + 32; }
    };
    
    std::vector<HpackEntry> dynamic_table_;
    size_t dynamic_table_size_ = 0;
    size_t max_dynamic_table_size_ = 4096;
    
    // Frame validation
    bool validate_frame_header(const Http2Frame& frame) noexcept;
    bool validate_stream_state(u32 stream_id, Http2FrameType frame_type) noexcept;
    
    // HPACK helper functions
    bool decode_integer(const u8*& data, const u8* end, u32& value, u8 prefix_bits) noexcept;
    bool decode_string(const u8*& data, const u8* end, std::string_view& str) noexcept;
    bool huffman_decode(const u8* data, size_t length, std::string& decoded) noexcept;
};

// Protocol detection and routing
class ProtocolDetector {
public:
    enum class Protocol {
        HTTP_1_0,
        HTTP_1_1,
        HTTP_2_0,
        UNKNOWN
    };
    
    // Detect protocol from initial bytes
    Protocol detect_protocol(const char* data, size_t length) noexcept;
    
    // Check for HTTP/2 connection preface
    bool is_http2_preface(const char* data, size_t length) noexcept;
    
    // Check for HTTP/1.x request line
    bool is_http1_request(const char* data, size_t length) noexcept;
    
private:
    static constexpr const char* HTTP2_PREFACE = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    static constexpr size_t HTTP2_PREFACE_LENGTH = 24;
};

} // namespace ultra::network