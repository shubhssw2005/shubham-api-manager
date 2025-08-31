#include "network/tcp_connection.hpp"
#ifdef ULTRA_LOGGER_AVAILABLE
    #include "common/logger.hpp"
#else
    #include "common/simple_logger.hpp"
#endif
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <errno.h>
#include <cstring>

// Platform-specific includes
#ifdef __linux__
    #include <sys/sendfile.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
    // macOS doesn't have sendfile in the same way
    #define sendfile(out_fd, in_fd, offset, count) (-1)
#endif

namespace ultra::network {

// TcpConnection Implementation
TcpConnection::TcpConnection() 
    : socket_fd_(-1), state_(State::DISCONNECTED) {
}

TcpConnection::TcpConnection(int socket_fd) 
    : socket_fd_(socket_fd), state_(State::CONNECTED) {
    if (socket_fd_ >= 0) {
        set_non_blocking();
        stats_.connected_at = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
}

TcpConnection::~TcpConnection() {
    close();
}

TcpConnection::TcpConnection(TcpConnection&& other) noexcept 
    : socket_fd_(other.socket_fd_), state_(other.state_), 
      config_(other.config_), stats_(other.stats_) {
    other.socket_fd_ = -1;
    other.state_ = State::DISCONNECTED;
}

TcpConnection& TcpConnection::operator=(TcpConnection&& other) noexcept {
    if (this != &other) {
        close();
        socket_fd_ = other.socket_fd_;
        state_ = other.state_;
        config_ = other.config_;
        stats_ = other.stats_;
        
        other.socket_fd_ = -1;
        other.state_ = State::DISCONNECTED;
    }
    return *this;
}

bool TcpConnection::bind_and_listen(const std::string& address, u16 port, const Config& config) noexcept {
    config_ = config;
    
    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        ULTRA_LOG_ERROR("Failed to create socket: {}", std::strerror(errno));
        return false;
    }
    
    // Configure socket
    if (!configure_socket(config_)) {
        close();
        return false;
    }
    
    // Bind to address
    struct sockaddr_in server_addr = {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (address.empty() || address == "0.0.0.0") {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, address.c_str(), &server_addr.sin_addr) <= 0) {
            ULTRA_LOG_ERROR("Invalid address: {}", address);
            close();
            return false;
        }
    }
    
    if (bind(socket_fd_, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        ULTRA_LOG_ERROR("Failed to bind to {}:{}: {}", address, port, std::strerror(errno));
        close();
        return false;
    }
    
    // Listen for connections
    if (listen(socket_fd_, SOMAXCONN) < 0) {
        ULTRA_LOG_ERROR("Failed to listen: {}", std::strerror(errno));
        close();
        return false;
    }
    
    state_ = State::CONNECTED;
    return true;
}

bool TcpConnection::connect(const std::string& address, u16 port, const Config& config) noexcept {
    config_ = config;
    
    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        ULTRA_LOG_ERROR("Failed to create socket: {}", std::strerror(errno));
        return false;
    }
    
    // Configure socket
    if (!configure_socket(config_)) {
        close();
        return false;
    }
    
    // Set non-blocking for timeout support
    if (!set_non_blocking()) {
        close();
        return false;
    }
    
    // Connect to server
    struct sockaddr_in server_addr = {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, address.c_str(), &server_addr.sin_addr) <= 0) {
        ULTRA_LOG_ERROR("Invalid address: {}", address);
        close();
        return false;
    }
    
    state_ = State::CONNECTING;
    
    int result = ::connect(socket_fd_, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr));
    if (result == 0) {
        // Connected immediately
        state_ = State::CONNECTED;
        stats_.connected_at = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return true;
    } else if (errno == EINPROGRESS) {
        // Connection in progress - use select/poll to wait with timeout
        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(socket_fd_, &write_fds);
        
        struct timeval timeout;
        timeout.tv_sec = config_.connect_timeout_ms / 1000;
        timeout.tv_usec = (config_.connect_timeout_ms % 1000) * 1000;
        
        int select_result = select(socket_fd_ + 1, nullptr, &write_fds, nullptr, &timeout);
        if (select_result > 0) {
            // Check if connection succeeded
            int error = 0;
            socklen_t error_len = sizeof(error);
            if (getsockopt(socket_fd_, SOL_SOCKET, SO_ERROR, &error, &error_len) == 0 && error == 0) {
                state_ = State::CONNECTED;
                stats_.connected_at = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                return true;
            }
        }
    }
    
    ULTRA_LOG_ERROR("Failed to connect to {}:{}: {}", address, port, std::strerror(errno));
    state_ = State::ERROR_STATE;
    close();
    return false;
}

std::unique_ptr<TcpConnection> TcpConnection::accept() noexcept {
    if (socket_fd_ < 0 || state_ != State::CONNECTED) {
        return nullptr;
    }
    
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    int client_fd = ::accept(socket_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            ULTRA_LOG_ERROR("Failed to accept connection: {}", std::strerror(errno));
        }
        return nullptr;
    }
    
    auto client_conn = std::make_unique<TcpConnection>(client_fd);
    client_conn->config_ = config_;
    
    return client_conn;
}

void TcpConnection::close() noexcept {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
    state_ = State::CLOSED;
}

ssize_t TcpConnection::send(const void* data, size_t length, int flags) noexcept {
    if (socket_fd_ < 0 || !data || length == 0) {
        return -1;
    }
    
    ssize_t bytes_sent = ::send(socket_fd_, data, length, flags | MSG_NOSIGNAL);
    if (bytes_sent > 0) {
        stats_.bytes_sent += bytes_sent;
        stats_.packets_sent++;
        stats_.last_activity = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    } else if (bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        state_ = State::ERROR_STATE;
    }
    
    return bytes_sent;
}

ssize_t TcpConnection::receive(void* buffer, size_t length, int flags) noexcept {
    if (socket_fd_ < 0 || !buffer || length == 0) {
        return -1;
    }
    
    ssize_t bytes_received = ::recv(socket_fd_, buffer, length, flags);
    if (bytes_received > 0) {
        stats_.bytes_received += bytes_received;
        stats_.packets_received++;
        stats_.last_activity = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    } else if (bytes_received == 0) {
        // Connection closed by peer
        state_ = State::CLOSED;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        state_ = State::ERROR_STATE;
    }
    
    return bytes_received;
}

ssize_t TcpConnection::sendfile(int file_fd, off_t offset, size_t count) noexcept {
    if (socket_fd_ < 0 || file_fd < 0) {
        return -1;
    }
    
    ssize_t bytes_sent = ::sendfile(socket_fd_, file_fd, &offset, count);
    if (bytes_sent > 0) {
        stats_.bytes_sent += bytes_sent;
        stats_.packets_sent++;
        stats_.last_activity = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    } else if (bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        state_ = State::ERROR_STATE;
    }
    
    return bytes_sent;
}

ssize_t TcpConnection::sendv(const struct iovec* iov, int iovcnt) noexcept {
    if (socket_fd_ < 0 || !iov || iovcnt <= 0) {
        return -1;
    }
    
    struct msghdr msg = {};
    msg.msg_iov = const_cast<struct iovec*>(iov);
    msg.msg_iovlen = iovcnt;
    
    ssize_t bytes_sent = sendmsg(socket_fd_, &msg, MSG_NOSIGNAL);
    if (bytes_sent > 0) {
        stats_.bytes_sent += bytes_sent;
        stats_.packets_sent++;
        stats_.last_activity = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    } else if (bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        state_ = State::ERROR_STATE;
    }
    
    return bytes_sent;
}

ssize_t TcpConnection::receivev(const struct iovec* iov, int iovcnt) noexcept {
    if (socket_fd_ < 0 || !iov || iovcnt <= 0) {
        return -1;
    }
    
    struct msghdr msg = {};
    msg.msg_iov = const_cast<struct iovec*>(iov);
    msg.msg_iovlen = iovcnt;
    
    ssize_t bytes_received = recvmsg(socket_fd_, &msg, 0);
    if (bytes_received > 0) {
        stats_.bytes_received += bytes_received;
        stats_.packets_received++;
        stats_.last_activity = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    } else if (bytes_received == 0) {
        state_ = State::CLOSED;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        state_ = State::ERROR_STATE;
    }
    
    return bytes_received;
}

bool TcpConnection::would_block() const noexcept {
    return errno == EAGAIN || errno == EWOULDBLOCK;
}

bool TcpConnection::is_connected() const noexcept {
    return state_ == State::CONNECTED && socket_fd_ >= 0;
}

bool TcpConnection::has_error() const noexcept {
    return state_ == State::ERROR_STATE;
}

struct sockaddr_in TcpConnection::get_local_address() const noexcept {
    struct sockaddr_in addr = {};
    if (socket_fd_ >= 0) {
        socklen_t addr_len = sizeof(addr);
        getsockname(socket_fd_, reinterpret_cast<struct sockaddr*>(&addr), &addr_len);
    }
    return addr;
}

struct sockaddr_in TcpConnection::get_remote_address() const noexcept {
    struct sockaddr_in addr = {};
    if (socket_fd_ >= 0) {
        socklen_t addr_len = sizeof(addr);
        getpeername(socket_fd_, reinterpret_cast<struct sockaddr*>(&addr), &addr_len);
    }
    return addr;
}

bool TcpConnection::configure_socket(const Config& config) noexcept {
    if (socket_fd_ < 0) {
        return false;
    }
    
    // Enable SO_REUSEPORT for load balancing
    if (config.enable_reuseport) {
        int enable = 1;
        if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable)) < 0) {
            ULTRA_LOG_WARN("Failed to set SO_REUSEPORT: {}", std::strerror(errno));
        }
    }
    
    // Enable SO_REUSEADDR
    int enable = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        ULTRA_LOG_WARN("Failed to set SO_REUSEADDR: {}", std::strerror(errno));
    }
    
    // Set TCP_NODELAY for low latency
    if (config.enable_nodelay) {
        if (!enable_tcp_nodelay()) {
            ULTRA_LOG_WARN("Failed to enable TCP_NODELAY");
        }
    }
    
    // Set keepalive
    if (config.enable_keepalive) {
        if (!enable_tcp_keepalive(config.keepalive_idle, config.keepalive_interval, config.keepalive_probes)) {
            ULTRA_LOG_WARN("Failed to enable TCP keepalive");
        }
    }
    
    // Set buffer sizes
    if (!set_send_buffer_size(config.send_buffer_size)) {
        ULTRA_LOG_WARN("Failed to set send buffer size");
    }
    
    if (!set_recv_buffer_size(config.recv_buffer_size)) {
        ULTRA_LOG_WARN("Failed to set receive buffer size");
    }
    
    // Enable TCP Fast Open
    if (config.enable_fast_open) {
        enable_tcp_fastopen();
    }
    
    return true;
}

bool TcpConnection::set_non_blocking() noexcept {
    if (socket_fd_ < 0) {
        return false;
    }
    
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    
    return fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) >= 0;
}

bool TcpConnection::enable_tcp_nodelay() noexcept {
    int enable = 1;
    return set_socket_option(IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable));
}

bool TcpConnection::enable_tcp_keepalive(u32 idle, u32 interval, u32 probes) noexcept {
    int enable = 1;
    if (!set_socket_option(SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable))) {
        return false;
    }
    
    int idle_val = static_cast<int>(idle);
    int interval_val = static_cast<int>(interval);
    int probes_val = static_cast<int>(probes);
    
    return set_socket_option(IPPROTO_TCP, TCP_KEEPIDLE, &idle_val, sizeof(idle_val)) &&
           set_socket_option(IPPROTO_TCP, TCP_KEEPINTVL, &interval_val, sizeof(interval_val)) &&
           set_socket_option(IPPROTO_TCP, TCP_KEEPCNT, &probes_val, sizeof(probes_val));
}

bool TcpConnection::set_send_buffer_size(u32 size) noexcept {
    int size_val = static_cast<int>(size);
    return set_socket_option(SOL_SOCKET, SO_SNDBUF, &size_val, sizeof(size_val));
}

bool TcpConnection::set_recv_buffer_size(u32 size) noexcept {
    int size_val = static_cast<int>(size);
    return set_socket_option(SOL_SOCKET, SO_RCVBUF, &size_val, sizeof(size_val));
}

bool TcpConnection::enable_tcp_fastopen() noexcept {
#ifdef TCP_FASTOPEN
    int enable = 1;
    return set_socket_option(IPPROTO_TCP, TCP_FASTOPEN, &enable, sizeof(enable));
#else
    return false;
#endif
}

bool TcpConnection::set_socket_option(int level, int optname, const void* optval, socklen_t optlen) noexcept {
    if (socket_fd_ < 0) {
        return false;
    }
    
    return setsockopt(socket_fd_, level, optname, optval, optlen) == 0;
}

bool TcpConnection::get_socket_option(int level, int optname, void* optval, socklen_t* optlen) noexcept {
    if (socket_fd_ < 0) {
        return false;
    }
    
    return getsockopt(socket_fd_, level, optname, optval, optlen) == 0;
}

TcpConnection::Stats TcpConnection::get_stats() const noexcept {
    update_stats();
    return stats_;
}

void TcpConnection::update_stats() const noexcept {
    if (socket_fd_ < 0) {
        return;
    }
    
    // Get TCP info for additional statistics
    struct tcp_info tcp_info;
    socklen_t tcp_info_len = sizeof(tcp_info);
    
    if (getsockopt(socket_fd_, IPPROTO_TCP, TCP_INFO, &tcp_info, &tcp_info_len) == 0) {
        stats_.retransmissions = tcp_info.tcpi_total_retrans;
        stats_.rtt_microseconds = tcp_info.tcpi_rtt;
    }
}

} // namespace ultra::network