#include "common/error_handling.hpp"
#include "common/logger.hpp"

namespace ultra {
namespace common {

ErrorHandler::ErrorCallback ErrorHandler::global_callback_ = nullptr;

void ErrorHandler::set_global_error_handler(ErrorCallback callback) {
    global_callback_ = callback;
}

void ErrorHandler::handle_error(const std::exception& e) {
    if (global_callback_) {
        global_callback_(e);
    } else {
        LOG_ERROR("Unhandled exception: {}", e.what());
    }
}

ErrorHandler::ErrorScope::ErrorScope(ErrorCallback callback) 
    : previous_callback_(global_callback_) {
    global_callback_ = callback;
}

ErrorHandler::ErrorScope::~ErrorScope() {
    global_callback_ = previous_callback_;
}

} // namespace common
} // namespace ultra