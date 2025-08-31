#pragma once

#include <iostream>
#include <string>

// Simple logger fallback when full logger is not available
#ifndef ULTRA_LOG_INFO
#define ULTRA_LOG_INFO(msg, ...) std::cout << "[INFO] " << msg << std::endl
#endif

#ifndef ULTRA_LOG_ERROR
#define ULTRA_LOG_ERROR(msg, ...) std::cerr << "[ERROR] " << msg << std::endl
#endif

#ifndef ULTRA_LOG_WARN
#define ULTRA_LOG_WARN(msg, ...) std::cout << "[WARN] " << msg << std::endl
#endif

#ifndef ULTRA_LOG_DEBUG
#define ULTRA_LOG_DEBUG(msg, ...) std::cout << "[DEBUG] " << msg << std::endl
#endif