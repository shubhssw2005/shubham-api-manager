# Ultra Low-Latency C++ System Makefile

.PHONY: all debug release clean test install setup docker

# Default target
all: release

# Setup development environment
setup:
	@echo "Setting up development environment..."
	@./scripts/setup-dev-env.sh

# Debug build
debug:
	@echo "Building debug version..."
	@./scripts/build.sh Debug

# Release build  
release:
	@echo "Building release version..."
	@./scripts/build.sh Release

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/

# Simple build without Conan
simple:
	@echo "Building with simple configuration..."
	@./simple-build.sh

# Run tests
test: release
	@echo "Running tests..."
	@cd build/Release && ctest --output-on-failure

# Install system-wide
install: release
	@echo "Installing system-wide..."
	@cd build/Release && sudo make install

# Docker build
docker:
	@echo "Building Docker image..."
	@docker build -t ultra-cpp:latest .

# Docker compose
docker-up:
	@echo "Starting with Docker Compose..."
	@docker-compose up --build

docker-down:
	@echo "Stopping Docker Compose..."
	@docker-compose down

# Development helpers
format:
	@echo "Formatting code..."
	@find src include tests -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" | xargs clang-format -i

lint:
	@echo "Running static analysis..."
	@clang-tidy src/**/*.cpp -- -Iinclude

# Performance profiling
profile: release
	@echo "Running performance profile..."
	@cd build/Release && perf record -g ./bin/ultra-api-gateway
	@cd build/Release && perf report

# Memory check
memcheck: debug
	@echo "Running memory check..."
	@cd build/Debug && valgrind --tool=memcheck --leak-check=full ./bin/unit_tests

# Help
help:
	@echo "Available targets:"
	@echo "  setup     - Setup development environment"
	@echo "  debug     - Build debug version"
	@echo "  release   - Build release version (default)"
	@echo "  clean     - Clean build artifacts"
	@echo "  test      - Run test suite"
	@echo "  install   - Install system-wide"
	@echo "  docker    - Build Docker image"
	@echo "  docker-up - Start with Docker Compose"
	@echo "  format    - Format source code"
	@echo "  lint      - Run static analysis"
	@echo "  profile   - Run performance profiling"
	@echo "  memcheck  - Run memory leak detection"