# Multi-stage build for Media service (Go-based)
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY services/media/ ./

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o media-service .

# Production stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata ffmpeg imagemagick

# Create non-root user
RUN addgroup -g 1001 -S appuser && adduser -S appuser -u 1001 -G appuser

WORKDIR /app

# Copy binary
COPY --from=builder --chown=appuser:appuser /app/media-service .

# Set environment
ENV GIN_MODE=release
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Switch to non-root user
USER appuser

EXPOSE 8080

CMD ["./media-service"]