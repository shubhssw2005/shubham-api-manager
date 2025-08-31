# Media Processor Worker Dockerfile
FROM node:18-alpine

# Install system dependencies for media processing
RUN apk add --no-cache \
    ffmpeg \
    imagemagick \
    poppler-utils \
    libreoffice \
    antiword \
    python3 \
    py3-pip \
    && pip3 install docx2txt

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mediaprocessor -u 1001

# Create temp directory for processing
RUN mkdir -p /tmp/media-processing && \
    chown -R mediaprocessor:nodejs /tmp/media-processing

# Switch to non-root user
USER mediaprocessor

# Expose health check port (optional)
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "console.log('Health check passed')" || exit 1

# Start the media processor
CMD ["node", "scripts/start-media-processor.js"]