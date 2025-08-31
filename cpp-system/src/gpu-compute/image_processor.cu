#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace ultra {
namespace gpu {

// Bilinear interpolation for image resizing
__device__ float bilinear_interpolate(const uint8_t* image, int width, int height, 
                                    int channels, float x, float y, int channel) {
    int x1 = (int)floorf(x);
    int y1 = (int)floorf(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    // Clamp coordinates
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    x2 = max(0, min(x2, width - 1));
    y2 = max(0, min(y2, height - 1));
    
    float dx = x - x1;
    float dy = y - y1;
    
    // Get pixel values
    float p11 = image[(y1 * width + x1) * channels + channel];
    float p12 = image[(y2 * width + x1) * channels + channel];
    float p21 = image[(y1 * width + x2) * channels + channel];
    float p22 = image[(y2 * width + x2) * channels + channel];
    
    // Bilinear interpolation
    float p1 = p11 * (1 - dx) + p21 * dx;
    float p2 = p12 * (1 - dx) + p22 * dx;
    
    return p1 * (1 - dy) + p2 * dy;
}

// Image resize kernel
__global__ void resize_image_kernel(const uint8_t* input, uint8_t* output,
                                  int input_width, int input_height,
                                  int output_width, int output_height,
                                  int channels, size_t input_pitch, size_t output_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < output_width && y < output_height) {
        float scale_x = (float)input_width / output_width;
        float scale_y = (float)input_height / output_height;
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        for (int c = 0; c < channels; ++c) {
            float value = bilinear_interpolate(input, input_width, input_height, 
                                             channels, src_x, src_y, c);
            
            size_t output_idx = y * output_pitch + x * channels + c;
            output[output_idx] = (uint8_t)fminf(255.0f, fmaxf(0.0f, value));
        }
    }
}

// Gaussian blur kernel
__global__ void gaussian_blur_kernel(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   size_t pitch, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int kernel_size = (int)(sigma * 3) * 2 + 1;
        int half_kernel = kernel_size / 2;
        
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    // Clamp to image boundaries
                    px = max(0, min(px, width - 1));
                    py = max(0, min(py, height - 1));
                    
                    float weight = expf(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
                    
                    size_t input_idx = py * pitch + px * channels + c;
                    sum += input[input_idx] * weight;
                    weight_sum += weight;
                }
            }
            
            size_t output_idx = y * pitch + x * channels + c;
            output[output_idx] = (uint8_t)(sum / weight_sum);
        }
    }
}

// Sobel edge detection kernel
__global__ void sobel_edge_kernel(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel
        int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        // Sobel Y kernel  
        int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        
        for (int c = 0; c < channels; ++c) {
            float gx = 0.0f, gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    size_t idx = (y + ky) * pitch + (x + kx) * channels + c;
                    float pixel = input[idx];
                    
                    gx += pixel * sobel_x[ky + 1][kx + 1];
                    gy += pixel * sobel_y[ky + 1][kx + 1];
                }
            }
            
            float magnitude = sqrtf(gx * gx + gy * gy);
            size_t output_idx = y * pitch + x * channels + c;
            output[output_idx] = (uint8_t)fminf(255.0f, magnitude);
        }
    }
}

// RGB to YUV conversion kernel
__global__ void rgb_to_yuv_kernel(const uint8_t* input, uint8_t* output,
                                int width, int height, size_t input_pitch, size_t output_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        size_t input_idx = y * input_pitch + x * 3;
        size_t output_idx = y * output_pitch + x * 3;
        
        float r = input[input_idx + 0];
        float g = input[input_idx + 1];
        float b = input[input_idx + 2];
        
        // RGB to YUV conversion matrix
        float Y = 0.299f * r + 0.587f * g + 0.114f * b;
        float U = -0.14713f * r - 0.28886f * g + 0.436f * b + 128.0f;
        float V = 0.615f * r - 0.51499f * g - 0.10001f * b + 128.0f;
        
        output[output_idx + 0] = (uint8_t)fminf(255.0f, fmaxf(0.0f, Y));
        output[output_idx + 1] = (uint8_t)fminf(255.0f, fmaxf(0.0f, U));
        output[output_idx + 2] = (uint8_t)fminf(255.0f, fmaxf(0.0f, V));
    }
}

// YUV to RGB conversion kernel
__global__ void yuv_to_rgb_kernel(const uint8_t* input, uint8_t* output,
                                int width, int height, size_t input_pitch, size_t output_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        size_t input_idx = y * input_pitch + x * 3;
        size_t output_idx = y * output_pitch + x * 3;
        
        float Y = input[input_idx + 0];
        float U = input[input_idx + 1] - 128.0f;
        float V = input[input_idx + 2] - 128.0f;
        
        // YUV to RGB conversion matrix
        float r = Y + 1.13983f * V;
        float g = Y - 0.39465f * U - 0.58060f * V;
        float b = Y + 2.03211f * U;
        
        output[output_idx + 0] = (uint8_t)fminf(255.0f, fmaxf(0.0f, r));
        output[output_idx + 1] = (uint8_t)fminf(255.0f, fmaxf(0.0f, g));
        output[output_idx + 2] = (uint8_t)fminf(255.0f, fmaxf(0.0f, b));
    }
}

// Histogram equalization kernel
__global__ void histogram_equalization_kernel(const uint8_t* input, uint8_t* output,
                                            const int* cdf, int width, int height,
                                            int channels, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = y * pitch + x * channels + c;
            uint8_t pixel_value = input[idx];
            
            // Apply histogram equalization using CDF
            int equalized_value = (cdf[pixel_value] * 255) / (width * height);
            output[idx] = (uint8_t)fminf(255.0f, fmaxf(0.0f, equalized_value));
        }
    }
}

// Brightness and contrast adjustment kernel
__global__ void brightness_contrast_kernel(const uint8_t* input, uint8_t* output,
                                         int width, int height, int channels,
                                         size_t pitch, float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = y * pitch + x * channels + c;
            float pixel = input[idx];
            
            // Apply brightness and contrast
            float adjusted = (pixel - 128.0f) * contrast + 128.0f + brightness;
            output[idx] = (uint8_t)fminf(255.0f, fmaxf(0.0f, adjusted));
        }
    }
}

// Kernel launch functions
namespace kernels {
    
    cudaError_t launch_resize_kernel(const ImageData& input, ImageData& output,
                                   int target_width, int target_height,
                                   cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((target_width + block_size.x - 1) / block_size.x,
                      (target_height + block_size.y - 1) / block_size.y);
        
        resize_image_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height,
            target_width, target_height,
            input.channels, input.pitch, output.pitch);
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_gaussian_blur(const ImageData& input, ImageData& output,
                                   float sigma, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                      (input.height + block_size.y - 1) / block_size.y);
        
        gaussian_blur_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height, input.channels,
            input.pitch, sigma);
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_edge_detection(const ImageData& input, ImageData& output,
                                    cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                      (input.height + block_size.y - 1) / block_size.y);
        
        sobel_edge_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height, input.channels, input.pitch);
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_rgb_to_yuv(const ImageData& input, ImageData& output,
                                cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                      (input.height + block_size.y - 1) / block_size.y);
        
        rgb_to_yuv_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height, input.pitch, output.pitch);
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_yuv_to_rgb(const ImageData& input, ImageData& output,
                                cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                      (input.height + block_size.y - 1) / block_size.y);
        
        yuv_to_rgb_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height, input.pitch, output.pitch);
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_brightness_contrast(const ImageData& input, ImageData& output,
                                         float brightness, float contrast,
                                         cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                      (input.height + block_size.y - 1) / block_size.y);
        
        brightness_contrast_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data, output.data,
            input.width, input.height, input.channels,
            input.pitch, brightness, contrast);
        
        return cudaGetLastError();
    }
}

} // namespace gpu
} // namespace ultra