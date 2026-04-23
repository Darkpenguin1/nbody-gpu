/**
 From https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html
 Check the webpage for description
 */
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cstring>
using namespace std;
using namespace cv;
#include <cuda_runtime.h>
#include <chrono>


short lpf_filter_6[3][3] =
   { {0, 1, 0},
     {1, 2, 1},
     {0, 1, 0}};

short lpf_filter_9[3][3] =
   { {1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}};

short lpf_filter_10[3][3] =
   { {1, 1, 1},
     {1, 2, 1},
     {1, 1, 1}};

short lpf_filter_16[3][3] =
   { {1, 2, 1},
     {2, 4, 2},
     {1, 2, 1}};

short lpf_filter_32[3][3] =
   { {1,  4, 1},
     {4, 12, 4},
     {1,  4, 1}};

short hpf_filter_1[3][3] =
   { { 0, -1,  0},
     {-1,  5, -1},
     { 0, -1,  0}};

short hpf_filter_2[3][3] =
   { {-1, -1, -1},
     {-1,  9, -1},
     {-1, -1, -1}};

short hpf_filter_3[3][3] =
   { { 1, -2,  1},
     {-2,  5, -2},
     { 1, -2,  1}};


    
__global__ void convolve3x3(
    const uchar* input,
    uchar* output,
    int rows,
    int cols,
    int channels,
    const short* filter,
    int divisor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < cols - 1 && y >= 1 && y < rows - 1) {
        for (int c = 0; c < channels; c++) {
            int sum = 0;

            for (int a = -1; a <= 1; a++) {
                for (int b = -1; b <= 1; b++) {
                    int nx = x + b;
                    int ny = y + a;

                    int imgIdx = (ny * cols + nx) * channels + c;
                    int filtIdx = (a + 1) * 3 + (b + 1);

                    sum += input[imgIdx] * filter[filtIdx];
                }
            }

            sum /= divisor;
            sum = (sum < 0) ? 0 : sum;
            sum = (sum > 255) ? 255 : sum;

            int outIdx = (y * cols + x) * channels + c;
            output[outIdx] = (uchar)sum;
        }
    }
}

int main( int argc, char** argv )
{

    String imageName("data/lena.jpg"); // by default
    short filter[3][3];
    int filterType; 
    char low_high;
    int divisor = 1; // assume high pass filter default 

    // Setup filter type 
    cout << "Enter l for low-pass or h for high-pass: ";
    cin >> low_high;

    cout << "Low-pass options: 6, 9, 10, 16, 32\n";
    cout << "High-pass options: 1, 2, 3\n";

    cout << "Enter filter type: ";
    cin >> filterType;
    
    
    if (low_high == 'l' || low_high == 'L') {
        
        switch (filterType)
        {
            case 6:
                /* code */
                divisor = 6;
                memcpy(filter, lpf_filter_6, sizeof(filter));
                break;
            
            case 9:
                divisor = 9; 
                memcpy(filter, lpf_filter_9, sizeof(filter));
                break;

            case 10:
                divisor = 10; 
                memcpy(filter, lpf_filter_10, sizeof(filter));
                break;

            case 16: 
                divisor = 16; 
                memcpy(filter, lpf_filter_16, sizeof(filter));
                break;
            case 32: 
                divisor = 32; 
                memcpy(filter, lpf_filter_32, sizeof(filter));
                break;
            

            default:
                cout << "Invalid low-pass filter type\n";
                return 1;
                break;
        }
    }

    else if (low_high == 'h' || low_high == 'H') {
        switch (filterType)
        {
          
            case 1: 
                
                memcpy(filter, hpf_filter_1, sizeof(filter));
                break;

            case 2: 
                memcpy(filter, hpf_filter_2, sizeof(filter));
                break;
            case 3: 
                memcpy(filter, hpf_filter_3, sizeof(filter));
                break;
            
            default:
                cout << "Invalid high-pass filter type\n";
                return 1;
                break;
        }
    }

    else {
        cout << "Something broke\n";
        return 1; // error etc 
    }



    if (argc > 1)
    {
        imageName = argv[1];
    }
    Mat image = imread( imageName );
    if (image.empty()) {
        cout << "Could not open or find the image\n";
        return 1;
    }


    Mat new_image = Mat::zeros( image.size(), image.type() );
    // for open cv mat image how to copy image -> gpu 
    size_t imgBytes = image.rows * image.cols * image.channels() * sizeof(uchar);
    size_t filterBytes = 3 * 3 * sizeof(short);
    if (!image.isContinuous()) {
        image = image.clone();
    }
    if (!new_image.isContinuous()) {
        new_image = new_image.clone();
    }

    auto start = std::chrono::high_resolution_clock::now();
    uchar* d_input = nullptr;
    uchar* d_output = nullptr;
    short* d_filter = nullptr;
    

    cudaMalloc((void**)&d_input, imgBytes);
    cudaMalloc((void**)&d_output, imgBytes);
    cudaMalloc((void**)&d_filter, filterBytes);

    cudaMemcpy(d_input, image.data, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, imgBytes);

    dim3 block(16, 16);
    dim3 grid((image.cols + block.x - 1) / block.x,
            (image.rows + block.y - 1) / block.y);

    convolve3x3<<<grid, block>>>(
        d_input,
        d_output,
        image.rows,
        image.cols,
        image.channels(),
        d_filter,
        divisor
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(new_image.data, d_output, imgBytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

   
    imwrite("filtered_output.jpg", new_image);
    cout << "Saved filtered_output.jpg" << endl;

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    cout << "GPU Total Exec Time: " << elapsedMs << " ms\n";

    return 0;
}
