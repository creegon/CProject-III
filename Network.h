#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>
#include <math.h>
//数据集的数据
#define IMG_SIZE 28
#define NUM_TRAIN_IMAGES 10000
#define NUM_TEST_IMAGES 10000

#define DATASET_INPUT_CHANNELS 1

typedef enum {
    RELU,
    SIGMOID,
    TANH,
} ActivationType; // 其他激活函数类型

typedef enum {
    SAME, 
    VALID, 
} PaddingType; //填充枚举类

typedef struct {
    int input_channels;
    int output_channels;
} Layer;

typedef struct {
    Layer base;
    int kernel_size; // 默认为正方形
    int stride; // 步长
    int padding; // 填充
    float** kernel; //一个二维float型指针数组，用于存储卷积层中每个卷积核的权重。kernel[i][j]表示卷积层中第i个卷积核的第j个权重值。
    float* biases; // 一个float型指针，用于存储卷积层中每个卷积核的偏置项。biases[i]表示卷积层中第i个卷积核的偏置项。
    float* bias_grad_accum; // 累积
    PaddingType padding_type; // 填充类型
    ActivationType activation; // 激活函数类型

    // 批量归一化参数
    float* bn_gamma; // 可学习的scale参数，长度等于输出通道数
    float* bn_beta;  // 可学习的shift参数，长度等于输出通道数
    float* bn_gamma_accum; // 累积
    float* bn_beta_accum; // 累积
    float* bn_gamma_m;
    float* bn_gamma_v;
    float* bn_beta_m;
    float* bn_beta_v;

    float* bn_mean;  // 均值，长度等于输出通道数
    float* bn_variance; // 方差，长度等于输出通道数
    float* bn_mean_accum;
    float* bn_variance_accum;
    float* bn_mean_m;
    float* bn_mean_v;
    float* bn_variance_m;
    float* bn_variance_v;

} ConvLayer; // 卷积层

typedef struct {
    Layer base;
    int pool_size;      // 池化区域的大小，假设为正方形
    int stride;         // 池化的步长
} MaxPoolLayer;

typedef struct {
    Layer base;
    int num_classes; // 分类数(10)
    float* weights; // 一个float型指针，用于存储全连接层中每个神经元的权重。weights[i]表示全连接层中第i个神经元的权重。
    float* biases; // 一个float型指针，用于存储全连接层中每个神经元的偏置项。biases[i]表示全连接层中第i个神经元的偏置项。
    float* m;
    float* v;
} FullyConnectedLayer; // 全连接层

typedef struct {
    uint8_t  label;
    float  image[IMG_SIZE * IMG_SIZE];
} ImageData; // 用于存储图像数据

typedef struct {
    ImageData* data;
    int size;
} Batch; // 用于存储一批图像数据




//正向传播部分
int read_fashion_mnist_data(const char* image_file, const char* label_file, ImageData* dataset, int num_images);
void shuffle_dataset(ImageData* dataset, int num_images);
Batch* create_batches(ImageData* dataset, int num_images, int batch_size, int num_batches);
void initialize_conv_layer(ConvLayer* conv_layer, int input_channels, int output_channels, int kernel_size, int stride, PaddingType padding_type, ActivationType activation);
void conv_forward(ConvLayer* conv_layer, float* input_data, int image_size, float* output_feature_map);
void conv_forward_combined(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map);
void conv_forward_winograd(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map);
void conv_forward_fft(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map);
void pad_kernel(float* kernel, float* padded_kernel, int kernel_size, int padded_size);
float apply_activation(ActivationType activation, float value);
void padding_matrix(float *src, float *dest, int original_size, int padded_size);
void initialize_pool_layer(MaxPoolLayer* max_pool_layer, int input_channels, int output_channels, int pool_size, int stride);
void max_pool_forward(MaxPoolLayer* max_pool_layer, float* input, int input_size, float* output);
void initialize_fully_connected_layer(FullyConnectedLayer* fc_layer, int input_channels, int input_size, int output_size);
void fully_connected_forward(FullyConnectedLayer* fully_connected_layer, float* input, int input_size, float* output);
void softmax(float* input, int input_size, float* output);


//反向传播部分
void softmax(float* input, int input_size, float* output);
float cross_entropy_loss(float* output, int output_size, int label);
void fc_accumulate_gradients(FullyConnectedLayer* fc_layer, int input_size, int output_size, float* output_grad, float* input_grad, float* input_grad_accum);
void update_fc_layer(FullyConnectedLayer* fc_layer, int batch_size, int input_size, float* weight_grad_accum, float* m, float* v, float learning_rate, int timestep);
void max_pool_backward(MaxPoolLayer* max_pool_layer, ConvLayer* conv_layer, float* input, float* input_grad, float* pool_output_grad);
void conv_accumulate_gradients(ConvLayer* conv_layer, float* input, float* output, float* input_grad, float* output_grad, float* kernel_grad_accum, int size);
void update_conv_layer(ConvLayer* conv_layer, float* kernel_grad_accum, int batch_size, float* m, float* v, float learning_rate, int timestamp);


#endif /* NETWORK_H */