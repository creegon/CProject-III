#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <falcon/convolution/winograd_convolution.h>
#include "NetWork.h"
#include "../fftw/fftw3.h"

// 初始化卷积层结构体
void initialize_conv_layer(ConvLayer* conv_layer, int input_channels, int output_channels, int kernel_size, int stride, PaddingType padding_type, ActivationType activation) {
    Layer base;
    base.input_channels = input_channels;
    base.output_channels = output_channels;
    
    conv_layer->base = base;
    conv_layer->kernel_size = kernel_size;
    conv_layer->stride = stride;
    conv_layer->padding_type = padding_type;
    conv_layer->activation = activation;

    conv_layer->kernel = (float**)malloc(output_channels * sizeof(float*));
    

    #pragma omp parallel
    for (int i = 0; i < output_channels; i++) {
        conv_layer->kernel[i] = (float*)malloc(input_channels * kernel_size * kernel_size * sizeof(float));
    }

    conv_layer->biases = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bias_grad_accum = (float*)malloc(output_channels * sizeof(float));   
    
    //初始化padding的值（要填充多少）
    if (padding_type == SAME) {
        conv_layer->padding = (kernel_size - 1) / 2;
    }
    else if (padding_type == VALID) {
        conv_layer->padding = 0;
    }

    // 计算 He initialization 的标准差
    float stddev = sqrtf(2.0f / (input_channels * kernel_size * kernel_size));

    // #pragma omp parallel
    for (int i = 0; i < output_channels; i++) {
        for (int j = 0; j < input_channels * kernel_size * kernel_size; j++) {
            // 使用标准差生成一个符合正态分布的随机数
            float rand_num = stddev * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
            conv_layer->kernel[i][j] = rand_num;
        }
        conv_layer->biases[i] = 0.0f; // 初始化为0
        conv_layer->bias_grad_accum[i] = 0.0f;
    }

    // 初始化批量归一化参数
    conv_layer->bn_gamma = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_beta = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_gamma_accum = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_beta_accum = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_gamma_m = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_beta_m = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_gamma_v = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_beta_v = (float*)malloc(output_channels * sizeof(float));

    conv_layer->bn_mean = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_variance = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_mean_accum = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_variance_accum = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_mean_m = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_variance_m = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_mean_v = (float*)malloc(output_channels * sizeof(float));
    conv_layer->bn_variance_v = (float*)malloc(output_channels * sizeof(float));

    // #pragma omp parallel
    for (int i = 0; i < output_channels; i++) {
        conv_layer->bn_gamma[i] = 1.0f;
        conv_layer->bn_beta[i] = 0.0f;
        conv_layer->bn_gamma_accum[i] = 0.0f;
        conv_layer->bn_beta_accum[i] = 0.0f;
        conv_layer->bn_gamma_m[i] = 0.0f;
        conv_layer->bn_beta_m[i] = 0.0f;
        conv_layer->bn_gamma_v[i] = 0.0f;
        conv_layer->bn_beta_v[i] = 0.0f;

        conv_layer->bn_mean[i] = 0.0f;
        conv_layer->bn_variance[i] = 1.0f;
        conv_layer->bn_mean_accum[i] = 0.9f;
        conv_layer->bn_variance_accum[i] = 0.9f;
        conv_layer->bn_mean_m[i] = 0.0f;
        conv_layer->bn_variance_m[i] = 0.0f;
        conv_layer->bn_mean_v[i] = 0.0f;
        conv_layer->bn_variance_v[i] = 0.0f;
    }

    //debug专用
    // printf("初始化卷积层成功!\n");
}


void conv_forward(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map) {
    int output_size = (input_size - conv_layer->kernel_size + 2 * conv_layer->padding) / conv_layer->stride + 1; //在本次project中，output_size和input_size相等

    //先将input_data进行0填充
    padding_matrix(input_data, input_data, input_size, input_size + 2 * conv_layer->padding);
    
    //然后进行卷积运算
    float* input = input_data;
    // 批量归一化参数，例如可学习的gamma和beta，它们的长度应该等于输出通道数
    float* gamma = conv_layer->bn_gamma;
    float* beta = conv_layer->bn_beta;


    // 批量归一化中的均值和方差，它们的长度也应该等于输出通道数
    float* mean = conv_layer->bn_mean;
    float* variance = conv_layer->bn_variance;

    // 批量归一化的超参数，用于平滑计算，通常设置为1e-5
    float eps = 1e-5;

    
    for (int oc = 0; oc < conv_layer->base.output_channels; oc++) {
        
        for (int oh = 0; oh < output_size; oh++) {
            for (int ow = 0; ow < output_size; ow++) {
                float value = 0.0f;

                for (int ic = 0; ic < conv_layer->base.input_channels; ic++) {
                    for (int kh = 0; kh < conv_layer->kernel_size; kh++) {
                        #pragma omp parallel
                        for (int kw = 0; kw < conv_layer->kernel_size; kw++) {
                            int ih = oh * conv_layer->stride - conv_layer->padding + kh;
                            int iw = ow * conv_layer->stride - conv_layer->padding + kw;
                            if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                                value += input[ic * input_size * input_size + ih * input_size + iw] *
                                         conv_layer->kernel[oc][ic * conv_layer->kernel_size * conv_layer->kernel_size + kh * conv_layer->kernel_size + kw];
                               
                            }
                        }
                    }
                }

                value += conv_layer->biases[oc];

                // 应用批量归一化操作
                value = (value - mean[oc]) / sqrtf(variance[oc] + eps);
                value = gamma[oc] * value + beta[oc];


                // 在这里应用激活函数
                value = apply_activation(conv_layer->activation, value);

                // 将输出值存储到输出特征图中
                output_feature_map[oc * output_size * output_size + oh * output_size + ow] = value;
            }
        }

    }
}

void conv_forward_combined(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map) {
    if (conv_layer->kernel_size > 3) {
        conv_forward_fft(conv_layer, input_data, input_size, output_feature_map);
    } else {
        conv_forward_winograd(conv_layer, input_data, input_size, output_feature_map);
    }
}



void conv_forward_winograd(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map) {
    int output_size = (input_size - conv_layer->kernel_size + 2 * conv_layer->padding) / conv_layer->stride + 1;
    float* input_padded = calloc(input_size * input_size, sizeof(float));

     // 批量归一化参数，例如可学习的gamma和beta，它们的长度应该等于输出通道数
    float* gamma = conv_layer->bn_gamma;
    float* beta = conv_layer->bn_beta;


    // 批量归一化中的均值和方差，它们的长度也应该等于输出通道数
    float* mean = conv_layer->bn_mean;
    float* variance = conv_layer->bn_variance;

    // 批量归一化的超参数，用于平滑计算，通常设置为1e-5
    float eps = 1e-5;

    padding_matrix(input_data, input_padded, input_size, input_size + 2 * conv_layer->padding);

    for (int oc = 0; oc < conv_layer->base.output_channels; oc++) {
        for (int ic = 0; ic < conv_layer->base.input_channels; ic++) {
            falcon::WinogradConvolution<float> winograd_conv(input_size, input_size, conv_layer->kernel_size, conv_layer->kernel_size);
            winograd_conv.set_input(input_padded);
            winograd_conv.set_kernel(conv_layer->kernel[oc]);

            // Create a temporary output buffer for the Winograd convolution result
            float* temp_output = new float[output_size * output_size];
            winograd_conv.set_output(temp_output);
            winograd_conv.execute();

            // Add the Winograd convolution result to the output_feature_map
            #pragma omp parallel for collapse(2)
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    float value;
                    value += temp_output[oh * output_size + ow];
                    value += conv_layer->biases[oc];

                    // 应用批量归一化操作
                    value = (value - mean[oc]) / sqrtf(variance[oc] + eps);
                    value = gamma[oc] * value + beta[oc];


                    // 在这里应用激活函数
                    value = apply_activation(conv_layer->activation, value);

                    // 将输出值存储到输出特征图中
                    output_feature_map[oc * output_size * output_size + oh * output_size + ow] = value;
                }
            }

            // Free the temporary output buffer
            delete[] temp_output;
        }
    }

    free(input_padded);
}




void conv_forward_fft(ConvLayer* conv_layer, float* input_data, int input_size, float* output_feature_map) {
    int output_size = (input_size - conv_layer->kernel_size + 2 * conv_layer->padding) / conv_layer->stride + 1;
    int new_input_size = input_size + 2 * conv_layer->padding;

    float* input_padded = calloc(input_size * input_size, sizeof(float));
    float* kernel_padded = calloc(input_size * input_size, sizeof(float));
    double* output_padded = calloc(input_size * input_size, sizeof(float));

    fftw_complex* input_fft = (fftw_complex*)fftw_malloc(input_size * input_size * sizeof(fftw_complex));
    fftw_complex* kernel_fft = (fftw_complex*)fftw_malloc(input_size * input_size * sizeof(fftw_complex));
    fftw_complex* output_fft = (fftw_complex*)fftw_malloc(input_size * input_size * sizeof(fftw_complex));

    padding_matrix(input_data, input_padded, input_size, new_input_size);

    //将input_padded转为double类型
    double* input_padded_double = (double*)malloc(input_size * input_size * sizeof(double));
    for (int i = 0; i < input_size * input_size; i++) {
        input_padded_double[i] = (double)input_padded[i];
    }

    // 批量归一化参数，例如可学习的gamma和beta，它们的长度应该等于输出通道数
    float* gamma = conv_layer->bn_gamma;
    float* beta = conv_layer->bn_beta;

    // 批量归一化中的均值和方差，它们的长度也应该等于输出通道数
    float* mean = conv_layer->bn_mean;
    float* variance = conv_layer->bn_variance;

    // 批量归一化的超参数，用于平滑计算，通常设置为1e-5
    float eps = 1e-5;

    // #pragma omp parallel for collapse(2)
    for (int oc = 0; oc < conv_layer->base.output_channels; oc++) {
        for (int ic = 0; ic < conv_layer->base.input_channels; ic++) {
            // Pad the kernel and compute its FFT
            pad_kernel(conv_layer->kernel[oc], kernel_padded, conv_layer->kernel_size, new_input_size);

            //将kernel_padded转化为double类型
            double* kernel_padded_double = calloc(new_input_size * new_input_size, sizeof(double));
            for (int i = 0; i < new_input_size * new_input_size; i++) {
                kernel_padded_double[i] = kernel_padded[i];
            }

            fftw_plan kernel_plan = fftw_plan_dft_r2c_2d(new_input_size, new_input_size, kernel_padded_double, kernel_fft, FFTW_ESTIMATE);
            fftw_execute(kernel_plan);

            
            // Compute the FFT of the input
            fftw_plan input_plan = fftw_plan_dft_r2c_2d(new_input_size, new_input_size, input_padded_double, input_fft, FFTW_ESTIMATE);
            fftw_execute(input_plan);

            // Multiply in frequency domain
            #pragma omp parallel
            for (int i = 0; i < new_input_size * new_input_size; i++) {
                output_fft[i][0] = input_fft[i][0] * kernel_fft[i][0] - input_fft[i][1] * kernel_fft[i][1];
                output_fft[i][1] = input_fft[i][1] * kernel_fft[i][1] + input_fft[i][1] * kernel_fft[i][0];
            }

            // Compute the inverse FFT of the result
            fftw_plan output_plan = fftw_plan_dft_c2r_2d(new_input_size, new_input_size, output_fft, output_padded, FFTW_ESTIMATE);
                        fftw_execute(output_plan);

            // Extract the output feature map values
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    float value = output_padded[oh * input_size + ow];

                    // Add bias
                    value += conv_layer->biases[oc];

                    // Apply batch normalization
                    value = (value - mean[oc]) / sqrtf(variance[oc] + eps);
                    value = gamma[oc] * value + beta[oc];

                    // Apply activation function
                    value = apply_activation(conv_layer->activation, value);

                    // Store the output value in the output feature map
                    output_feature_map[conv_layer->base.output_channels * output_size * output_size + oc * output_size * output_size + oh * output_size + ow] = value;
                }
            }
        }
    }

    free(input_padded);
    free(kernel_padded);
    free(output_padded);

    fftw_free(input_fft);
    fftw_free(kernel_fft);
    fftw_free(output_fft);
}

void padding_matrix(float *src, float *dest, int original_size, int padded_size) {
    int i, j;

    // Fill the corners with 0
    dest[0] = 0.0f;
    dest[padded_size - 1] = 0.0f;
    dest[padded_size * (padded_size - 1)] = 0.0f;
    dest[padded_size * padded_size - 1] = 0.0f;

    // Fill the top and bottom rows with 0
    for (i = 1; i < padded_size - 1; i++) {
        dest[i] = 0.0f;
        dest[padded_size * (padded_size - 1) + i] = 0.0f;
    }

    // Fill the left and right columns with 0, and copy the original matrix to the center
    for (i = 0; i < original_size; i++) {
        dest[(i + 1) * padded_size] = 0.0f;
        dest[(i + 1) * padded_size + padded_size - 1] = 0.0f;

        for (j = 0; j < original_size; j++) {
            dest[(i + 1) * padded_size + j + 1] = src[i * original_size + j];
        }
    }
}

float apply_activation(ActivationType activation, float value) {
    switch (activation) {
        case RELU:
            return value > 0.0f ? value : 0.0f;
        case SIGMOID:
            return 1.0f / (1.0f + expf(-value));
        case TANH:
            return tanhf(value);
        default:
            return value;
    }
}
                            

void conv_accumulate_gradients(ConvLayer* conv_layer, float* input, float* output, float* input_grad, float* output_grad, float* kernel_grad_accum, int size) {
            int input_channels = conv_layer->base.input_channels;
            int output_channels = conv_layer->base.output_channels;
            int kernel_size = conv_layer->kernel_size;
            int stride = conv_layer->stride;
            float** kernels = conv_layer->kernel;
            float eps = 1e-8;
            int count = 0;

            // 累积梯度
            #pragma omp atomic
            for (int oc = 0; oc < output_channels; oc++) {
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            float grad_sum = 0.0f;
                            for (int oh = 0; oh < size; oh++) {
                                for (int ow = 0; ow < size; ow++) {
                                    int ih = oh * stride + kh;
                                    int iw = ow * stride + kw;

                                    grad_sum += input[ic * size * size + ih * size + iw] * output_grad[oc * size * size + oh * size + ow];
                                }
                            }   
                            int idx = oc * input_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            kernel_grad_accum[idx] += grad_sum;
                        }
                    }
                }
            }
        

            // 计算输入梯度
            #pragma omp atomic
            for (int ic = 0; ic < input_channels; ic++) {
                for (int ih = 0; ih < size; ih++) {
                    for (int iw = 0; iw < size; iw++) {
                        float grad_sum = 0.0f;
                        for (int oc = 0; oc < output_channels; oc++) {
                            for (int kh = 0; kh < kernel_size; kh++) {
                                for (int kw = 0; kw < kernel_size; kw++) {
                                    int padded_ih = ih + conv_layer->padding - kh;
                                    int padded_iw = iw + conv_layer->padding - kw;
                                    int oh = padded_ih / stride;
                                    int ow = padded_iw / stride;
                                    if (oh >= 0 && oh < size && ow >= 0 && ow < size) {
                                        int idx = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                        grad_sum += kernels[oc][idx] * output_grad[oc * size * size + oh * size + ow];
                                    }
                                }
                            }
                        }
                        input_grad[ic * size * size + ih * size + iw] = grad_sum;
                    }
                }
            }

            //debug
            // printf("计算输入梯度完成！\n");

            // 计算批量归一化参数梯度累积
            #pragma omp atomic
            for (int oc = 0; oc < output_channels; oc++) {
                float gamma_grad_sum = 0.0f;
                float beta_grad_sum = 0.0f;
                for (int oh = 0; oh < size; oh++) {
                    for (int ow = 0; ow < size; ow++) {
                        int idx = oc * size * size + oh * size + ow;
                        float normalized_output = (output[idx] - conv_layer->bn_mean[oc]) / sqrtf(conv_layer->bn_variance[oc] + eps);
                        gamma_grad_sum += output_grad[idx] * normalized_output;
                        beta_grad_sum += output_grad[idx];
                    }
                }
                conv_layer->bn_gamma_accum[oc] += gamma_grad_sum;
                conv_layer->bn_beta_accum[oc] += beta_grad_sum;
            }

            //debug
            // printf("计算批量归一化参数梯度累积完成！\n");

            // 计算均值和方差的梯度累积
            // #pragma omp parallel
            for (int oc = 0; oc < output_channels; oc++) {
                float mean_grad_sum = 0.0f;
                float variance_grad_sum = 0.0f;
                for (int oh = 0; oh < size; oh++) {
                    for (int ow = 0; ow < size; ow++) {
                        int idx = oc * size * size + oh * size + ow;
                        float normalized_input = (output[idx] - conv_layer->bn_mean[oc]) / sqrtf(conv_layer->bn_variance[oc] + eps);
                        mean_grad_sum -= output_grad[idx] * conv_layer->bn_gamma[oc] * normalized_input;
                        variance_grad_sum -= output_grad[idx] * conv_layer->bn_gamma[oc] * normalized_input * (output[idx] - conv_layer->bn_mean[oc]) / (2.0f * conv_layer->bn_variance[oc] * sqrtf(conv_layer->bn_variance[oc] + eps));
                    }
                }
                conv_layer->bn_mean_accum[oc] += mean_grad_sum;
                conv_layer->bn_variance_accum[oc] += variance_grad_sum;
            }

            //debug
            // printf("计算均值和方差的梯度累积完成！\n");

            // 计算偏置梯度累积
            // 
            for (int oc = 0; oc < output_channels; oc++) {
                float bias_grad_sum = 0.0f;
                #pragma omp parallel for collapse(2)
                for (int oh = 0; oh < size; oh++) {
                    for (int ow = 0; ow < size; ow++) {
                        bias_grad_sum += output_grad[oc * size * size + oh * size + ow];
                    }
                }
            conv_layer->bias_grad_accum[oc] += bias_grad_sum;
            }
        
        //debug
        // printf("卷积层反向传播成功！\n");
    }

void update_conv_layer(ConvLayer* conv_layer, float* kernel_grad_accum, int batch_size, float* m, float* v, float learning_rate, int timestamp) {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-6;
    int input_channels = conv_layer->base.input_channels;
    int output_channels = conv_layer->base.output_channels;
    int kernel_size = conv_layer->kernel_size;
    float** kernels = conv_layer->kernel;
    float* bn_gamma_accum = conv_layer->bn_gamma_accum;
    float* bn_beta_accum = conv_layer->bn_beta_accum;
    float* bn_mean_accum = conv_layer->bn_mean_accum;
    float* bn_variance_accum = conv_layer->bn_variance_accum;

    // 更新卷积核
    // #pragma omp parallel
    for (int oc = 0; oc < output_channels; oc++) {
        for (int ic = 0; ic < input_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int idx = oc * input_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    int idx2 = ic * kernel_size * kernel_size + kh * kernel_size + kw;

                    float grad_avg = kernel_grad_accum[idx] / (float) batch_size;

                    // Update biased first moment estimate
                    m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad_avg;

                    // Update biased second raw moment estimate
                    v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad_avg * grad_avg;

                    // Compute bias-corrected first moment estimate
                    float m_hat = m[idx] / (1.0f - powf(beta1, timestamp));

                    // Compute bias-corrected second raw moment estimate
                    float v_hat = v[idx] / (1.0f - powf(beta2, timestamp));

                    // Update kernels
                    kernels[oc][idx2] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
                    // //试着打印一些kernels的值，看看有没有变化
                    // if(oc == 0 && idx2 < 10){
                    //     printf("kernels[%d][%d] = %f\n", oc, idx2, kernels[oc][idx2]);
                    // }

                    // Reset the accumulated gradient for the next batch
                    kernel_grad_accum[idx] = 0.0f;
                }
            }
        }

       // 更新bn_gamma、bn_beta、bn_mean和bn_variance
        float grad_avg_gamma = conv_layer->bn_gamma_accum[oc] / (float) batch_size;
        float grad_avg_beta = conv_layer->bn_beta_accum[oc] / (float) batch_size;
        float grad_avg_mean = conv_layer->bn_mean_accum[oc] / (float) batch_size;
        float grad_avg_variance = conv_layer->bn_variance_accum[oc] / (float) batch_size;

        // Update biased first moment estimates
        conv_layer->bn_gamma_m[oc] = beta1 * conv_layer->bn_gamma_m[oc] + (1.0f - beta1) * grad_avg_gamma;
        conv_layer->bn_beta_m[oc] = beta1 * conv_layer->bn_beta_m[oc] + (1.0f - beta1) * grad_avg_beta;
        conv_layer->bn_mean_m[oc] = beta1 * conv_layer->bn_mean_m[oc] + (1.0f - beta1) * grad_avg_mean;
        conv_layer->bn_variance_m[oc] = beta1 * conv_layer->bn_variance_m[oc] + (1.0f - beta1) * grad_avg_variance;

        // Update biased second raw moment estimates
        conv_layer->bn_gamma_v[oc] = beta2 * conv_layer->bn_gamma_v[oc] + (1.0f - beta2) * grad_avg_gamma * grad_avg_gamma;
        conv_layer->bn_beta_v[oc] = beta2 * conv_layer->bn_beta_v[oc] + (1.0f - beta2) * grad_avg_beta * grad_avg_beta;
        conv_layer->bn_mean_v[oc] = beta2 * conv_layer->bn_mean_v[oc] + (1.0f - beta2) * grad_avg_mean * grad_avg_mean;
        conv_layer->bn_variance_v[oc] = beta2 * conv_layer->bn_variance_v[oc] + (1.0f - beta2) * grad_avg_variance * grad_avg_variance;

        // Compute bias-corrected first moment estimates
        float m_hat_gamma = conv_layer->bn_gamma_m[oc] / (1.0f - powf(beta1, timestamp));
        float m_hat_beta = conv_layer->bn_beta_m[oc] / (1.0f - powf(beta1, timestamp));
        float m_hat_mean = conv_layer->bn_mean_m[oc] / (1.0f - powf(beta1, timestamp));
        float m_hat_variance = conv_layer->bn_variance_m[oc] / (1.0f - powf(beta1, timestamp));

        // Compute bias-corrected second raw moment estimates
        float v_hat_gamma = conv_layer->bn_gamma_v[oc] / (1.0f - powf(beta2, timestamp));
        float v_hat_beta = conv_layer->bn_beta_v[oc] / (1.0f - powf(beta2, timestamp));
        float v_hat_mean = conv_layer->bn_mean_v[oc] / (1.0f - powf(beta2, timestamp));
        float v_hat_variance = conv_layer->bn_variance_v[oc] / (1.0f - powf(beta2, timestamp));

        // Update bn_gamma, bn_beta, bn_mean, and bn_variance
        conv_layer->bn_gamma[oc] -= learning_rate * m_hat_gamma / (sqrtf(v_hat_gamma) + epsilon);
        conv_layer->bn_beta[oc] -= learning_rate * m_hat_beta / (sqrtf(v_hat_beta) + epsilon);
        conv_layer->bn_mean[oc] -= learning_rate * m_hat_mean / (sqrtf(v_hat_mean) + epsilon);
        conv_layer->bn_variance[oc] -= learning_rate * m_hat_variance / (sqrtf(v_hat_variance) + epsilon);

        // Reset the accumulated gradients for the next batch
        bn_gamma_accum[oc] = 0.0f;
        bn_beta_accum[oc] = 0.0f;
        bn_mean_accum[oc] = 0.0f;
        bn_variance_accum[oc] = 0.0f;;

        // 重置累积梯度
        conv_layer->bn_mean_accum[oc] = 0.0f;
        conv_layer->bn_variance_accum[oc] = 0.0f;


        // 更新偏置
        float bias_grad_avg = conv_layer->bias_grad_accum[oc] / (float)batch_size;
        conv_layer->biases[oc] -= learning_rate * bias_grad_avg;

        // 重置累积偏置梯度
        conv_layer->bias_grad_accum[oc] = 0.0f;
    }
}

void pad_kernel(float* kernel, float* padded_kernel, int kernel_size, int padded_size) {
    int padding = (padded_size - kernel_size) / 2;

    #pragma omp parallel atomic
    for (int row = 0; row < padded_size; row++) {
        for (int col = 0; col < padded_size; col++) {
            if (row >= padding && row < (padding + kernel_size) &&
                col >= padding && col < (padding + kernel_size)) {
                // 将原始卷积核的值复制到扩充矩阵的相应位置
                int kernel_row = row - padding;
                int kernel_col = col - padding;
                padded_kernel[row * padded_size + col] = kernel[kernel_row * kernel_size + kernel_col];
            } else {
                // 填充0
                padded_kernel[row * padded_size + col] = 0.0f;
            }
        }
    }
}
