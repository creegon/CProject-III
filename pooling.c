#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include "NetWork.h"


void initialize_pool_layer(MaxPoolLayer* max_pool_layer, int input_channels, int output_channels, int pool_size, int stride) {
    Layer base;
    base.input_channels = input_channels;
    base.output_channels = output_channels;
    max_pool_layer->base = base;

    max_pool_layer->pool_size = pool_size;
    max_pool_layer->stride = stride;

    //debug
    // printf("初始化池化层成功！\n");
    // printf("\n");
}


//input_size：卷积层的特征图的尺寸
void max_pool_forward(MaxPoolLayer* max_pool_layer, float* input, int input_size, float* output) {
    int output_size = (input_size - max_pool_layer->pool_size) / max_pool_layer->stride + 1;
    // #pragma omp parallel for collapse(3)
    for (int c = 0; c < max_pool_layer->base.input_channels; c++) {
        for (int oh = 0; oh < output_size; oh++) {
            for (int ow = 0; ow < output_size; ow++) {
                float max_value = -FLT_MAX;

                for (int ph = 0; ph < max_pool_layer->pool_size; ph++) {
                    for (int pw = 0; pw < max_pool_layer->pool_size; pw++) {
                        int ih = oh * max_pool_layer->stride + ph;
                        int iw = ow * max_pool_layer->stride + pw;
                        if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                            float current_value = input[max_pool_layer->base.input_channels * input_size * input_size + c * input_size * input_size + ih * input_size + iw];
                            max_value = fmaxf(max_value, current_value);
                        }
                    }
                }
                output[max_pool_layer->base.input_channels * output_size * output_size + c * output_size * output_size + oh * output_size + ow] = max_value;
            }
        }
    }

    //debug
    // printf("池化层前向传播成功！\n");
    // printf("\n");
}

void max_pool_backward(MaxPoolLayer* max_pool_layer, ConvLayer* conv_layer, float* input, float* input_grad, float* pool_output_grad) {
    int input_channels = max_pool_layer->base.input_channels;
    int input_size = conv_layer->kernel_size;
    int pool_size = max_pool_layer->pool_size;
    int stride = max_pool_layer->stride;
    int pooled_size = (input_size - pool_size) / stride + 1;

    // #pragma omp parallel for collapse(3)
    for (int c = 0; c < input_channels; c++) {
        for (int h = 0; h < pooled_size; h++) {
            for (int w = 0; w < pooled_size; w++) {
                // Find the maximum value in the input region
                float max_val = -FLT_MAX;
                int max_h = -1;
                int max_w = -1;
                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        int h_in = h * stride + i;
                        int w_in = w * stride + j;
                        float current_val = input[c * input_size * input_size + h_in * input_size + w_in]; 
                        if (current_val > max_val) {
                            max_val = current_val;
                            max_h = h_in;
                            max_w = w_in;
                        }
                    }
                }
                // Pass the gradient to the location of the maximum value
                input_grad[c * input_size * input_size + max_h * input_size + max_w] = pool_output_grad[c * pooled_size * pooled_size + h * pooled_size + w];
            }
        }
    }
    //debug
    // printf("池化层反向传播成功！\n");
}



