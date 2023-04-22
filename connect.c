#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include "NetWork.h"


void initialize_fully_connected_layer(FullyConnectedLayer* fc_layer, int input_channels, int input_size, int output_size) {
    Layer base;
    base.input_channels = input_channels;
    base.output_channels = output_size; //10
    fc_layer->base = base;

    int num_weights = input_channels * input_size * input_size * output_size; // 512 * 2 * 2 * 10 = 20480
    fc_layer->weights = (float*)malloc(num_weights * sizeof(float));
    fc_layer->biases = (float*)malloc(output_size * sizeof(float));

    // Initialize weights and biases with random values.
    srand(time(NULL));
    // #pragma omp parallel
    for (int i = 0; i < num_weights; i++) {
        fc_layer->weights[i] = ((float)rand() / (float)(RAND_MAX)) - 0.5f;
    }
    // #pragma omp parallel
    for (int i = 0; i < output_size; i++) {
        fc_layer->biases[i] = ((float)rand() / (float)(RAND_MAX)) - 0.5f;
    }

    //debug
    // printf("初始化全连接层成功！\n");
    // printf("\n");
}

void fully_connected_forward(FullyConnectedLayer* fc_layer, float* input, int input_size, float* output) {

    // #pragma omp parallel
    for (int o = 0; o < fc_layer->base.output_channels; o++) {
        output[o] = 0.0f;

        for (int i = 0; i < input_size; i++) {
            output[o] += input[i] * fc_layer->weights[o * input_size + i];
        }

        output[o] += fc_layer->biases[o];
    }
    //debug
    // printf("全连接层前向传播成功！\n");
    // printf("\n");
}

void softmax(float* input, int input_size, float* output) {
    float sum = 0.0;

    for (int i = 0; i < input_size; i++) {
        output[i] = expf(input[i]);
        sum += output[i];
    }

    for (int i = 0; i < input_size; i++) {
        output[i] /= sum;
    }
}

//交叉熵损失函数
float cross_entropy_loss(float* output, int output_size, int label) {
    float loss = 0.0f;
    for (int i = 0; i < output_size; i++) {
        if (i == label) {
            loss -= logf(output[i]  + 1e-8);
        }
    }
    //debug
    // printf("交叉熵损失函数成功！\n");
    return loss;
}

//weights: 代表权重的指针
void fc_accumulate_gradients(FullyConnectedLayer* fc_layer, int input_size, int output_size, float* output_grad, float* input_grad, float* weight_grad_accum) {
    float* weights = fc_layer->weights;
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            input_grad[i] += output_grad[j] * weights[j * input_size + i];
            weight_grad_accum[j * input_size + i] += output_grad[j] * input_grad[i];
        }
    }

    //debug
    // printf("fc_accumulate_gradients成功！\n");
    // printf("\n");
}

void update_fc_layer(FullyConnectedLayer* fc_layer, int batch_size, int input_size, float* weight_grad_accum, float* m, float* v, float learning_rate, int timestep) {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;
    float* weights = fc_layer->weights;
    int output_size = fc_layer->num_classes;
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            int idx = i * input_size + j;
            float grad_avg = weight_grad_accum[idx] / (float) batch_size;

            // Update biased first moment estimate
            m[idx] = beta1 * m[idx] + (1 - beta1) * grad_avg;

            // Update biased second raw moment estimate
            v[idx] = beta2 * v[idx] + (1 - beta2) * grad_avg * grad_avg;

            // Compute bias-corrected first moment estimate
            float m_hat = m[idx] / (1 - powf(beta1, timestep));

            // Compute bias-corrected second raw moment estimate
            float v_hat = v[idx] / (1 - powf(beta2, timestep));

            // Update weights using Adam rule
            weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);

            // Reset the accumulated gradient for the next batch
            weight_grad_accum[idx] = 0.0f;
        }
    }
}
