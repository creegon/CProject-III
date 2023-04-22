#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NetWork.h" // 假设您已将上述代码保存在fashion_mnist.h文件中

int main() {
    //学习率
    float learning_rate = 0.005f;

    // 分配内存空间
    ImageData* train_data = (ImageData*) malloc(NUM_TRAIN_IMAGES * sizeof(ImageData));
    ImageData* test_data = (ImageData*) malloc(NUM_TEST_IMAGES * sizeof(ImageData));

    
    ActivationType activation_RELU = RELU; //  ReLU 激活函数
    PaddingType CONV_PADDING_SAME = SAME; // SAME 填充

    // 初始化卷积层和池化层

    //unit1
    ConvLayer conv_layer_1,conv_layer_2;
    MaxPoolLayer max_pool_layer_1;
    initialize_conv_layer(&conv_layer_1, 1, 64, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_conv_layer(&conv_layer_2, 64, 64, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_pool_layer(&max_pool_layer_1, 64, 64, 2, 2); // 池化层的输入通道数与卷积层的输出通道数相同(unit1完后的尺寸：64*14*14)

    //对应的特征图
    int conv_output_size_1 = (IMG_SIZE - conv_layer_1.kernel_size + 2 * conv_layer_1.padding) / conv_layer_1.stride + 1;
    // float* conv_output_feature_map_1 = (float*) malloc(conv_layer_1.base.output_channels * conv_output_size_1 * conv_output_size_1 * sizeof(float));

    int conv_output_size_2 = (IMG_SIZE - conv_layer_2.kernel_size + 2 * conv_layer_2.padding) / conv_layer_2.stride + 1;
    // float* conv_output_feature_map_2 = (float*) malloc(conv_layer_2.base.output_channels * conv_output_size_2 * conv_output_size_2 * sizeof(float));

    int pooled_output_size_1 = (conv_output_size_2 - max_pool_layer_1.pool_size) / max_pool_layer_1.stride + 1;
    // float* pooled_output_feature_map_1 = (float*) malloc(conv_layer_2.base.output_channels * pooled_output_size_1 * pooled_output_size_1 * sizeof(float));


    //unit2
    ConvLayer conv_layer_3,conv_layer_4,conv_layer_5,conv_layer_999;
    MaxPoolLayer max_pool_layer_2;
    initialize_conv_layer(&conv_layer_999, 64, 128, 3, 1, CONV_PADDING_SAME, activation_RELU); //奇怪的bug？？？？！必须要在这里放一个没用的，否则会报错
    initialize_conv_layer(&conv_layer_5, 128, 256, 3, 1, CONV_PADDING_SAME, activation_RELU); 
    initialize_conv_layer(&conv_layer_4, 128, 128, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_conv_layer(&conv_layer_3, 64, 128, 3, 1, CONV_PADDING_SAME, activation_RELU);
        
    initialize_pool_layer(&max_pool_layer_2, 128, 128, 2, 2); // 池化层的输入通道数与卷积层的输出通道数相同（unit2完后的尺寸：128*7*7）
    
    //对应的特征图
    int conv_output_size_3 = (IMG_SIZE - conv_layer_3.kernel_size + 2 * conv_layer_3.padding) / conv_layer_3.stride + 1;
    // float* conv_output_feature_map_3 = (float*) malloc(conv_layer_3.base.output_channels * conv_output_size_3 * conv_output_size_3 * sizeof(float));

    int conv_output_size_4 = (IMG_SIZE - conv_layer_4.kernel_size + 2 * conv_layer_4.padding) / conv_layer_4.stride + 1;
    // float* conv_output_feature_map_4 = (float*) malloc(conv_layer_4.base.output_channels * conv_output_size_4 * conv_output_size_4 * sizeof(float));

    int pooled_output_size_2 = (conv_output_size_4 - max_pool_layer_2.pool_size) / max_pool_layer_2.stride + 1;
    // float* pooled_output_feature_map_2 = (float*) malloc(conv_layer_4.base.output_channels * pooled_output_size_2 * pooled_output_size_2 * sizeof(float));
        

    //unit3
    ConvLayer conv_layer_6;
    MaxPoolLayer max_pool_layer_3;
            
    initialize_conv_layer(&conv_layer_6, 256, 256, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_pool_layer(&max_pool_layer_3, 256, 256, 3, 2); // 池化层的输入通道数与卷积层的输出通道数相同（unit3完后的尺寸：256*3*3）

    //对应的特征图
    int conv_output_size_5 = (IMG_SIZE - conv_layer_5.kernel_size + 2 * conv_layer_5.padding) / conv_layer_5.stride + 1;
    // float* conv_output_feature_map_5 = (float*) malloc(conv_layer_5.base.output_channels * conv_output_size_5 * conv_output_size_5 * sizeof(float));

    int conv_output_size_6 = (IMG_SIZE - conv_layer_6.kernel_size + 2 * conv_layer_6.padding) / conv_layer_6.stride + 1;
    // float* conv_output_feature_map_6 = (float*) malloc(conv_layer_6.base.output_channels * conv_output_size_6 * conv_output_size_6 * sizeof(float));

    int pooled_output_size_3 = (conv_output_size_6 - max_pool_layer_3.pool_size) / max_pool_layer_3.stride + 1;
    // float* pooled_output_feature_map_3 = (float*) malloc(conv_layer_6.base.output_channels * pooled_output_size_3 * pooled_output_size_3 * sizeof(float));
    
    
    //unit4
    ConvLayer conv_layer_7,conv_layer_8;
    MaxPoolLayer max_pool_layer_4;
    initialize_conv_layer(&conv_layer_7, 256, 512, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_conv_layer(&conv_layer_8, 512, 512, 3, 1, CONV_PADDING_SAME, activation_RELU);
    initialize_pool_layer(&max_pool_layer_4, 512, 512, 2, 1); // 池化层的输入通道数与卷积层的输出通道数相同(unit4完后的尺寸：512*2*2)

    //对应的特征图
    int conv_output_size_7 = (IMG_SIZE - conv_layer_7.kernel_size + 2 * conv_layer_7.padding) / conv_layer_7.stride + 1;
    // float* conv_output_feature_map_7 = (float*) malloc(conv_layer_7.base.output_channels * conv_output_size_7 * conv_output_size_7 * sizeof(float));

    int conv_output_size_8 = (IMG_SIZE - conv_layer_8.kernel_size + 2 * conv_layer_8.padding) / conv_layer_8.stride + 1;
    // float* conv_output_feature_map_8 = (float*) malloc(conv_layer_8.base.output_channels * conv_output_size_8 * conv_output_size_8 * sizeof(float));

    int pooled_output_size_4 = (conv_output_size_8 - max_pool_layer_4.pool_size) / max_pool_layer_4.stride + 1;
    // float* pooled_output_feature_map_4 = (float*) malloc(conv_layer_8.base.output_channels * pooled_output_size_4 * pooled_output_size_4 * sizeof(float)); 
    
    //最终单元：全连接层
    FullyConnectedLayer fc_layer_1;
    initialize_fully_connected_layer(&fc_layer_1, 512, 2, 10);
    
    float *m = (float *)calloc(512 * 2 * 2 * 10, sizeof(float));
    float *v = (float *)calloc(512 * 2 * 2 * 10, sizeof(float));

    float *m_8 = (float *)calloc(512 * 512 * 3 * 3, sizeof(float));
    float *v_8 = (float *)calloc(512 * 512 * 3 * 3, sizeof(float));
    float *m_7 = (float *)calloc(512 * 256 * 3 * 3, sizeof(float));
    float *v_7 = (float *)calloc(512 * 256 * 3 * 3, sizeof(float));

    float *m_6 = (float *)calloc(256 * 256 * 3 * 3, sizeof(float));
    float *v_6 = (float *)calloc(256 * 256 * 3 * 3, sizeof(float));
    float *m_5 = (float *)calloc(256 * 128 * 3 * 3, sizeof(float));
    float *v_5 = (float *)calloc(256 * 128 * 3 * 3, sizeof(float));

    float *m_4 = (float *)calloc(128 * 128 * 3 * 3, sizeof(float));
    float *v_4 = (float *)calloc(128 * 128 * 3 * 3, sizeof(float));
    float *m_3 = (float *)calloc(128 * 64 * 3 * 3, sizeof(float));
    float *v_3 = (float *)calloc(128 * 64 * 3 * 3, sizeof(float));

    float *m_2 = (float *)calloc(64 * 64 * 3 * 3, sizeof(float));
    float *v_2 = (float *)calloc(64 * 64 * 3 * 3, sizeof(float));
    float *m_1 = (float *)calloc(64 * 1 * 3 * 3, sizeof(float));
    float *v_1 = (float *)calloc(64 * 1 * 3 * 3, sizeof(float));

     // 读取训练集数据
    if (read_fashion_mnist_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", train_data, NUM_TRAIN_IMAGES) == 0) {
        printf("Training data loaded successfully.\n");
        shuffle_dataset(train_data, NUM_TRAIN_IMAGES);
        int batch_size = 50; //每个batch的大小
        int num_batches = 5; //batch的数量(只取一千张图片。。跑的速度有点慢。。)
        Batch* train_batches = create_batches(train_data, NUM_TRAIN_IMAGES, batch_size, num_batches); //训练的批次

        int train_times = 20; //每一个batch，都训练二十次（本来想训练一百次，但是太慢了，所以只训练二十次）
        for (int i = 0; i < num_batches; i++) {
            printf("batch %d start training!\n", i);
            //反向传播

            //Adam算法更新权重矩阵

            float* fc_input_grad_accum = (float*)malloc(sizeof(float) * 512 * 2 * 2);

            //unit4
            float* conv_kernel_grad_accum_8 = (float*)malloc(sizeof(float) * 512 * 512 * 3 * 3); //卷积层反向传播的梯度
            
            float* conv_kernel_grad_accum_7 = (float*)malloc(sizeof(float) * 512 * 256 * 3 * 3); //卷积层反向传播的梯度
            

            //unit3
            float* conv_kernel_grad_accum_6 = (float*)malloc(sizeof(float) * 256 * 256 * 3 * 3); //卷积层反向传播的梯度
            
            float* conv_kernel_grad_accum_5 = (float*)malloc(sizeof(float) * 256 * 128 * 3 * 3); //卷积层反向传播的梯度
            

            //unit2
            float* conv_kernel_grad_accum_4 = (float*)malloc(sizeof(float) * 128 * 128 * 3 * 3); //卷积层反向传播的梯度
           
            float* conv_kernel_grad_accum_3 = (float*)malloc(sizeof(float) * 128 * 64 * 3 * 3); //卷积层反向传播的梯度
            

            //unit1
            float* conv_kernel_grad_accum_2 = (float*)malloc(sizeof(float) * 64 * 64 * 3 * 3); //卷积层反向传播的梯度
            
            float* conv_kernel_grad_accum_1 = (float*)malloc(sizeof(float) * 64 * 1 * 3 * 3); //卷积层反向传播的梯度
            


            //临时的梯度
            float* conv_input_grad_1 = (float*)malloc(sizeof(float) * 1 * 28 * 28);
            float* conv_input_grad_2 = (float*)malloc(sizeof(float) * 64 * 28 * 28); 
            float* pool_input_grad_1 = (float*)malloc(sizeof(float) * 64 * 28 * 28);

            float* conv_input_grad_3 = (float*)malloc(sizeof(float) * 64 * 14 * 14);
            float* conv_input_grad_4 = (float*)malloc(sizeof(float) * 128 * 14 * 14);
            float* pool_input_grad_2 = (float*)malloc(sizeof(float) * 128 * 14 * 14);

            float* conv_input_grad_5 = (float*)malloc(sizeof(float) * 128 * 7 * 7);
            float* conv_input_grad_6 = (float*)malloc(sizeof(float) * 256 * 7 * 7);
            float* pool_input_grad_3 = (float*)malloc(sizeof(float) * 256 * 7 * 7);

            float* conv_input_grad_7 = (float*)malloc(sizeof(float) * 256 * 3 * 3);
            float* conv_input_grad_8 = (float*)malloc(sizeof(float) * 512 * 3 * 3);
            float* pool_input_grad_4 = (float*)malloc(sizeof(float) * 512 * 3 * 3);

            float* fc_input_grad = (float*)malloc(sizeof(float) * 512 * 2 * 2);


                //训练20次
                for(int timestamp = 1; timestamp <= train_times; timestamp++){
                    float epoch_loss = 0.0;
                    for(int j = 0; j < batch_size; j++){
                        //读出第i个batch的第j个数据
                        ImageData cur_data = train_batches[i].data[j];

                        //直接在这里初始化！！！
                        float* conv_output_feature_map_1 = (float*) malloc(conv_layer_1.base.output_channels * conv_output_size_1 * conv_output_size_1 * sizeof(float));
                        float* conv_output_feature_map_2 = (float*) malloc(conv_layer_2.base.output_channels * conv_output_size_2 * conv_output_size_2 * sizeof(float));
                        float* pooled_output_feature_map_1 = (float*) malloc(conv_layer_2.base.output_channels * pooled_output_size_1 * pooled_output_size_1 * sizeof(float));

                        float* conv_output_feature_map_3 = (float*) malloc(conv_layer_3.base.output_channels * conv_output_size_3 * conv_output_size_3 * sizeof(float));
                        float* conv_output_feature_map_4 = (float*) malloc(conv_layer_4.base.output_channels * conv_output_size_4 * conv_output_size_4 * sizeof(float));
                        float* pooled_output_feature_map_2 = (float*) malloc(conv_layer_4.base.output_channels * pooled_output_size_2 * pooled_output_size_2 * sizeof(float));

                        float* conv_output_feature_map_5 = (float*) malloc(conv_layer_5.base.output_channels * conv_output_size_5 * conv_output_size_5 * sizeof(float));
                        float* conv_output_feature_map_6 = (float*) malloc(conv_layer_6.base.output_channels * conv_output_size_6 * conv_output_size_6 * sizeof(float));
                        float* pooled_output_feature_map_3 = (float*) malloc(conv_layer_6.base.output_channels * pooled_output_size_3 * pooled_output_size_3 * sizeof(float));

                        float* conv_output_feature_map_7 = (float*) malloc(conv_layer_7.base.output_channels * conv_output_size_7 * conv_output_size_7 * sizeof(float));
                        float* conv_output_feature_map_8 = (float*) malloc(conv_layer_8.base.output_channels * conv_output_size_8 * conv_output_size_8 * sizeof(float));
                        float* pooled_output_feature_map_4 = (float*) malloc(conv_layer_8.base.output_channels * pooled_output_size_4 * pooled_output_size_4 * sizeof(float)); 
                        float* fc_output_feature_map_1 = (float*) malloc(fc_layer_1.base.output_channels * sizeof(float));


                        //unit 1
                        // 调用卷积层前向传播函数
                        conv_forward(&conv_layer_1, cur_data.image, 28, conv_output_feature_map_1);
                        conv_forward(&conv_layer_2, conv_output_feature_map_1, 28, conv_output_feature_map_2);
                        // 调用池化层前向传播函数
                        max_pool_forward(&max_pool_layer_1, conv_output_feature_map_2, 28, pooled_output_feature_map_1);

                        //unit 2
                        // 调用卷积层前向传播函数
                        conv_forward(&conv_layer_3, pooled_output_feature_map_1, 14, conv_output_feature_map_3);
                        conv_forward(&conv_layer_4, conv_output_feature_map_3, 14, conv_output_feature_map_4);
                        // 调用池化层前向传播函数
                        max_pool_forward(&max_pool_layer_2, conv_output_feature_map_4, 14, pooled_output_feature_map_2);

                        //unit 3
                        // 调用卷积层前向传播函数
                        conv_forward(&conv_layer_5, pooled_output_feature_map_2, 7, conv_output_feature_map_5);
                        conv_forward(&conv_layer_6, conv_output_feature_map_5, 7, conv_output_feature_map_6);
                        // 调用池化层前向传播函数
                        max_pool_forward(&max_pool_layer_3, conv_output_feature_map_6, 7, pooled_output_feature_map_3);

                        //unit 4
                        // 调用卷积层前向传播函数
                        conv_forward(&conv_layer_7, pooled_output_feature_map_3, 3, conv_output_feature_map_7);
                        conv_forward(&conv_layer_8, conv_output_feature_map_7, 3, conv_output_feature_map_8);
                        // 调用池化层前向传播函数
                        max_pool_forward(&max_pool_layer_4, conv_output_feature_map_8, 3, pooled_output_feature_map_4);


                        //开始全连接，此时的size为512*2*2
                        //调用全连接层前向传播函数
                        fully_connected_forward(&fc_layer_1, pooled_output_feature_map_4, 2048, fc_output_feature_map_1);


                        //开始反向传播


                        //全连接层
                        //先用softmax函数，转化数据
                        softmax(fc_output_feature_map_1, fc_layer_1.base.output_channels,fc_output_feature_map_1);
                        //然后用交叉熵函数(这个loss是一个体现，不会用到实际计算)
                        float loss = cross_entropy_loss(fc_output_feature_map_1, fc_layer_1.base.output_channels, cur_data.label);
                        epoch_loss += loss;
                        //计算两者结合的梯度(只用把label那里减1即可)
                        fc_output_feature_map_1[cur_data.label] -= 1;
                        //计算输入的梯度
                        fc_accumulate_gradients(&fc_layer_1, 2048, 10, fc_output_feature_map_1, fc_input_grad, fc_input_grad_accum);
                        
                        
                         
                        //unit4    
                        max_pool_backward(&max_pool_layer_4, &conv_layer_8, conv_output_feature_map_8, pool_input_grad_4, fc_input_grad);
                        conv_accumulate_gradients(&conv_layer_8, conv_output_feature_map_7, conv_output_feature_map_8, conv_input_grad_8, pool_input_grad_4, conv_kernel_grad_accum_8, 3);
                        conv_accumulate_gradients(&conv_layer_7, pooled_output_feature_map_3, conv_output_feature_map_7, conv_input_grad_7, conv_input_grad_8, conv_kernel_grad_accum_7, 3);



                        // printf("\n");
                        //unit3
                        max_pool_backward(&max_pool_layer_3, &conv_layer_6, conv_output_feature_map_6, pool_input_grad_3, conv_input_grad_7 );
                        conv_accumulate_gradients(&conv_layer_6, conv_output_feature_map_5, conv_output_feature_map_6, conv_input_grad_6, pool_input_grad_3, conv_kernel_grad_accum_6, 7);
                        conv_accumulate_gradients(&conv_layer_5, pooled_output_feature_map_2, conv_output_feature_map_5, conv_input_grad_5, conv_input_grad_6, conv_kernel_grad_accum_5, 7);

                        // printf("\n");
                        //unit2
                        max_pool_backward(&max_pool_layer_2, &conv_layer_4, conv_output_feature_map_4, pool_input_grad_2, conv_input_grad_5 );
                        conv_accumulate_gradients(&conv_layer_4, conv_output_feature_map_3, conv_output_feature_map_4, conv_input_grad_4, pool_input_grad_2, conv_kernel_grad_accum_4, 14);
                        conv_accumulate_gradients(&conv_layer_3, pooled_output_feature_map_1, conv_output_feature_map_3, conv_input_grad_3, conv_input_grad_4, conv_kernel_grad_accum_3, 14);

                        // printf("\n");
                        //unit1
                        max_pool_backward(&max_pool_layer_1, &conv_layer_2, conv_output_feature_map_2, pool_input_grad_1, conv_input_grad_3 );
                        conv_accumulate_gradients(&conv_layer_2, conv_output_feature_map_1, conv_output_feature_map_2, conv_input_grad_2, pool_input_grad_1, conv_kernel_grad_accum_2, 28);
                        conv_accumulate_gradients(&conv_layer_1, cur_data.image, conv_output_feature_map_1, conv_input_grad_1, conv_input_grad_2, conv_kernel_grad_accum_1, 28);

                        // printf("\n");
                        
                        //清空临时梯度
                        memset(conv_input_grad_1, 0, sizeof(float) * 1 * 28 * 28);
                        memset(conv_input_grad_2, 0, sizeof(float) * 64 * 28 * 28);
                        memset(pool_input_grad_1, 0, sizeof(float) * 64 * 28 * 28);

                        memset(conv_input_grad_3, 0, sizeof(float) * 64 * 14 * 14);
                        memset(conv_input_grad_4, 0, sizeof(float) * 128 * 14 * 14);
                        memset(pool_input_grad_2, 0, sizeof(float) * 128 * 14 * 14);

                        memset(conv_input_grad_5, 0, sizeof(float) * 128 * 7 * 7);
                        memset(conv_input_grad_6, 0, sizeof(float) * 256 * 7 * 7);
                        memset(pool_input_grad_3, 0, sizeof(float) * 256 * 7 * 7);

                        memset(conv_input_grad_7, 0, sizeof(float) * 256 * 3 * 3);
                        memset(conv_input_grad_8, 0, sizeof(float) * 512 * 3 * 3);
                        memset(pool_input_grad_4, 0, sizeof(float) * 512 * 3 * 3);

                        memset(fc_input_grad, 0, sizeof(float) * 2048);

                        //debug
                        // printf("第%d张图片的第%d轮结束！\n", j, timestamp);
                        // printf("\n");
                    }
                    
                    //更新权重（同时将累积的归0）
                    update_conv_layer(&conv_layer_1, conv_kernel_grad_accum_1, batch_size, m_1, v_1, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_2, conv_kernel_grad_accum_2, batch_size, m_2, v_2, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_3, conv_kernel_grad_accum_3, batch_size, m_3, v_3, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_4, conv_kernel_grad_accum_4, batch_size, m_4, v_4, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_5, conv_kernel_grad_accum_5, batch_size, m_5, v_5, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_6, conv_kernel_grad_accum_6, batch_size, m_6, v_6, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_7, conv_kernel_grad_accum_7, batch_size, m_7, v_7, learning_rate, timestamp);
                    update_conv_layer(&conv_layer_8, conv_kernel_grad_accum_8, batch_size, m_8, v_8, learning_rate, timestamp);
                    update_fc_layer(&fc_layer_1, batch_size, 2048, fc_input_grad, m, v, learning_rate, timestamp);


                    float avg_loss = epoch_loss / batch_size;
                    printf("round %d loss: %.4f\n", timestamp, avg_loss);

                }

            }

            // 读取测试集数据
            if (read_fashion_mnist_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", test_data, NUM_TEST_IMAGES) == 0) {
                printf("Test data loaded successfully.\n");
                shuffle_dataset(test_data, NUM_TEST_IMAGES);

                int correct_predictions = 0;

                for (int i = 0; i < NUM_TEST_IMAGES / 100; i++) { //减少验证的量
                    // 获取测试图像和标签
                    ImageData cur_data = test_data[i];
                    int true_label = test_data[i].label;

                    //直接在这里初始化！！！
                    float* conv_output_feature_map_1 = (float*) malloc(conv_layer_1.base.output_channels * conv_output_size_1 * conv_output_size_1 * sizeof(float));
                    float* conv_output_feature_map_2 = (float*) malloc(conv_layer_2.base.output_channels * conv_output_size_2 * conv_output_size_2 * sizeof(float));
                    float* pooled_output_feature_map_1 = (float*) malloc(conv_layer_2.base.output_channels * pooled_output_size_1 * pooled_output_size_1 * sizeof(float));

                    float* conv_output_feature_map_3 = (float*) malloc(conv_layer_3.base.output_channels * conv_output_size_3 * conv_output_size_3 * sizeof(float));
                    float* conv_output_feature_map_4 = (float*) malloc(conv_layer_4.base.output_channels * conv_output_size_4 * conv_output_size_4 * sizeof(float));
                    float* pooled_output_feature_map_2 = (float*) malloc(conv_layer_4.base.output_channels * pooled_output_size_2 * pooled_output_size_2 * sizeof(float));

                    float* conv_output_feature_map_5 = (float*) malloc(conv_layer_5.base.output_channels * conv_output_size_5 * conv_output_size_5 * sizeof(float));
                    float* conv_output_feature_map_6 = (float*) malloc(conv_layer_6.base.output_channels * conv_output_size_6 * conv_output_size_6 * sizeof(float));
                    float* pooled_output_feature_map_3 = (float*) malloc(conv_layer_6.base.output_channels * pooled_output_size_3 * pooled_output_size_3 * sizeof(float));

                    float* conv_output_feature_map_7 = (float*) malloc(conv_layer_7.base.output_channels * conv_output_size_7 * conv_output_size_7 * sizeof(float));
                    float* conv_output_feature_map_8 = (float*) malloc(conv_layer_8.base.output_channels * conv_output_size_8 * conv_output_size_8 * sizeof(float));
                    float* pooled_output_feature_map_4 = (float*) malloc(conv_layer_8.base.output_channels * pooled_output_size_4 * pooled_output_size_4 * sizeof(float)); 
                    float* fc_output_feature_map_1 = (float*) malloc(fc_layer_1.base.output_channels * sizeof(float));
                    
                    //unit 1
                    // 调用卷积层前向传播函数
                    conv_forward(&conv_layer_1, cur_data.image, 28, conv_output_feature_map_1);
                    conv_forward(&conv_layer_2, conv_output_feature_map_1, 28, conv_output_feature_map_2);
                    // 调用池化层前向传播函数
                    max_pool_forward(&max_pool_layer_1, conv_output_feature_map_2, 28, pooled_output_feature_map_1);

                    //unit 2
                    // 调用卷积层前向传播函数
                    conv_forward(&conv_layer_3, pooled_output_feature_map_1, 14, conv_output_feature_map_3);
                    conv_forward(&conv_layer_4, conv_output_feature_map_3, 14, conv_output_feature_map_4);
                    // 调用池化层前向传播函数
                    max_pool_forward(&max_pool_layer_2, conv_output_feature_map_4, 14, pooled_output_feature_map_2);

                    //unit 3
                    // 调用卷积层前向传播函数
                    conv_forward(&conv_layer_5, pooled_output_feature_map_2, 7, conv_output_feature_map_5);
                    conv_forward(&conv_layer_6, conv_output_feature_map_5, 7, conv_output_feature_map_6);
                    // 调用池化层前向传播函数
                    max_pool_forward(&max_pool_layer_3, conv_output_feature_map_6, 7, pooled_output_feature_map_3);

                    //unit 4
                    // 调用卷积层前向传播函数
                    conv_forward(&conv_layer_7, pooled_output_feature_map_3, 3, conv_output_feature_map_7);
                    conv_forward(&conv_layer_8, conv_output_feature_map_7, 3, conv_output_feature_map_8);
                    // 调用池化层前向传播函数
                    max_pool_forward(&max_pool_layer_4, conv_output_feature_map_8, 3, pooled_output_feature_map_4);


                    //开始全连接，此时的size为512*2*2
                    //调用全连接层前向传播函数
                    fully_connected_forward(&fc_layer_1, pooled_output_feature_map_4, 2048, fc_output_feature_map_1);

                    //根据fc_output_feature_map_1的值，判断当前图片是哪个数字
                    int predicted_label = -1;
                    float max_value = -1.0f;
                    for (int i = 0; i < 10; i++) {
                        if (fc_output_feature_map_1[i] > max_value) {
                            max_value = fc_output_feature_map_1[i];
                            predicted_label = i;
                        }
                    }

                    // 如果预测正确，增加正确预测计数
                    if (predicted_label == true_label) {
                        correct_predictions++;
                    }
                }

                // 计算准确率并输出结果
                float accuracy = (float)correct_predictions / ((float)NUM_TEST_IMAGES / 100);
                printf("Verification results: accuracy is %.3f\n", accuracy);

            } else {
                printf("Failed to load test data.\n");
            }

        } else {
        printf("Failed to load training data.\n");
    }


    // 释放内存
    free(train_data);
    free(test_data);
    free(m);
    free(v);
    free(m_1);
    free(v_1);
    free(m_2);
    free(v_2);
    free(m_3);
    free(v_3);
    free(m_4);
    free(v_4);
    free(m_5);
    free(v_5);
    free(m_6);
    free(v_6);
    free(m_7);
    free(v_7);
    free(m_8);
    free(v_8);

    return 0;
}

