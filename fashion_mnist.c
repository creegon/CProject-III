#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "Network.h"

int read_fashion_mnist_data(const char* image_file, const char* label_file, ImageData* dataset, int num_images) {
    char imagepath[100];
    char labelpath[100];
    sprintf(imagepath, "dataset/%s", image_file);
    sprintf(labelpath, "dataset/%s", label_file);
    FILE* img_fp = fopen(imagepath, "rb");
    FILE* lbl_fp = fopen(labelpath, "rb");

    if (!img_fp || !lbl_fp) {
        printf("无法打开文件！\n");
        return -1;
    }

    // 读取文件头信息
    uint32_t magic_number, num_imgs, img_width, img_height;
    uint32_t lbl_magic_number, lbl_num_labels;
    fread(&magic_number, sizeof(uint32_t), 1, img_fp);
    fread(&num_imgs, sizeof(uint32_t), 1, img_fp);
    fread(&img_width, sizeof(uint32_t), 1, img_fp);
    fread(&img_height, sizeof(uint32_t), 1, img_fp);
    fread(&lbl_magic_number, sizeof(uint32_t), 1, lbl_fp);
    fread(&lbl_num_labels, sizeof(uint32_t), 1, lbl_fp);

    // 大端转小端
    magic_number = __builtin_bswap32(magic_number);
    num_imgs = __builtin_bswap32(num_imgs);
    img_width = __builtin_bswap32(img_width);
    img_height = __builtin_bswap32(img_height);
    lbl_magic_number = __builtin_bswap32(lbl_magic_number);
    lbl_num_labels = __builtin_bswap32(lbl_num_labels);

    if (num_images > num_imgs || num_images > lbl_num_labels) {
        printf("指定的图像数量大于数据集大小！\n");
        fclose(img_fp);
        fclose(lbl_fp);
        return -1;
    }

    // 读取图像数据
    uint8_t pixel;
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < IMG_SIZE * IMG_SIZE; j++) {
            fread(&pixel, sizeof(uint8_t), 1, img_fp);
            dataset[i].image[j] = (float)pixel / 255.0f; // 转换为float并归一化
        }
        fread(&dataset[i].label, sizeof(uint8_t), 1, lbl_fp);
    }

    // 关闭文件
    fclose(img_fp);
    fclose(lbl_fp);

    //debug
    // printf("读取图像数据成功！\n");
    return 0;
}

//打乱数据集
void shuffle_dataset(ImageData* dataset, int num_images) {
    // 初始化随机数生成器
    srand(time(0));

    // 对数据集进行洗牌
    #pragma omp parallel for
    for (int i = num_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        ImageData temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }

    //debug
    // printf("数据集已打乱！\n");
}


// 创建批处理
Batch* create_batches(ImageData* dataset, int num_images, int batch_size, int num_batches) {
    // 分配批处理数组的内存
    Batch* batches = (Batch*)malloc(num_batches * sizeof(Batch));

    // 创建批处理
    for (int i = 0; i < num_batches; i++) {
        int start_index = i * batch_size;
        int end_index = start_index + batch_size;

        // 如果最后一个批次包含的图像数量小于批处理大小，则调整大小
        if (end_index > num_images) {
            end_index = num_images;
        }

        batches[i].data = &dataset[start_index];
        batches[i].size = end_index - start_index;
    }

    //debug
    // printf("批处理创建成功！\n");
    return batches;
}
