
#include "arducam_mipicamera.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <linux/v4l2-controls.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv2/xphoto/white_balance.hpp>

#include <chrono>
#include <iomanip> // put_time
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

#define VCOS_ALIGN_DOWN(p,n) (((ptrdiff_t)(p)) & ~((n)-1))
#define VCOS_ALIGN_UP(p,n) VCOS_ALIGN_DOWN((ptrdiff_t)(p)+(n)-1,(n))
#define LOG(fmt, args...) fprintf(stderr, fmt "\n", ##args)

int frame_count = 0;
cv::Mat *get_image(CAMERA_INSTANCE camera_instance, int width, int height) {
    IMAGE_FORMAT fmt = {IMAGE_ENCODING_I420, 50};
    BUFFER *buffer = arducam_capture(camera_instance, &fmt, 3000);
    if (!buffer)
        return NULL;

    // The actual width and height of the IMAGE_ENCODING_RAW_BAYER format and the IMAGE_ENCODING_I420 format are aligned,
    // width 32 bytes aligned, and height 16 byte aligned.
    width = VCOS_ALIGN_UP(width, 32);
    height = VCOS_ALIGN_UP(height, 16);
    cv::Mat *image = new cv::Mat(cv::Size(width,(int)(height * 1.5)), CV_8UC1, buffer->data);
    cv::cvtColor(*image, *image, cv::COLOR_YUV2BGR_I420);
    arducam_release_buffer(buffer);
    return image;
}

int main(int argc, char **argv) {
    CAMERA_INSTANCE camera_instance;
    int width = 0, height = 0;
    char file_name[100];

    LOG("Open camera...");
    int res = arducam_init_camera(&camera_instance);
    if (res) {
        LOG("init camera status = %d", res);
        return -1;
    }

    width = 2560;
    height = 800;
    LOG("Setting the resolution...");
    res = arducam_set_resolution(camera_instance, &width, &height);
    if (res) {
        LOG("set resolution status = %d", res);
        return -1;
    } else {
        LOG("Current resolution is %dx%d", width, height);
        LOG("Notice:You can use the list_format sample program to see the resolution and control supported by the camera.");
    }

    arducam_software_auto_exposure(camera_instance, 1);

//     int exposure = 4200;
//     arducam_set_control(camera_instance, V4L2_CID_EXPOSURE, exposure);
//     printf("Current exposure is %d\n", exposure);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S") << "-calibration";
    std::string folder_name = ss.str();
    fs::create_directories(folder_name);

    uint32_t img_cnt = 0;
    while(1){
        cv::Mat *image = get_image(camera_instance, width, height);
        if(!image)
            continue;

        cv::imshow("Arducam", *image);

        char ret = waitKey(5);

        if (ret == 'q')
            break;
        if (ret == 32)
        {
            std::string str = folder_name + "/" + std::to_string(img_cnt)+".tif";
            cv::imwrite(str, *image);
            img_cnt++;
        }

        delete image;
    }

    LOG("Close camera...");
    res = arducam_close_camera(camera_instance);
    if (res) {
        LOG("close camera status = %d", res);
    }
    return 0;
}

