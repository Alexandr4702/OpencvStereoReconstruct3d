
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

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

int frame_count = 0;

int main(int argc, char **argv) {
    auto video = VideoCapture(2);

    // video.set(CAP_PROP_FRAME_WIDTH , 2560);
    // video.set(CAP_PROP_FRAME_HEIGHT, 960);

    // video.set(CAP_PROP_FRAME_WIDTH , 2560);
    // video.set(CAP_PROP_FRAME_HEIGHT, 720);

    video.set(CAP_PROP_FRAME_WIDTH , 1280);
    video.set(CAP_PROP_FRAME_HEIGHT, 480);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S") << "-calibration";
    std::string folder_name = ss.str();
    fs::create_directories(folder_name);

    uint32_t img_cnt = 0;
    while(1){
        cv::Mat image;
        video.read(image);
        if(image.data == nullptr)
            continue;

        cv::imshow("Arducam", image);

        char ret = waitKey(5);

        if (ret == 'q' || ret == 27)
            break;
        if (ret == 32)
        {
            std::string str = folder_name + "/" + std::to_string(img_cnt)+".tif";
            cv::imwrite(str, image);
            img_cnt++;
        }
    }

    printf("Close camera...");
    video.release();
    return 0;
}

