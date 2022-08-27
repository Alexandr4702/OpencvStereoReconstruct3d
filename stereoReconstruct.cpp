#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

using std::filesystem::directory_iterator;

// initialize values for StereoSGBM parameters
int numDisparities = 8;
int blockSize = 5;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;
int dispType = CV_16S;

// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
Mat disp, disparity;
Mat img_gray;
Mat imgU_L, imgU_R;

// Defining callback functions for the trackbars to update parameter values

static void on_trackbar1( int, void* )
{
    stereo->setNumDisparities(numDisparities*16);
    numDisparities = numDisparities*16;

    stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar2( int, void* )
{
  stereo->setBlockSize(blockSize*2+5);
  blockSize = blockSize*2+5;

    stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(preFilterCap);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar6( int, void* )
{
  stereo->setTextureThreshold(textureThreshold);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar7( int, void* )
{
  stereo->setUniquenessRatio(uniquenessRatio);
}

static void on_trackbar8( int, void* )
{
  stereo->setSpeckleRange(speckleRange);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar9( int, void* )
{
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  speckleWindowSize = speckleWindowSize*2;

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar10( int, void* )
{
  stereo->setDisp12MaxDiff(disp12MaxDiff);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

static void on_trackbar11( int, void* )
{
  stereo->setMinDisparity(minDisparity);

      stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disparity, CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    imshow("disparity", disparity);
}

int main()
{
    string path = "./2022-08-27-13-18-52-calibration";
    vector <string> files;

    for (const auto & file : directory_iterator(path))
    {
        if (file.is_directory())
            continue;
        if (file.path().string().find(".jpg") == string::npos && file.path().string().find(".tif") == string::npos)
            continue;

        files.push_back(file.path().string());
    }

    FileStorage StereoCalib("./camera_data_2022-08-27-21-16-36_stereo.yml", FileStorage::READ);

    Mat CM1;
    Mat CM2;
    Mat D1;
    Mat D2;
    Mat R1;
    Mat R2;
    Mat P1;
    Mat P2;

    StereoCalib["CM1"] >> CM1;
    StereoCalib["CM2"] >> CM2;
    StereoCalib["D1" ] >> D1;
    StereoCalib["D2" ] >> D2;
    StereoCalib["R1" ] >> R1;
    StereoCalib["R2" ] >> R2;
    StereoCalib["P1" ] >> P1;
    StereoCalib["P2" ] >> P2;

    Mat mapX_L, mapY_L, mapX_R, mapY_R;

    initUndistortRectifyMap(CM1, D1, R1, P1, {1280, 800}, CV_32FC1, mapX_L, mapY_L);
    initUndistortRectifyMap(CM2, D2, R2, P2, {1280, 800}, CV_32FC1, mapX_R, mapY_R);

    cout << "Undistort complete\n";

    // Creating a named window to be linked to the trackbars
    namedWindow("disparity",cv::WINDOW_NORMAL);
    resizeWindow("disparity",1280, 800);

    // Creating trackbars to dynamically update the StereoBM parameters
    createTrackbar("numDisparities", "disparity", &numDisparities, 18, on_trackbar1);
    createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
    createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
    createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
    createTrackbar("textureThreshold", "disparity", &textureThreshold, 100, on_trackbar6);
    createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar7);
    createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar8);
    createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar9);
    createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar10);
    createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar11);

    for(auto file : files)
    {
        Mat img = imread(file);
        cvtColor(img, img_gray, COLOR_BGR2GRAY);

        int startX_L =    0, startY_L = 0, width_L = 1280, height_L = 800;
        int startX_R = 1280, startY_R = 0, width_R = 1280, height_R = 800;

        Mat ROI_L(img_gray, Rect(startX_L, startY_L, width_L, height_L));
        Mat ROI_R(img_gray, Rect(startX_R, startY_R, width_R, height_R));

        remap(ROI_L, imgU_L, mapX_L, mapY_L, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(ROI_R, imgU_R, mapX_R, mapY_R, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        stereo->compute(imgU_L, imgU_R, disp);
        disp.convertTo(disparity, CV_32F, 1.0);
        disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
        imshow("disparity", disparity);


        imshow(file + "_L", imgU_L);
        imshow(file + "_R", imgU_R);

        char k = waitKey();
        destroyWindow(file + "_L");
        destroyWindow(file + "_R");

        if(k == 27)
        {
            break;
        }
    }
    return 0;
}