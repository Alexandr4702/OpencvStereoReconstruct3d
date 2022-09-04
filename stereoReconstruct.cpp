#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv-helper.hpp"

using namespace std;
using namespace cv;

using std::filesystem::directory_iterator;


// stereo = cv.StereoSGBM_create(
//     minDisparity = min_disp,
//     numDisparities = num_disp,
//     blockSize = 12,
//     P1 = 8*3*window_size**2,
//     P2 = 32*3*window_size**2,
//     disp12MaxDiff = 1,
//     uniquenessRatio = 10,
//     speckleWindowSize = 1,
//     speckleRange = 1
// )
// initialize values for StereoSGBM parameters
int window_size = 5;

int minDisparity = 6;
int numDisparities = 80;
int block_size = 4; // 3 to 11 is recommended
int P1=8 * 3 * window_size * window_size;
int P2=32 * 3 * window_size * window_size;
int disp12MaxDiff=1;
int preFilterCap=1;
int uniquenessRatio=10;
int speckleWindowSize=1;
int speckleRange=1;
int StereoMode = StereoSGBM::MODE_SGBM;

// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
	minDisparity,
	numDisparities,
	block_size,
	P1,
	P2,
	disp12MaxDiff,
	preFilterCap,
	uniquenessRatio,
	speckleWindowSize,
	speckleRange,
	StereoMode
	);
Mat disp, disparity;
Mat imgU_L, imgU_R;

// Defining callback functions for the trackbars to update parameter values

void updateImage()
{
  stereo->compute(imgU_L, imgU_R, disp);
  disp.convertTo(disp, CV_32F, 1.0);
  disp /= 16.0f;
  disparity = (disp - (float)minDisparity)/((float)numDisparities);
  imshow("disparity", disparity);
}

static void minDisparity_CB( int val, void* )
{
  minDisparity = val + 1;
  stereo->setMinDisparity(minDisparity);

  updateImage();
}

static void numDisparities_CB( int val, void* )
{
  numDisparities = val + 1;
  stereo->setNumDisparities(numDisparities);

  updateImage();
}

static void block_size_CB( int val, void* )
{
  block_size = val;
  stereo->setBlockSize(block_size);

  updateImage();
}

static void windows_size_CB( int val, void* )
{
  window_size = val;
  P1 =  8 * 3 * window_size * window_size;
  P2 = 32 * 3 * window_size * window_size;
  stereo->setP1(P1);
  stereo->setP2(P2);

  updateImage();
}

static void disp12MaxDiff_CB( int val, void* )
{
  disp12MaxDiff = val;
  stereo->setDisp12MaxDiff(disp12MaxDiff);

  updateImage();
}

static void preFilterCap_CB( int val, void* )
{
  preFilterCap = val;
  stereo->setPreFilterCap(preFilterCap);

  updateImage();
}

static void uniquenessRatio_CB( int val, void* )
{
  uniquenessRatio = val;
  stereo->setUniquenessRatio(uniquenessRatio);

  updateImage();
}

static void speckleWindowSize_CB( int val, void* )
{
  speckleWindowSize = val;
  stereo->setSpeckleWindowSize(speckleWindowSize);

  updateImage();
}

static void speckleRange_CB( int val, void* )
{
  speckleRange = val;
  stereo->setSpeckleRange(speckleRange);

  updateImage();
}

static void fullDP_CB( int val, void* )
{
  StereoMode = val;
  stereo->setMode(StereoMode);

  updateImage();
}

int main()
{
    string path = "../2022-08-27-13-18-52-calibration";
    vector <string> files;

    for (const auto & file : directory_iterator(path))
    {
        if (file.is_directory())
            continue;
        if (file.path().string().find(".jpg") == string::npos && file.path().string().find(".tif") == string::npos)
            continue;

        files.push_back(file.path().string());
    }

    FileStorage StereoCalib("../camera_data_2022-08-27-21-16-36_stereo.yml", FileStorage::READ);

    Mat CM1;
    Mat CM2;
    Mat D1;
    Mat D2;
    Mat R1;
    Mat R2;
    Mat P1;
    Mat P2;
    Mat  Q;

    StereoCalib["CM1"] >> CM1;
    StereoCalib["CM2"] >> CM2;
    StereoCalib["D1" ] >> D1;
    StereoCalib["D2" ] >> D2;
    StereoCalib["R1" ] >> R1;
    StereoCalib["R2" ] >> R2;
    StereoCalib["P1" ] >> P1;
    StereoCalib["P2" ] >> P2;
    StereoCalib["Q" ]  >>  Q;

    Mat mapX_L, mapY_L, mapX_R, mapY_R;
    Mat img_gray;
    Mat Out3D;

    initUndistortRectifyMap(CM1, D1, R1, P1, {1280, 800}, CV_32FC1, mapX_L, mapY_L);
    initUndistortRectifyMap(CM2, D2, R2, P2, {1280, 800}, CV_32FC1, mapX_R, mapY_R);

    cout << "Undistort complete\n";

    // Creating a named window to be linked to the trackbars
    namedWindow("trackbar",cv::WINDOW_NORMAL);

    // Creating trackbars to dynamically update the StereoBM parameters
    createTrackbar("minDisparity",      "trackbar", &minDisparity        , 15  , minDisparity_CB);
    createTrackbar("numDisparities",    "trackbar", &numDisparities      , 1000, numDisparities_CB);
    createTrackbar("window_size",       "trackbar",   &window_size       , 1000, windows_size_CB);
	createTrackbar("block_size",        "trackbar",   &block_size       , 100, block_size_CB);
    createTrackbar("disp12MaxDiff",     "trackbar", &disp12MaxDiff       , 1000, disp12MaxDiff_CB);
    createTrackbar("preFilterCap",      "trackbar", &preFilterCap        , 1000, preFilterCap_CB);
    createTrackbar("uniquenessRatio",   "trackbar", &uniquenessRatio     , 1000, uniquenessRatio_CB);
    createTrackbar("speckleWindowSize", "trackbar", &speckleWindowSize   , 1000, speckleWindowSize_CB);
    createTrackbar("speckleRange",      "trackbar", &speckleRange        , 1000, speckleRange_CB);
    createTrackbar("fullDP",            "trackbar", &StereoMode          ,    4, fullDP_CB);

	bool wait = true;
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

        updateImage();

        reprojectImageTo3D(disp, Out3D, Q);

		string filename_stem = std::filesystem::path(file).stem() ;

        write_ply(path + "/ply/" + filename_stem + ".ply", disp, Out3D, imgU_L);

		if(wait)
		{
			imshow(file + "_L", imgU_L);
			imshow(file + "_R", imgU_R);

			char k = waitKey();
			if(k == 27)
			{
				break;
			}
			if(k == 'w')
			{
				wait = false;
				destroyAllWindows();
				continue;
			}
			destroyWindow(file + "_L");
			destroyWindow(file + "_R");
		}
    }
    return 0;
}