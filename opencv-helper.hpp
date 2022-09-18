#ifndef __OPENCV-HELPER_H__
#define __OPENCV-HELPER_H__

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <thread>
#include <math.h>
#include <mutex>
#include <functional>

#include <string>

#include "./eigen/Eigen/Core"
#include "./eigen/Eigen/Geometry"

void write_ply(std::string fileName, cv::Mat& disparcityMap, cv::Mat& points, cv::Mat& colors)
{
    using namespace std;
    using namespace cv;

    const char format[] =
#ifdef _WIN32
"ply\n\
format ascii 1.0\n\
element vertex %i\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
end_header\n";
#else
    "ply\r\n\
    format ascii 1.0\r\n\
    element vertex %i\r\n\
    property float x\r\n\
    property float y\r\n\
    property float z\r\n\
    property uchar red\r\n\
    property uchar green\r\n\
    property uchar blue\r\n\
    end_header\r\n";
#endif

    FILE* out_ply = fopen(fileName.data(), "w");

    double disparcityMapMin;
    minMaxLoc(disparcityMap, &disparcityMapMin);
    cout << "min map: " <<disparcityMapMin << endl;

    Mat mask = disparcityMap > disparcityMapMin;
    uint32_t VertexCnt = cv::sum(mask)[0];

    fprintf(out_ply, format, VertexCnt);

    switch(disparcityMap.type())
    {
    	case 0:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<uint8_t>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	case 1:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<int8_t>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	case 2:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<uint16_t>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	case 3:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<int16_t>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	case 4:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<int32_t>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	case 5:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<float>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						uint8_t col = colors.at<uint8_t> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col, col ,col );
#endif
					}
				}
			}
    	}break;
    	case 6:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
			{
				for (int x = 0; x < disparcityMap.cols; x++)
				{
					if(disparcityMap.at<double>(y, x) > disparcityMapMin)
					{
						Vec3f vec = points.at <Vec3f> (y, x);
						Vec3b col = colors.at <Vec3b> (y, x);
#ifdef _WIN32
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#else
						fprintf(out_ply, "%f %f %f %hhu %hhu %hhu \r\n", vec[0], vec[1], vec[2], col[2], col[1] ,col[0] );
#endif
					}
				}
			}
    	}break;
    	default:
    	{
    		cerr << "Incorrect disparcity map" << endl;
    		return;
    	};
    }
    fclose(out_ply);
}

class Camera
{
    public:
    Eigen::Vector3f getTranslation()
    {
        mtx.lock();
        Eigen::Vector3f ret = translation;
        mtx.unlock();
        return ret;
    }
    Eigen::Quaternionf getRotation()
    {
        mtx.lock();
        Eigen::Quaternionf ret = rotation;
        mtx.unlock();
        return ret;
    }
    Eigen::Vector3f getScale()
    {
        mtx.lock();
        Eigen::Vector3f ret = scale;
        mtx.unlock();
        return ret;
    }
    void TranslateCam(Eigen::Vector3f transl)
    {
        std::scoped_lock guard(mtx);
        translation += rotation.inverse() * transl;
    }
    void rotateCam(Eigen::Quaternionf rot)
    {
        std::scoped_lock guard(mtx);
        rotation = rot * rotation;
    }
    void setCamRotation(Eigen::Quaternionf& rot)
    {
        if(rot.coeffs().hasNaN())
            return;
        std::scoped_lock guard(mtx);
        rotation = rot;
    }
    void ScaleCam(Eigen::Vector3f scal)
    {
        std::scoped_lock guard(mtx);
        scale += scal;
    }
    Eigen::Matrix4f getCameraMatrix()
    {
        Eigen::Affine3f CamMatrix;
        CamMatrix.setIdentity();
        CamMatrix.scale(scale);
        CamMatrix.rotate(rotation);
        CamMatrix.translate(translation);

        return CamMatrix.matrix();
    }
    void printCameraParam(std::ostream& out)
    {
        out << translation.transpose() << "\r\n";
        out << rotation << "\r\n";
    }
    static Eigen::Matrix4f projective_matrix(float fovY, float aspectRatio, float zNear, float zFar)
    {
        float yScale = 1 / tan(fovY * M_PI / 360.0f);
        float xScale = yScale / aspectRatio;

        // float yScale = 1;
        // float xScale = 1;

        Eigen::Matrix4f pmat;
        pmat << xScale, 0, 0, 0,
                0, yScale, 0, 0,
                0, 0, -(zFar+zNear)/(zFar-zNear), -2*zNear*zFar/(zFar-zNear),
                0, 0, -1, 0;
        return pmat;
    }
    private:
    Eigen::Vector3f translation{0, 0, 1};
    Eigen::Quaternionf rotation = Eigen::Quaternionf(0, 0, 1, 0);
    Eigen::Vector3f       scale {1.0f, 1.0f, 1.0f};
    std::mutex mtx;
};

class TimeMeasure
{
public:
    TimeMeasure()
    {
        start = std::chrono::system_clock::now();
    }
    TimeMeasure(double* save)
    {
        save_diff = save;
        start = std::chrono::system_clock::now();
    }
    ~TimeMeasure()
    {
        stop = std::chrono::system_clock::now();
        auto diff = stop - start;
        if(save_diff)
            *save_diff = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
        std::cout << "Time in microsec: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << std::endl;
    }
private:
    double* save_diff = nullptr;
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point stop;
};

class StereoSGBMSetteings
{
    public:
    int window_size = 5;
    int minDisparity = 6;
    int numDisparities = 80;
    int block_size = 4; // 3 to 11 is recommended
    int P1=8 * 3 * window_size * window_size;
    int P2=32 * 3 * window_size * window_size;
    int disp12MaxDiff=1;
    int preFilterCap=1;
    int uniquenessRatio=53;
    int speckleWindowSize=1;
    int speckleRange=1;
    int StereoMode = cv::StereoSGBM::MODE_SGBM;

    cv::Mat Out3D, Q, disp;
    cv::Mat imgU_L, imgU_R;

    cv::Ptr<cv::StereoSGBM> stereoSGBMobject = cv::StereoSGBM::create(
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

    std::function<void()> user_callback = [](){};

    static void minDisparity_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);
        object->minDisparity = val + 1;
        object->stereoSGBMobject->setMinDisparity(object->minDisparity);

        object->update_data();
    }

    static void numDisparities_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->numDisparities = val + 1;
        object->stereoSGBMobject->setNumDisparities(object->numDisparities);

        object->update_data();
    }

    static void block_size_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->block_size = val;
        object->stereoSGBMobject->setBlockSize(object->block_size);

        object->update_data();
    }

    static void windows_size_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->window_size = val;
        object->P1 =  8 * 3 * object->window_size * object->window_size;
        object->P2 = 32 * 3 * object->window_size * object->window_size;
        object->stereoSGBMobject->setP1(object->P1);
        object->stereoSGBMobject->setP2(object->P2);

        object->update_data();
    }

    static void disp12MaxDiff_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->disp12MaxDiff = val;
        object->stereoSGBMobject->setDisp12MaxDiff(object->disp12MaxDiff);

        object->update_data();
    }

    static void preFilterCap_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->preFilterCap = val;
        object->stereoSGBMobject->setPreFilterCap(object->preFilterCap);

        object->update_data();
    }

    static void uniquenessRatio_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->uniquenessRatio = val;
        object->stereoSGBMobject->setUniquenessRatio(object->uniquenessRatio);

        object->update_data();
    }

    static void speckleWindowSize_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->speckleWindowSize = val;
        object->stereoSGBMobject->setSpeckleWindowSize(object->speckleWindowSize);

        object->update_data();
    }

    static void speckleRange_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->speckleRange = val;
        object->stereoSGBMobject->setSpeckleRange(object->speckleRange);

        object->update_data();
    }

    static void fullDP_CB( int val, void* object_)
    {
        StereoSGBMSetteings* object = reinterpret_cast<StereoSGBMSetteings*> (object_);

        object->StereoMode = val;
        object->stereoSGBMobject->setMode(object->StereoMode);

        object->update_data();
    }

    void update_data()
    {
        if(imgU_L.data == nullptr)
            return;
        TimeMeasure time;
        stereoSGBMobject->compute(imgU_L, imgU_R, disp);
        disp.convertTo(disp, CV_32F, 1.0f / 16.0f);
        cv::reprojectImageTo3D(disp, Out3D, Q, true);
        user_callback();
    }

    void fillBuffer(std::vector <Eigen::Vector3f>& vertex_point_buffer, std::vector <uint8_t>& color_point_buffer)
    {
        TimeMeasure time;

        double disparcityMapMin;
        minMaxLoc(disp, &disparcityMapMin);
        std::cout << "min map: " << disparcityMapMin << std::endl;

        cv::Mat mask = disp > disparcityMapMin;
        uint32_t NumberOfPoints = cv::sum(mask)[0];

        vertex_point_buffer.clear();
        vertex_point_buffer.reserve(static_cast<size_t>(NumberOfPoints));

        color_point_buffer.clear();
        color_point_buffer.reserve(static_cast<size_t>(NumberOfPoints));

        for (int y = 0; y < disp.rows; y++)
            for (int x = 0; x < disp.cols; x++)
                if(mask.at<uint8_t>(y, x))
                {
                    Eigen::Vector3f& vec = Out3D.at <Eigen::Vector3f> (y, x);
                    uint8_t& col = imgU_L.at <uint8_t> (y, x);
                    vertex_point_buffer.push_back(vec);
                    color_point_buffer.push_back(col);
                }
    }

    void init(cv::Mat& Q)
    {
        using namespace cv;

        this->Q = Q;

        namedWindow("trackbar",cv::WINDOW_NORMAL);

        // Creating trackbars to dynamically update the StereoBM parameters
        createTrackbar("minDisparity",      "trackbar", nullptr , 15  , minDisparity_CB     , this);
        createTrackbar("numDisparities",    "trackbar", nullptr , 250 , numDisparities_CB   , this);
        createTrackbar("window_size",       "trackbar", nullptr , 20  , windows_size_CB     , this);
        createTrackbar("block_size",        "trackbar", nullptr , 100 , block_size_CB       , this);
        createTrackbar("disp12MaxDiff",     "trackbar", nullptr , 1000, disp12MaxDiff_CB    , this);
        createTrackbar("preFilterCap",      "trackbar", nullptr , 1000, preFilterCap_CB     , this);
        createTrackbar("uniquenessRatio",   "trackbar", nullptr , 1000, uniquenessRatio_CB  , this);
        createTrackbar("speckleWindowSize", "trackbar", nullptr , 1000, speckleWindowSize_CB, this);
        createTrackbar("speckleRange",      "trackbar", nullptr , 1000, speckleRange_CB     , this);
        createTrackbar("fullDP",            "trackbar", nullptr ,    4, fullDP_CB           , this);

        setTrackbarPos("minDisparity",      "trackbar", minDisparity      );
        setTrackbarPos("numDisparities",    "trackbar", numDisparities    );
        setTrackbarPos("window_size",       "trackbar", window_size       );
        setTrackbarPos("block_size",        "trackbar", block_size        );
        setTrackbarPos("disp12MaxDiff",     "trackbar", disp12MaxDiff     );
        setTrackbarPos("preFilterCap",      "trackbar", preFilterCap      );
        setTrackbarPos("uniquenessRatio",   "trackbar", uniquenessRatio   );
        setTrackbarPos("speckleWindowSize", "trackbar", speckleWindowSize );
        setTrackbarPos("speckleRange",      "trackbar", speckleRange      );
        setTrackbarPos("fullDP",            "trackbar", StereoMode        );
    }
};

#endif // __OPENCV-HELPER_H__