#ifndef __OPENCV-HELPER_H__
#define __OPENCV-HELPER_H__

#include <opencv2/core.hpp>
#include <string>

template<int n>
struct Typer
{
};

template<>
struct Typer<CV_8UC1>
{
    uint8_t Type;
};

template<>
struct Typer<CV_8UC2>
{
    cv::Vec<uint8_t, 2> Type;
};

template<>
struct Typer<CV_8UC3>
{
    cv::Vec<uint8_t, 3> Type;
};

template<>
struct Typer<CV_8SC1>
{
    int8_t Type;
};

template<>
struct Typer<CV_8SC2>
{
    cv::Vec<int8_t, 2> Type;
};

template<>
struct Typer<CV_8SC3>
{
    cv::Vec<int8_t, 3> Type;
};

template<>
struct Typer<CV_16UC1>
{
    uint16_t Type;
};

template<>
struct Typer<CV_16UC2>
{
    cv::Vec<uint16_t, 2> Type;
};

template<>
struct Typer<CV_16UC3>
{
    cv::Vec<uint16_t, 3> Type;
};

template<>
struct Typer<CV_16SC1>
{
    int16_t Type;
};

template<>
struct Typer<CV_16SC2>
{
    cv::Vec<int16_t, 2> Type;
};

template<>
struct Typer<CV_16SC3>
{
    cv::Vec<int16_t, 3> Type;
};

template <int x = 0>
Typer<x> getMatrixType()
{
    return Typer<x>(0);
}

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
    uint32_t VertexCnt = 0;

    switch(disparcityMap.type())
    {
    	case 0:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<uint8_t>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 1:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<int8_t>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 2:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<uint16_t>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 3:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<int16_t>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 4:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<int32_t>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 5:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<float>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	case 6:
    	{
			for (int y = 0; y < disparcityMap.rows; y++)
				for (int x = 0; x < disparcityMap.cols; x++)
					if(disparcityMap.at<double>(y, x) > disparcityMapMin)
						VertexCnt++;
    	}break;
    	default:
    	{
    		cerr << "Incorrect disparcity map" << endl;
    		return;
    	};
    }

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

#endif // __OPENCV-HELPER_H__