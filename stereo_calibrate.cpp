#include <iostream>
#include <filesystem>
#include <string>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

using std::filesystem::directory_iterator;

Size Nmbr(10,7);
float squareSize = 2e-2;

void generateObjPoints(std::vector<Point3f> &objp)
{
    objp.clear();
    objp.reserve(Nmbr.area());

    for( int i = 0; i < Nmbr.height; i++ )
    {
        for( int j = 0; j < Nmbr.width; j++ )
        {
            objp.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
        }
    }
}

static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       const vector<Point3f>& newObjPoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE);

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if( !newObjPoints.empty() )
    {
        fs << "grid_points" << newObjPoints;
    }
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

    TermCriteria criteria(TermCriteria::EPS |TermCriteria::MAX_ITER, 3000, 1e-5);

    std::vector<Point3f> objp;
    generateObjPoints(objp);

    vector<std::vector<cv::Point3f>> objpoints_L;
    std::vector<std::vector<cv::Point2f>> imgpoints_L;

    vector<std::vector<cv::Point3f>> objpoints_R;
    std::vector<std::vector<cv::Point2f>> imgpoints_R;

    Mat imgGray_L;
    Mat imgGray_R;
    int goodImgCounter = 0;

    for(auto file : files)
    {
        Mat img = imread(file);
        if (img.data == NULL)
            continue;

        int startX_L =    0, startY_L = 0, width_L = 1280, height_L = 800;
        int startX_R = 1280, startY_R = 0, width_R = 1280, height_R = 800;

        Mat ROI_L(img, Rect(startX_L, startY_L, width_L, height_L));
        Mat ROI_R(img, Rect(startX_R, startY_R, width_R, height_R));

        vector<Point2f> corners_L;
        vector<Point2f> corners_R;

        cvtColor(ROI_L, imgGray_L, COLOR_BGR2GRAY);
        cvtColor(ROI_R, imgGray_R, COLOR_BGR2GRAY);

        bool found_L = findChessboardCorners(imgGray_L, Nmbr, corners_L,  CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        bool found_R = findChessboardCorners(imgGray_R, Nmbr, corners_R,  CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if(found_L && found_R)
        {
            cornerSubPix(imgGray_L, corners_L, Size(5,5), Size(-1,-1), criteria);
            cornerSubPix(imgGray_R, corners_R, Size(5,5), Size(-1,-1), criteria);

            imgpoints_L.push_back(corners_L);
            objpoints_L.push_back(objp);

            imgpoints_R.push_back(corners_R);
            objpoints_R.push_back(objp);

            goodImgCounter++;
        }
    }

    Mat cameraMatrix_L, distCoeffs_L;
    vector <Mat> R_L, T_L;

    double data_L[10] = {
              1500, 0, static_cast <double> (imgGray_L.cols) / 2.0
            , 0, 1500, static_cast <double> (imgGray_L.rows) / 2.0
            , 0, 0, 1};

    cameraMatrix_L = Mat(3, 3, CV_64F,
            data_L);

    double RMS = calibrateCamera(objpoints_L, imgpoints_L, imgGray_L.size(), cameraMatrix_L, distCoeffs_L, R_L, T_L,
            CALIB_FIX_PRINCIPAL_POINT| CALIB_THIN_PRISM_MODEL );

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "camera_data_" <<std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S_L.yml");
    string fileName = ss.str();

    saveCameraParams(fileName, imgGray_L.size(), Nmbr, squareSize, 0, 0, cameraMatrix_L, distCoeffs_L, R_L, T_L, vector <float >(), imgpoints_L, vector<Point3f>(), 0);

    cout << "cameraMatrix : " << endl << cameraMatrix_L << std::endl;
    cout << "distCoeffs : " << distCoeffs_L << std::endl;
    cout << "Rotation vector : " << R_L[0] * 180 / 3.14 << std::endl;
    cout << "Translation vector : " << T_L[0] << std::endl;
    cout << "RMS : " << RMS << endl;
    cout << "Good img: " << goodImgCounter << endl;
    cout << "img: " << files.size() << endl;




    Mat cameraMatrix_R, distCoeffs_R;
    vector <Mat> R_R,T_R;

    double data_R[10] = {
              800, 0, static_cast <double> (imgGray_R.cols) / 2.0
            , 0, 800, static_cast <double> (imgGray_R.rows) / 2.0
            , 0, 0, 1};

    cameraMatrix_R = Mat(3, 3, CV_64F,
            data_R);

    RMS = calibrateCamera(objpoints_R, imgpoints_R, imgGray_R.size(), cameraMatrix_R, distCoeffs_R, R_R, T_R,
            CALIB_FIX_PRINCIPAL_POINT| CALIB_THIN_PRISM_MODEL );

    now = std::chrono::system_clock::now();
    in_time_t = std::chrono::system_clock::to_time_t(now);
    ss.str("");
    ss << "camera_data_" <<std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S_R.yml");
    fileName = ss.str();

    saveCameraParams(fileName, imgGray_R.size(), Nmbr, squareSize, 0, 0, cameraMatrix_R, distCoeffs_R, R_R, T_R, vector <float >(), imgpoints_R, vector<Point3f>(), 0);

    cout << "cameraMatrix : " << endl << cameraMatrix_R << std::endl;
    cout << "distCoeffs : " << distCoeffs_R << std::endl;
    cout << "Rotation vector : " << R_R[0] * 180 / 3.14 << std::endl;
    cout << "Translation vector : " << T_R[0] << std::endl;
    cout << "RMS : " << RMS << endl;
    cout << "Good img: " << goodImgCounter << endl;
    cout << "img: " << files.size() << endl;


    cout << "Starting Calibration\n";
    Mat CM1 = cameraMatrix_L;
    Mat CM2 = cameraMatrix_R;
    Mat D1 = distCoeffs_L, D2 = distCoeffs_R;
    Mat R, T, E, F;

    stereoCalibrate(objpoints_L, imgpoints_L, imgpoints_R,
                    CM1, D1, CM2, D2, imgGray_L.size(), R, T, E, F,
                    0 ,
                    TermCriteria(TermCriteria::EPS |TermCriteria::MAX_ITER, 3000, 1e-5)
                    );

    now = std::chrono::system_clock::now();
    in_time_t = std::chrono::system_clock::to_time_t(now);
    ss.str("");
    ss << "camera_data_" <<std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S_stereo.yml");
    fileName = ss.str();

    FileStorage fs1(fileName, FileStorage::WRITE);
    fs1 << "CM1" << CM1;
    fs1 << "CM2" << CM2;
    fs1 << "D1" << D1;
    fs1 << "D2" << D2;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "E" << E;
    fs1 << "F" << F;

    cout << "Done Calibration\n";

    cout << "Starting Rectification\n";

    Mat R1, R2, P1, P2, Q;
    stereoRectify(CM1, D1, CM2, D2, imgGray_L.size(), R, T, R1, R2, P1, P2, Q);
    fs1 << "R1" << R1;
    fs1 << "R2" << R2;
    fs1 << "P1" << P1;
    fs1 << "P2" << P2;
    fs1 << "Q" << Q;

    cout << "Done Rectification\n";

    return 0;
}