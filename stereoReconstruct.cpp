#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv-helper.hpp"

#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
// #include <SFML/Keyboard.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace cv;
using namespace Eigen;

using std::filesystem::directory_iterator;

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

bool openglExit = false;

class DataFromReconstruct
{
public:
float x;
float y;
float z;
uint8_t color;
};

std::vector <DataFromReconstruct> data[2] = {
    {},
    {}
};
uint8_t current_bufer = 0;

class Camera
{
    public:
    Vector3f getTranslation()
    {
        mtx.lock();
        Vector3f ret = translation;
        mtx.unlock();
        return ret;
    }
    Quaternionf getRotation()
    {
        mtx.lock();
        Quaternionf ret = rotation;
        mtx.unlock();
        return ret;
    }
    Vector3f getScale()
    {
        mtx.lock();
        Vector3f ret = scale;
        mtx.unlock();
        return ret;
    }
    void TranslateCam(Vector3f transl)
    {
        scoped_lock guard(mtx);
        translation += transl;
    }
    void rotateCam(Quaternionf rot)
    {
        scoped_lock guard(mtx);
        rotation = rot * rotation;
    }
    void ScaleCam(Vector3f scal)
    {
        scoped_lock guard(mtx);
        scale += scal;
    }
    private:
    Vector3f translation{0, 0, 0};
    Quaternionf rotation = Quaternionf::Identity();
    Vector3f       scale {1.0f, 1.0f, 1.0f};
    std::mutex mtx;
};

Camera cam;

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
Mat Out3D;

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

void drawCube()
{
    glBegin(GL_QUADS);//draw some squares
    glVertex3f(-1.f, -1.f, -1.f);
    glVertex3f(-1.f,  1.f, -1.f);
    glVertex3f( 1.f,  1.f, -1.f);
    glVertex3f( 1.f, -1.f, -1.f);

    glVertex3f(-1.f, -1.f, 1.f);
    glVertex3f(-1.f,  1.f, 1.f);
    glVertex3f( 1.f,  1.f, 1.f);
    glVertex3f( 1.f, -1.f, 1.f);

    glVertex3f(-1.f, -1.f, -1.f);
    glVertex3f(-1.f,  1.f, -1.f);
    glVertex3f(-1.f,  1.f,  1.f);
    glVertex3f(-1.f, -1.f,  1.f);

    glVertex3f(1.f, -1.f, -1.f);
    glVertex3f(1.f,  1.f, -1.f);
    glVertex3f(1.f,  1.f,  1.f);
    glVertex3f(1.f, -1.f,  1.f);

    glVertex3f(-1.f, -1.f,  1.f);
    glVertex3f(-1.f, -1.f, -1.f);
    glVertex3f( 1.f, -1.f, -1.f);
    glVertex3f( 1.f, -1.f,  1.f);

    glVertex3f(-1.f, 1.f,  1.f);
    glVertex3f(-1.f, 1.f, -1.f);
    glVertex3f( 1.f, 1.f, -1.f);
    glVertex3f( 1.f, 1.f,  1.f);
    glEnd();
}

void display(sf::Clock& Clock)
{
    Vector3f translation = cam.getTranslation();
    AngleAxisf rotation  = AngleAxisf(cam.getRotation());
    Vector3f   scale     = cam.getScale();

    // glTranslatef(translation.x(), translation.y(), translation.z());
    // glRotatef(rotation.angle(), rotation.axis().x(), rotation.axis().y(), rotation.axis().z());
    // glScalef(scale.x(), scale.y(), scale.z());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Apply some transformations for the cube
    glMatrixMode(GL_MODELVIEW);

    glLoadIdentity();

    glScalef(scale.x(), scale.y(), scale.z());
    glRotatef(rotation.angle() * 180 / M_PI, rotation.axis().x(), rotation.axis().y(), rotation.axis().z());
    glTranslatef(translation.x(), translation.y(), translation.z());

    glTranslatef(0.f, 0.f, -10.f);
    glRotatef( - Clock.getElapsedTime().asSeconds() * 50, 0, 1, 0);

    glColor3b(0, 0, 127);
    drawCube();

    glLoadIdentity();

    glScalef(scale.x(), scale.y(), scale.z());
    glRotatef(rotation.angle() * 180 / M_PI, rotation.axis().x(), rotation.axis().y(), rotation.axis().z());
    glTranslatef(translation.x(), translation.y(), translation.z());

    glTranslatef(0.f, 5.f, -10.f);
    glRotatef(Clock.getElapsedTime().asSeconds() * 50, 0, 1, 0);

    glColor3b(0, 0, 127);
    drawCube();
}

void keyPressedHandler()
{

}

void keyReleasedHandler()
{

}

void opengl_init(int argc, char** argv)
{
    // create the window
    sf::Window window(sf::VideoMode(800, 600), "OpenGL", sf::Style::Default, sf::ContextSettings(32));
    window.setVerticalSyncEnabled(true);

    // activate the window
    window.setActive(true);

    // load resources, initialize the OpenGL states, ...
    // Create a clock for measuring time elapsed
    sf::Clock Clock;

    //prepare OpenGL surface for HSR
	glClearDepth(1.f);
    glClearColor(0.3f, 0.3f, 0.3f, 0.f);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    //// Setup a perspective projection & Camera position
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.f, 1.f, 03.f, 100.0f);//fov, aspect, zNear, zFar
    // glOrtho(-1, 1, -1, 1, -100, 100);


    // run the main loop
    bool running = true;
    while (running)
    {
        // handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type )
            {
            case sf::Event::Closed:
            {
                running = false;
            } break;
            case sf::Event::Resized:
            {
                glViewport(0, 0, event.size.width, event.size.height);
            } break;
            case sf::Event::KeyPressed:
            {
                // event.key.code
            } break;
            case sf::Event::KeyReleased:
            {

            } break;
            default:
                break;
            }
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
        {
            cam.TranslateCam(Vector3f(0, 0, 0.1));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
        {
            cam.TranslateCam(Vector3f(0, 0, -0.1));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
        {
            float angle = 0.5 * M_PI / 180.0f;
            cam.rotateCam(Quaternionf(cos(angle / 2), 0,sin(angle / 2),  0));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
        {
            float angle = -0.5 * M_PI / 180.0f;
            cam.rotateCam(Quaternionf(cos(angle / 2), 0, sin(angle / 2), 0));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
        {
            float angle = 0.5 * M_PI / 180.0f;
            cam.rotateCam(Quaternionf(cos(angle / 2), 0, 0, sin(angle / 2)));
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E))
        {
            float angle = -0.5 * M_PI / 180.0f;
            cam.rotateCam(Quaternionf(cos(angle / 2), 0, 0, sin(angle / 2)));
        }
        // sf::Mouse::getPosition
        // clear the buffers
        display(Clock);
        // draw...

        // end the current frame (internally swaps the front and back buffers)
        window.display();

        if(openglExit)
            break;
    }
}

int main(int argc, char** argv)
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

    thread opengl(opengl_init, argc, argv);
    opengl.detach();

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
                openglExit = true;
                break;
            }
            if(k == 'w')
            {
                wait = false;
                destroyAllWindows();
                openglExit = true;
                continue;
            }
            destroyWindow(file + "_L");
            destroyWindow(file + "_R");
        }
    }
    return 0;
}