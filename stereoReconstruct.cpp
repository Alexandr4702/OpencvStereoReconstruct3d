#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <thread>
#include <math.h>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv-helper.hpp"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>

#include "./eigen/Eigen/Core"
#include "./eigen/Eigen/Geometry"

using namespace std;
using namespace cv;
using namespace Eigen;

using std::filesystem::directory_iterator;

void drawCube();
void drawPoint();
void updateImage();
void fillBuffer(cv::Mat& disparcityMap, cv::Mat& points, cv::Mat& colors);

class DataFromReconstruct
{
public:
DataFromReconstruct(Vec3f& pos, uint8_t color):color0(color / 255.0f)
{
    x = pos[0];
    y = pos[1];
    z = pos[2];
}
DataFromReconstruct(Vector3f& pos, uint8_t color):color0(color / 255.0f)
{
    x = pos[0];
    y = pos[1];
    z = pos[2];
}
float x;
float y;
float z;
float color0;
float color1;
float color2;
};

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
        translation += rotation.inverse() * transl;
    }
    void rotateCam(Quaternionf rot)
    {
        scoped_lock guard(mtx);
        rotation = rot * rotation;
    }
    void setCamRotation(Quaternionf& rot)
    {
        if(rot.coeffs().hasNaN())
            return;
        scoped_lock guard(mtx);
        rotation = rot;
    }
    void ScaleCam(Vector3f scal)
    {
        scoped_lock guard(mtx);
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
    void printCameraParam(ostream& out)
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
    Vector3f translation{0, 0, 1};
    Quaternionf rotation = Quaternionf(0, 0, 1, 0);
    Vector3f       scale {1.0f, 1.0f, 1.0f};
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

Camera cam;

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
int StereoMode = StereoSGBM::MODE_SGBM;

bool openglExit = false;
bool mainThreadExit = false;

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
Mat  Q;

std::vector <DataFromReconstruct> point_buffer[2] = {
    {},
    {}
};

std::vector <Vector3f> vertex_point_buffer[2] = {
    {}
};

std::vector <uint8_t> color_point_buffer[2] = {
    {}
};

uint8_t opengl_current_bufer = 0;
bool opengl_request_change_buffer = false;
bool nextImage = false;


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

void drawImage3D(sf::Shader& shader)
{
    glLoadIdentity();

    static const Matrix4f projective_matrix_ = Camera::projective_matrix(60.f, 1.f, 01.f, 100.0f);
    Eigen::Matrix4f mvp = projective_matrix_ * cam.getCameraMatrix();

    sf::Glsl::Mat4 dsa(mvp.data());

    shader.setUniform("mvp_matrix", dsa);

    sf::Shader::bind(&shader);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, &vertex_point_buffer[opengl_current_bufer][0][0]);
    glEnableVertexAttribArray(0);

    // glVertexPointer(3, GL_FLOAT, sizeof(point_buffer[opengl_current_bufer][0]), &point_buffer[opengl_current_bufer][0].x);
    // glEnableClientState(GL_VERTEX_ARRAY);

    glVertexAttribPointer(1, 1, GL_UNSIGNED_BYTE, GL_FALSE, 0, &color_point_buffer[opengl_current_bufer][0]);
    glEnableVertexAttribArray(1);

    // glColorPointer(3, GL_FLOAT, sizeof(point_buffer[opengl_current_bufer][0]), &point_buffer[opengl_current_bufer][0].color0);
    // glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(5);
    glDrawArrays(GL_POINTS, 0, vertex_point_buffer[opengl_current_bufer].size());

    drawCube();
}

void updateImage()
{
    if(imgU_L.data == nullptr)
        return;
    TimeMeasure time;
    stereo->compute(imgU_L, imgU_R, disp);
    disp.convertTo(disp, CV_32F, 1.0f / 16.0f);
    reprojectImageTo3D(disp, Out3D, Q, true);
    fillBuffer(disp, Out3D, imgU_L);
}

void drawCube()
{
    double vertex[] =
    {
        -1.f,  1.f, -1.f,   255.0,
        1.f,  1.f, -1.f,    255.0,
        -1.f, -1.f, -1.f,   255.0,
        1.f, -1.f, -1.f,    255.0,
        -1.f, -1.f,  1.f,   255.0,
        -1.f,  1.f,  1.f,   255.0,
        1.f,  1.f,  1.f,    255.0,
        1.f, -1.f,  1.f,    255.0,
        -1.f, -1.f, -1.f,   255.0,
        -1.f,  1.f, -1.f,   255.0,
        -1.f,  1.f,  1.f,   255.0,
        -1.f, -1.f,  1.f,   255.0,
        1.f, -1.f, -1.f,    255.0,
        1.f,  1.f, -1.f,    255.0,
        1.f,  1.f,  1.f,    255.0,
        1.f, -1.f,  1.f,    255.0,
        -1.f, -1.f,  1.f,   255.0,
        -1.f, -1.f, -1.f,   255.0,
        1.f, -1.f, -1.f,    255.0,
        1.f, -1.f,  1.f,    255.0,
        -1.f,  1.f,  1.f,   255.0,
        -1.f,  1.f, -1.f,   255.0,
        1.f,  1.f, -1.f,    255.0,
        1.f,  1.f,  1.f    ,255.0,

      };

    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(vertex[0]) * 4, vertex);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 1, GL_DOUBLE, GL_FALSE, sizeof(vertex[0]) * 4, vertex+3);
    glEnableVertexAttribArray(1);

    glDrawArrays(GL_QUADS, 0, 24);
}

void display(sf::Clock& Clock, sf::Shader& shader )
{

    // glTranslatef(translation.x(), translation.y(), translation.z());
    // glRotatef(rotation.angle(), rotation.axis().x(), rotation.axis().y(), rotation.axis().z());
    // glScalef(scale.x(), scale.y(), scale.z());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawImage3D(shader);
}

void fillBuffer(cv::Mat& disparcityMap, cv::Mat& points, cv::Mat& colors)
{
    // TimeMeasure time;

    double disparcityMapMin;
    minMaxLoc(disparcityMap, &disparcityMapMin);
    cout << "min map: " <<disparcityMapMin << endl;

    uint8_t buffer_index = opengl_current_bufer == 0 ? 1: 0;

    Mat mask = disparcityMap > disparcityMapMin;
    uint32_t NumberOfPoints = cv::sum(mask)[0];

    vertex_point_buffer[buffer_index].clear();
    vertex_point_buffer[buffer_index].reserve(static_cast<size_t>(NumberOfPoints));

    color_point_buffer[buffer_index].clear();
    color_point_buffer[buffer_index].reserve(static_cast<size_t>(NumberOfPoints));

    for (int y = 0; y < disparcityMap.rows; y++)
        for (int x = 0; x < disparcityMap.cols; x++)
            if(disparcityMap.at<float>(y, x) > disparcityMapMin)
            {
                Vector3f& vec = points.at <Vector3f> (y, x);
                uint8_t& col = colors.at <uint8_t> (y, x);
                vertex_point_buffer[buffer_index].push_back(vec);
                color_point_buffer[buffer_index].push_back(col);
                // current_buffer.push_back({vec, col});
            }
    opengl_current_bufer = buffer_index;
}

void opengl_init(int argc, char** argv)
{
    // create the window
    sf::Window window(sf::VideoMode(800, 600), "OpenGL", sf::Style::Default, sf::ContextSettings(24));
    window.setVerticalSyncEnabled(true);
    glewInit();

    window.setActive(true);

    sf::Clock Clock;

    //prepare OpenGL surface for HSR
    glClearDepth(1.f);
    glClearColor(0.3f, 0.3f, 0.3f, 0.f);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    sf::Shader shader;

    if (!shader.loadFromFile("../vertex_shader.vert", "../fragment_shader.frag"))
    {
        cout << "Can't load vertex shader\r\n";
        return;
    }

    float x0 = 0,y0 = 0;
    Quaternionf q0 = cam.getRotation();

    while (!openglExit)
    {
        // handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type )
            {
            case sf::Event::Closed:
            {
                openglExit = true;
            } break;
            case sf::Event::Resized:
            {
                glViewport(0, 0, event.size.width, event.size.height);
            } break;
            case sf::Event::KeyPressed:
            {
                if(event.key.code == sf::Keyboard::Key::Escape)
                {
                    mainThreadExit = true;
                    openglExit = true;
                }
                if(event.key.code == sf::Keyboard::Key::N)
                {
                    nextImage = true;
                }
            } break;
            case sf::Event::KeyReleased:
            {

            } break;
            case sf::Event::MouseButtonPressed:
            {
                if(event.mouseButton.button == sf::Mouse::Button::Right)
                {
                    x0 = sf::Mouse::getPosition().x;
                    y0 = sf::Mouse::getPosition().y;
                    q0 = cam.getRotation();
                }
            } break;
            case sf::Event::MouseWheelMoved:
            {
                float delta = event.mouseWheel.delta * 0.1;
                cam.ScaleCam(Eigen::Vector3f(delta, delta, delta));
                cam.printCameraParam(cout);
            } break;
            default:
                break;
            }
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
        {
            cam.TranslateCam(Vector3f(0, 0, 0.05));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
        {
            cam.TranslateCam(Vector3f(0, 0, -0.05));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
        {
            cam.TranslateCam(Vector3f(0.05, 0, 0));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
        {
            cam.TranslateCam(Vector3f(-0.05, 0, 0));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift))
        {
            cam.TranslateCam(Vector3f( 0, 0.05, 0));
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            cam.TranslateCam(Vector3f(0, -0.05, 0));
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

        if(sf::Mouse::isButtonPressed(sf::Mouse::Button::Right))
        {
            sf::Vector2i mouse_pos = sf::Mouse::getPosition();
            sf::Vector2u size =  window.getSize();
            float x_rot = (mouse_pos.x - x0) / size.x / 1;
            float y_rot = (mouse_pos.y - y0) / size.y / 1;
            float w_rot = sqrt(1- x_rot*x_rot - y_rot*y_rot);
            Quaternionf q (w_rot, y_rot, x_rot, 0);
            q = q * q0;
            q.normalize();
            cam.setCamRotation(q);
        }

        display(Clock, shader);
        window.display();
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
    createTrackbar("minDisparity",      "trackbar", nullptr , 15  , minDisparity_CB);
    createTrackbar("numDisparities",    "trackbar", nullptr , 250, numDisparities_CB);
    createTrackbar("window_size",       "trackbar", nullptr , 20, windows_size_CB);
    createTrackbar("block_size",        "trackbar", nullptr , 100, block_size_CB);
    createTrackbar("disp12MaxDiff",     "trackbar", nullptr , 1000, disp12MaxDiff_CB);
    createTrackbar("preFilterCap",      "trackbar", nullptr , 1000, preFilterCap_CB);
    createTrackbar("uniquenessRatio",   "trackbar", nullptr , 1000, uniquenessRatio_CB);
    createTrackbar("speckleWindowSize", "trackbar", nullptr , 1000, speckleWindowSize_CB);
    createTrackbar("speckleRange",      "trackbar", nullptr , 1000, speckleRange_CB);
    createTrackbar("fullDP",            "trackbar", nullptr ,    4, fullDP_CB);

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

    thread opengl(opengl_init, argc, argv);

    bool wait = true;
    for(auto file : files)
    {
        Mat img_gray = imread(file, IMREAD_GRAYSCALE);
        // cvtColor(img, img_gray, COLOR_BGR2GRAY);

        int startX_L =    0, startY_L = 0, width_L = 1280, height_L = 800;
        int startX_R = 1280, startY_R = 0, width_R = 1280, height_R = 800;

        Mat ROI_L(img_gray, Rect(startX_L, startY_L, width_L, height_L));
        Mat ROI_R(img_gray, Rect(startX_R, startY_R, width_R, height_R));

        remap(ROI_L, imgU_L, mapX_L, mapY_L, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(ROI_R, imgU_R, mapX_R, mapY_R, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        updateImage();

        // string filename_stem = std::filesystem::path(file).stem() ;
        // write_ply(path + "/ply/" + filename_stem + ".ply", disp, Out3D, imgU_L);

        if(wait)
        {
            // imshow(file + "_L", imgU_L);
            // imshow(file + "_R", imgU_R);

            char k = waitKey(10);
            while(k == -1)
            {
                k = waitKey(40);
                if(mainThreadExit)
                    break;
                if(nextImage)
                {
                    nextImage = false;
                    break;
                }
            }

            if(k == 27)
            {
                openglExit = true;
                break;
            }
            if(k == 'p')
            {
                wait = false;
                destroyAllWindows();
                openglExit = true;
                continue;
            }
            // destroyWindow(file + "_L");
            // destroyWindow(file + "_R");
        }
        if(mainThreadExit)
            break;
    }
    opengl.join();
    return 0;
}
