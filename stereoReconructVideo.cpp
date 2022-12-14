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

Camera cam;

bool openglExit = false;
bool mainThreadExit = false;

int numDisparities = 80;
int block_size = 4;

cv::Ptr<cv::StereoBM> stereoBMobject = cv::StereoBM::create(
    numDisparities,
    block_size
);

Mat  Q;

std::vector <Vector3f> vertex_point_buffer[2] = {
    {}
};

std::vector <uint8_t> color_point_buffer[2] = {
    {}
};

uint8_t opengl_current_bufer = 0;
bool opengl_request_change_buffer = false;

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

    // drawCube();
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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawImage3D(shader);
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
    auto video = VideoCapture(2);

    // video.set(CAP_PROP_FRAME_WIDTH , 2560);
    // video.set(CAP_PROP_FRAME_HEIGHT, 960);

    // video.set(CAP_PROP_FRAME_WIDTH , 2560);
    // video.set(CAP_PROP_FRAME_HEIGHT, 720);

    video.set(CAP_PROP_FRAME_WIDTH , 1280);
    video.set(CAP_PROP_FRAME_HEIGHT, 480);

    FileStorage StereoCalib("../camera_data_2022-09-20-23-05-04_stereo.yml", FileStorage::READ);

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
    video.read(img_gray);

    int startX_L =                    0 , startY_L = 0, width_L = img_gray.size[1] / 2, height_L = img_gray.size[0];
    int startX_R =  img_gray.size[1] / 2, startY_R = 0, width_R = img_gray.size[1] / 2, height_R = img_gray.size[0];

    initUndistortRectifyMap(CM1, D1, R1, P1, {width_L, height_L}, CV_32FC1, mapX_L, mapY_L);
    initUndistortRectifyMap(CM2, D2, R2, P2, {width_R, height_R}, CV_32FC1, mapX_R, mapY_R);

    cout << "Undistort complete\n";

    StereoSGBMSetteings stereoSGBMobject;

    stereoSGBMobject.init(Q);
    stereoSGBMobject.user_callback = [&](){
        uint8_t buffer_index = opengl_current_bufer == 0 ? 1: 0;
        stereoSGBMobject.fillBuffer(vertex_point_buffer[buffer_index], color_point_buffer[buffer_index]);
        opengl_current_bufer = buffer_index;
        imshow("disparsity", (stereoSGBMobject.disp - stereoSGBMobject.minDisparity) / stereoSGBMobject.numDisparities);
    };

    thread opengl(opengl_init, argc, argv);

    bool wait = true;
    for(;;)
    {
         video.read(img_gray);
         cvtColor(img_gray, img_gray, COLOR_BGR2GRAY);

        Mat ROI_L(img_gray, Rect(startX_L, startY_L, width_L, height_L));
        Mat ROI_R(img_gray, Rect(startX_R, startY_R, width_R, height_R));

        remap(ROI_L, stereoSGBMobject.imgU_L, mapX_L, mapY_L, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(ROI_R, stereoSGBMobject.imgU_R, mapX_R, mapY_R, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        stereoSGBMobject.update_data();

        if(wait)
        {
            imshow("L", stereoSGBMobject.imgU_L);
            imshow("R", stereoSGBMobject.imgU_R);

            char k = waitKey(10);

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
            // destroyWindow("L");
            // destroyWindow("R");
        }
        if(mainThreadExit)
            break;
    }
    opengl.join();
    return 0;
}
