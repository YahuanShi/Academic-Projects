#ifndef CV_CONTROL_H
#define CV_CONTROL_H
#include "GlobalVariables.h"
#include <pthread.h>

// opencv includes
#include <opencv2/core.hpp>
//#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui.hpp>

// Names of the windows where we display our images
#define VS_WINDOW_NAME	"Annotated Camera image"
#define BACKPROJ_HIST_IMAGE_WINDOW_NAME	"Object Likelihood (backproject)"
#define BACKPROJ_HIST_PROC_IMAGE_WINDOW_NAME	"Object Likelihood (backproject_processed)"
#define SV_MASK_WINDOW_NAME "Saturation and Value based Mask"
#define MESSAGE_WINDOW_NAME	"Message"
/**
* Circle
* A simple structure encoding a the parameters of a detected circle in an image
*/
struct Circle
{
	// Default constructor
	Circle() : center(), radius(0.f) {}
	cv::Point2f center;
	float radius;
};

/**
* initVisualServoing
* Initialize the parameters necessary for visual servoing
* @param _f - Focal length of the camera [pixels]
* @param _img_width - Width of the camera images [pixels]
* @param _diameter_real - Diameter of the tracked circle in the real world [meters]
* @param _diameter_desired_px - Desired diameter of the tracked circle in the image [pixels]
* @param _dt - Delta time (used as scaling factor for the commanded velocity)
* @param _desired_s_opencvf - Vector that contains the desired feature values in opencv frame
*/
void initVisualServoing(float _f, float _img_width,float _img_height, float _diameter_real, float _diameter_desired_px, float _dt, PrVector3 _desired_s_opencvf);

/**
* initFeatureDetector
* Initialize the visual detector for circles in the camera images
*/
void initFeatureDetector();
/**
* findCircleFeature
* Detects a circle in the image and stores its parameters
* @param color_img - Original color image. You can display your results here
* @param backproject_img - Backprojection of the image using the histogram. You should find circles in this image
* @param crcl - Structure to save the parameters of the detected circle
*/
bool findCircleFeature(cv::Mat& color_img, cv::Mat &backproject_img, Circle& crcl);

/**
* estimateCircleDepth
* Estimates the depth of the tracked circle wrt the camera
* @param f - Focal length of the camera [pixels]
* @param diameter - Diameter of the tracked circle in the real world [meters]
* @param crcl - Structure with the parameters of the detected circle in the camera images
*/
float estimateCircleDepth(float f, float diameter, Circle &crcl);


/**
* transformFromOpenCVFToFF
* Transform a feature vector from openCV frame (origin in upper left corner of the image) to feature frame (origin at the center of the image)
*
* @param vector_opencvf - feature vector defined in opencv frame
* @param vector_ff - feature vector defined in feature frame
*/
void transformFromOpenCVFToFF(PrVector3 vector_opencvf, PrVector3& vector_ff);

/**
* transformStepFromCFToEEF
* Transform the desired step vector from camera frame to end-effector frame
*
* @param vector_cf - feature vector defined in camera frame
* @param vector_eef - feature vector defined in end-effector frame
*/
void transformVelocityFromCFToEEF(PrVector3 vector_cf, PrVector3& vector_eef);

/**
* transformStepFromEEFToBF
* Transform the desired step vector from end-effector frame to base frame
*
* @param x_current_bf - current state of the robot - pose of the end-effector in base frame coordinates
* @param vector_eef - feature vector defined in end-effector frame
* @param vector_bf - feature vector defined in base frame
*/
void transformVelocityFromEEFToBF(PrVector x_current_bf, PrVector3 vector_eef, PrVector3& vector_bf);

/**
* controlRobot
* The real visual servoing
* @param crcl - Structure with the parameters of the detected circle in the camera images
* @param x - Pose of the end effector wrt the base frame (7 dof!)
* @param img - Original color image. You can display your results here
* @param cmdbuf - Command buffer with the computed command for the robot
*/
void controlRobot(Circle& crcl, PrVector &x, cv::Mat& img, char *cmdbuf);

// This macro is defined during compilation as a preprocessor directive (in the options of the visual studio project, differently for each compilation type)
// The macro is defined for compilation types DEBUG and RELEASE (simulation), but not for DEBUG_RCLIENT (real robot)
// Undefine the macro to grab images from a real camera
//#undef	USE_SIMULATED_CAMERA

#endif // CV_CONTROL_H
