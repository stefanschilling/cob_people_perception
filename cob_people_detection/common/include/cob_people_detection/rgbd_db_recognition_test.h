//#ifndef __FACE_DETECTOR_NODE_H__
//#define __FACE_DETECTOR_NODE_H__

#ifdef __LINUX__
	#include "cob_people_detection/face_detector.h"
	#include "cob_people_detection/face_recognizer.h"
#else
#endif

// ROS includes
#include <ros/ros.h>
#include <ros/package.h>		// use as: directory_ = ros::package::getPath("cob_people_detection") + "/common/files/windows/";

// ROS message includes
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <cob_people_detection_msgs/DetectionArray.h>
#include <cob_people_detection_msgs/ColorDepthImageArray.h>

//boost includes
#include<boost/filesystem.hpp>

// Actions
#include <actionlib/server/simple_action_server.h>
#include <cob_people_detection/loadModelAction.h>

namespace ipa_PeopleDetector {

typedef actionlib::SimpleActionServer<cob_people_detection::loadModelAction> LoadModelServer;

class RgbdDbRecognitionTest
{
public:

	/// Constructor
	/// @param nh ROS node handle
	RgbdDbRecognitionTest(ros::NodeHandle nh);
	~RgbdDbRecognitionTest(void); ///< Destructor

	// Cycle Through Images
	void TestRecognition();

protected:


	// Find Faces in Image
	void GetFaces();

	// Recognize Faces
	void RecognizeFaces();

	bool loadModel(std::vector<std::string>& identification_labels_to_recognize);

	void writeSetup(std::ofstream& output_textfile);

	ros::NodeHandle node_handle_;

	FaceDetector face_detector_;	///< implementation of the face detector
	FaceRecognizer face_recognizer_;
	LoadModelServer* load_model_server_;				///< Action server that handles load requests for a new recognition model

	// parameters
	std::string data_directory_;	///< path to the classifier model
	std::string rgbd_db_directory_; ///< path to database
	std::string classifier_directory_;	///< path to the face feature haarcascades
	bool enable_face_recognition_;	///< this flag enables or disables the face recognition step
	bool display_timing_;
	bool detail_;
	int face_found;
	std::vector<int> perspectives_;
	std::vector<std::string> sets_;
	std::vector<std::string> identification_labels_to_recognize_;
};

} // end namespace
