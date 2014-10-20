#ifdef __LINUX__
	#include "cob_people_detection/face_detector.h"
	#include "cob_people_detection/face_recognizer.h"
#else
#endif

// ROS includes
#include <ros/ros.h>
#include <ros/package.h>

//boost includes
#include<boost/filesystem.hpp>

// Actions
#include <actionlib/server/simple_action_server.h>
#include <cob_people_detection/loadModelAction.h>

// time for labeling of output file
#include <sys/time.h>

// stream to handle path-assembly
#include <fstream>


namespace ipa_PeopleDetector {

typedef actionlib::SimpleActionServer<cob_people_detection::loadModelAction> LoadModelServer;

class RgbdDbRecognitionTest
{
public:

	/// Constructor
	/// @param nh ROS node handle
	RgbdDbRecognitionTest(ros::NodeHandle nh);
	~RgbdDbRecognitionTest(void); ///< Destructor

	/// cycles through requested images
	void TestRecognition();

protected:


	/// Find Faces in Image
	void GetFaces();

	/// Recognize Faces
	void RecognizeFaces();

	/// Load a new model while running - not used currently.
	bool loadModel(std::vector<std::string>& identification_labels_to_recognize);

	/// Writes beginning of output file: labels loaded for recognition, labels and perspectives used in this test.
	void writeSetup(std::ofstream& output_textfile);

	/// Compares label axis tag to perspective of tested file
	bool checkOrientation(char label_tag, int perspective);

	ros::NodeHandle node_handle_;

	/// Face Detector instance created here, to avoid loss of ros messages influencing test.
	FaceDetector face_detector_;
	/// Face Recognizer instance created here, to avoid loss of ros messages influencing test.
	FaceRecognizer face_recognizer_;
	LoadModelServer* load_model_server_; ///< Action server that handles load requests for a new recognition model

	// parameters
	std::string data_directory_;	///< path to the classifier model
	std::string rgbd_db_directory_; ///< path to database
	std::string classifier_directory_;	///< path to the face feature haarcascades
	bool enable_face_recognition_;	///< this flag enables or disables the face recognition step
	bool display_timing_;
	bool detail_;
	bool trans_labels_;

	int face_found_;
	std::vector<int> perspectives_;
	std::vector<std::string> sets_;
	std::vector<std::string> identification_labels_to_recognize_;
};

} // end namespace
