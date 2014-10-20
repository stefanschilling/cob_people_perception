#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>

#include "boost/filesystem.hpp"

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <cob_people_detection_msgs/DetectionArray.h>
#include <cob_people_detection_msgs/ColorDepthImageArray.h>
#include "cob_people_detection_msgs/stamped_string.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


namespace ipa_PeopleDetector {

class RgbdDbHeadIsolator
{
public:

	/// Constructor
	/// @param nh ROS node handle
	RgbdDbHeadIsolator(ros::NodeHandle nh);
	~RgbdDbHeadIsolator(void); ///< Destructor

	/// Callback to process related messages from Scene Publisher, Face Detector and Head Detector.
	/// @param face_detector_msg message from Face Detector, used to verify that exactly one face was detected in the head region.
	/// @param head_detector_msg message from Head Detector, used for the head image and xyz-data if one face was found in head region. Using Head Detector image, as Face Detector images are stripped of background information, sometimes making the detector unable to find the face again during recognition testing.
	/// @param file_name_msg message containing the file name of the source picture, which will be used again for saving the head image and xyz data.
	/// @param set_name_msg message containing the set of the source picture, to be used for creating and sorting into directories of each set.
	void callback(const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& face_detector_msg, const cob_people_detection_msgs::stamped_string::ConstPtr& file_name_msg,const cob_people_detection_msgs::stamped_string::ConstPtr& set_name_msg, const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& head_detector_msg);

protected:
	ros::NodeHandle node_handle_;

	/// folder to save head data to. Read from ros parameter.
	std::string rgbd_db_head_directory_;

	/// integers to keep track of proceedings.
	int received_, saved_, faulty_det_;

	// synchronized Subscribers
	message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_people_detection_msgs::ColorDepthImageArray, cob_people_detection_msgs::stamped_string, cob_people_detection_msgs::stamped_string,cob_people_detection_msgs::ColorDepthImageArray> >* sync_input_;
	message_filters::Subscriber<cob_people_detection_msgs::ColorDepthImageArray> det_sub;
	message_filters::Subscriber<cob_people_detection_msgs::stamped_string> file_sub;
	message_filters::Subscriber<cob_people_detection_msgs::stamped_string> set_sub;
	message_filters::Subscriber<cob_people_detection_msgs::ColorDepthImageArray> head_sub;
};
}
