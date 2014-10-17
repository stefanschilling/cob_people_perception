#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

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
	void callback(const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& Face_Detector_Msg, const cob_people_detection_msgs::stamped_string::ConstPtr& file_name,const cob_people_detection_msgs::stamped_string::ConstPtr& set_name, const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& head_msg);

	void isolateHeads();


protected:
	bool saveImages(cv::Mat& image, cv::Mat& xyz);
	ros::NodeHandle node_handle_;
	std::string data_directory_;
	int received_, saved_, faulty_det_;

	// synchronized Subscribers
	message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_people_detection_msgs::ColorDepthImageArray, cob_people_detection_msgs::stamped_string, cob_people_detection_msgs::stamped_string,cob_people_detection_msgs::ColorDepthImageArray> >* sync_input_;
	message_filters::Subscriber<cob_people_detection_msgs::ColorDepthImageArray> det_sub;
	message_filters::Subscriber<cob_people_detection_msgs::stamped_string> file_sub;
	message_filters::Subscriber<cob_people_detection_msgs::stamped_string> set_sub;
	message_filters::Subscriber<cob_people_detection_msgs::ColorDepthImageArray> head_sub;
};
}
