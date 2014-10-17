#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <boost/thread/mutex.hpp>
#include "boost/lexical_cast.hpp"
#include "boost/filesystem.hpp"
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "cob_people_detection_msgs/stamped_string.h"

#include <fstream>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <algorithm>

namespace ipa_PeopleDetector {

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class RgbdDbManualPublisher
{
	public:
		// Constructor
		RgbdDbManualPublisher(ros::NodeHandle nh);

		~RgbdDbManualPublisher(void); ///< Destructor

		// Cycle Through Images
		void keyLoop();

		ros::NodeHandle node_handle_;

	protected:
		ros::Publisher set_pub_;
		ros::Publisher file_name_pub_;
		ros::Publisher scene_pub_;
		ros::Publisher img_pub_;

		sensor_msgs::PointCloud2 out_pc2;
		pcl::PointCloud<pcl::PointXYZRGB> pc;
		sensor_msgs::Image out_img;
		std::string path_to_db_;
		std::vector<std::string> sets_;
		std::string file_;
		cob_people_detection_msgs::stamped_string set_path, file_name;
		int shot_, persp_,set_;
		std::stringstream xyz_stream_, bmp_stream_;

		void publish();
		bool changeSet(int incr);
		bool changePersp(int incr);
		bool changeShot(int incr);
		bool checkFile();

		void createPcl();
};
}
