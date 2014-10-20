// cv includes
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>

// boost includes
#include <boost/thread/mutex.hpp>
#include "boost/filesystem.hpp"

// pcl includes
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

//ros and msg-type includes
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
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

		/// Cycle Through Images
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
		cob_people_detection_msgs::stamped_string set_path, file_name;
		int shot_, persp_,set_;
		std::stringstream xyz_stream_, bmp_stream_;

		/// process request for next set. If current perspective and shot are unavailable for next set, will look for next available item in the same direction as the set change before searching next set
		bool changeSet(int incr);

		/// process request for next perspective. If current shot is unavailable for next perspective, will look for next available item in same direction as perspective change before searching next perspective
		bool changePersp(int incr);
		/// process request for next shot. tries 3 steps before returning false or coming out on the previously selected shot.
		bool changeShot(int incr);
		/// check file for availability, used by change* functions above
		bool checkFile();
		/// create pointcloud from current file selection
		void createPcl();
		/// publishes last pointcloud created to camera topic for cob_people_detection, along with it's related set and filename message for recognition testing.
		void publish();
};
}
