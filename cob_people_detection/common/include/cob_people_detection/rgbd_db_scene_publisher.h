// cv includes
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>

// boost includes
#include <boost/thread/mutex.hpp>
#include "boost/filesystem.hpp"

// ros includes
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include "cob_people_detection_msgs/stamped_string.h"

// pcl includes
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <fstream>


namespace ipa_PeopleDetector {

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class RgbdDbScenePublisher
{
	public:
		// Constructor
		RgbdDbScenePublisher(ros::NodeHandle nh);

		~RgbdDbScenePublisher(void); ///< Destructor

		// Cycle Through Images
		void PublishScenes();

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
		std::vector<int> perspectives_;
		std::string file_;
		cob_people_detection_msgs::stamped_string set_path, file_name;
		int total_published_msgs_, shot_, persp_;

		void Publish();

		void CreatePcl(std::stringstream& bmp_stream, std::stringstream& xyz_stream);
};
}
