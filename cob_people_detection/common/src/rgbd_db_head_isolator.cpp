#ifdef __LINUX__
	#include "cob_people_detection/rgbd_db_head_isolator.h"
	#include "cob_vision_utils/GlobalDefines.h"
#else
#endif

using namespace ipa_PeopleDetector;

RgbdDbHeadIsolator::RgbdDbHeadIsolator(ros::NodeHandle n)
:node_handle_(n)
{
	node_handle_.getParam("/cob_people_detection/rgbd_db_head_directory_", rgbd_db_head_directory_);
	std::cout << "rgbd_db_head_directory_ = " << rgbd_db_head_directory_ << "\n";

	det_sub.subscribe(node_handle_, "/cob_people_detection/face_detector/face_positions", 1);
	set_sub.subscribe(node_handle_, "/cob_people_detection/rgbd_db_scene_publisher/set_path", 1);
	file_sub.subscribe(node_handle_, "/cob_people_detection/rgbd_db_scene_publisher/file_name",1);
	head_sub.subscribe(node_handle_, "/cob_people_detection/head_detector/head_positions", 1);


	sync_input_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_people_detection_msgs::ColorDepthImageArray, cob_people_detection_msgs::stamped_string, cob_people_detection_msgs::stamped_string,cob_people_detection_msgs::ColorDepthImageArray> >(4);
	sync_input_->connectInput(det_sub, file_sub, set_sub,head_sub);
	sync_input_->registerCallback(boost::bind(&RgbdDbHeadIsolator::callback, this, _1, _2, _3, _4));
	this->received_=0;
	this->saved_=0;
	this->faulty_det_=0;
}

RgbdDbHeadIsolator::~RgbdDbHeadIsolator()
{
}

void voidDeleter(const sensor_msgs::Image* const) {}


void RgbdDbHeadIsolator::callback(const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& detection_msg,const cob_people_detection_msgs::stamped_string::ConstPtr& file_name,const cob_people_detection_msgs::stamped_string::ConstPtr& set_name, const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& head_msg)
{
	received_++;
	cv_bridge::CvImageConstPtr cv_ptr;
	std::string set_path = set_name->data.c_str();
    unsigned substring = set_path.find_last_of("/");
    std::string set = set_path.substr(substring);

    std::string file = file_name->data.c_str();
    std::cout << "received: " << set << "- " << file << std::endl;
    if (detection_msg->head_detections.size() == 1 && detection_msg->head_detections[0].face_detections.size() ==1)
    {
    	//color img
    	sensor_msgs::ImageConstPtr imgPtr = boost::shared_ptr<sensor_msgs::Image const>(&(head_msg->head_detections[0].color_image), voidDeleter);
		try
		{
			cv_ptr = cv_bridge::toCvShare(imgPtr, sensor_msgs::image_encodings::RGB8);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		cv::Mat img = cv_ptr->image;

		//depth
		sensor_msgs::ImageConstPtr msgPtr = boost::shared_ptr<sensor_msgs::Image const>(&(head_msg->head_detections[0].depth_image), voidDeleter);
		try
		{
			cv_ptr = cv_bridge::toCvShare(msgPtr, sensor_msgs::image_encodings::TYPE_32FC3);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		cv::Mat xyz = cv_ptr->image;

		if (!boost::filesystem::exists (rgbd_db_head_directory_)) boost::filesystem::create_directory(rgbd_db_head_directory_);
		std::ostringstream depth_p,img_p;
		std::string dir, depth_path, img_path;
		dir = rgbd_db_head_directory_;
		dir = dir+set;
		if (!boost::filesystem::exists (dir.c_str()))
			{
				if(boost::filesystem::create_directory(dir.c_str()))
					{
					std::cout << "created new folder for set: " << dir << std::endl;
					}
			}
		depth_p<<rgbd_db_head_directory_<<set<<"/"<<file<<"_d.xml";
		img_p<<rgbd_db_head_directory_<<set<<"/"<<file<<"_c.bmp";
		img_path=img_p.str();
		depth_path=depth_p.str();
		//std::cout << "writing to: " << depth_path << "\n" << img_path << "\n";

		cv::imwrite(img_path,img);
		cv::FileStorage fs(depth_path, cv::FileStorage::WRITE);
		fs << "depthmap" << xyz;
		fs.release();
		saved_++;
    }
    else
	{
    	faulty_det_++;
    	std::cout << "bad detection for " << set << "-" << file_name<<std::endl;
    	std::cout << "heads: " << detection_msg->head_detections.size();
    	if (detection_msg->head_detections.size()>0) std::cout << ", faces: " << detection_msg->head_detections[0].face_detections.size();
    	std::cout<< std::endl;
	}
    std::cout << "received: " << received_ << "\nAccepted: " << saved_ << "\nBad Detection: " << faulty_det_ << std::endl;

}

bool RgbdDbHeadIsolator::saveImages(cv::Mat& image, cv::Mat& xyz)
{
return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");

    ros::NodeHandle n;

    RgbdDbHeadIsolator head_isolator(n);

    ros::spin();

    return 0;
}
