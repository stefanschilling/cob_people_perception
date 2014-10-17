#ifdef __LINUX__
	#include "cob_people_detection/rgbd_db_scene_publisher.h"
	#include "cob_vision_utils/GlobalDefines.h"
#else
#endif

using namespace ipa_PeopleDetector;

RgbdDbScenePublisher::RgbdDbScenePublisher(ros::NodeHandle nh)
: node_handle_(nh)
{
	//get parameters for this run
	XmlRpc::XmlRpcValue perspectives_list, sets_list;
	if(!node_handle_.getParam("/cob_people_detection/rgbd_db_directory", path_to_db_)) std::cout<<"PARAM NOT AVAILABLE"<<std::endl;
	std::cout << "rgbd_db_directory = " << path_to_db_ << "\n";
	node_handle_.getParam("sets_list", sets_list);
	std::cout << "sets_list = " << sets_list << "\n";
	node_handle_.getParam("perspectives_list",perspectives_list);
	std::cout<< "perspectives_list: "<<perspectives_list<<"\n";

	scene_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/camera/depth_registered/points",1);
	img_pub_ = nh.advertise<sensor_msgs::Image>("/camera/rgb/image_color",1);
	set_pub_ = nh.advertise<cob_people_detection_msgs::stamped_string>("set_path",1);
	file_name_pub_ = nh.advertise<cob_people_detection_msgs::stamped_string>("file_name",1);


	if (sets_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
	{
		sets_.resize(sets_list.size());
		//std::cout << "copying sets_list elements to sets: ";
		for (int i = 0; i < sets_list.size(); i++)
		{
			ROS_ASSERT(sets_list[i].getType() == XmlRpc::XmlRpcValue::TypeString);
			sets_[i] = static_cast<std::string>(sets_list[i]);
			//std::cout << i << " ";
		}
	}
	//std::cout << "\nsets_list -> sets complete" << std::endl;
	if (perspectives_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
	{
		perspectives_.resize(perspectives_list.size());
		//std::cout << "copying perspectives_list elements to perspectives: ";
		for (int i = 0; i < perspectives_list.size(); i++)
		{
			ROS_ASSERT(perspectives_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt);
			perspectives_[i] = static_cast<int>(perspectives_list[i]);
			//std::cout << i << " ";
		}
	}
}

// Destructor
RgbdDbScenePublisher::~RgbdDbScenePublisher()
{
}

void RgbdDbScenePublisher::PublishScenes()
{
	std::stringstream bmp_stream,xyz_stream;
	cv::Mat bmp,xyz;
    //iterate through requested sets in requested perspectives, through all shots (maximum of 3 shots per perspective in database)
	for (int set=0; set < sets_.size(); set++)
	{
		//build path to set
		std::string path;
		path.append(path_to_db_);
		path.append(sets_[set]);
		file_=sets_[set];
		std::cout << path << std::endl;
        for (int persp=0; persp < perspectives_.size(); persp++)
		{
        	persp_=perspectives_[persp];
			for (int shot=1; shot < 4; shot++)
			{
				shot_=shot;
				// add perspective and shot to complete path
				bmp_stream.str(std::string());
				xyz_stream.str(std::string());
				bmp_stream << path.c_str() << "/" << std::setw(3)<<std::setfill('0')<<perspectives_[persp]<<"_"<<shot;
				xyz_stream << bmp_stream.rdbuf();
				bmp_stream << "_c.bmp";
				xyz_stream << "_d.xml";
				// check if files exists
				std::ifstream bmp_test(bmp_stream.str().c_str());
				std::ifstream xyz_test(xyz_stream.str().c_str());
				unsigned found = bmp_stream.str().find_last_of("/\\");
				if (bmp_test.good()&& xyz_test.good())
				{
					//create pcl
					CreatePcl(bmp_stream, xyz_stream);
					//publish pcl
					Publish();
				}
			}
		}
	}


}

void RgbdDbScenePublisher::CreatePcl(std::stringstream& bmp_stream, std::stringstream& xyz_stream)
{
	pc.clear();
//	path_to_db_ = "/home/stefan/rgbd_db/";

	std::stringstream jpg_stream,depth_stream, set_path_stream, file_name_stream;

	set_path_stream <<path_to_db_.c_str()<<file_.c_str();
	set_path.data = set_path_stream.str();
	file_name_stream << std::setw(3)<<std::setfill('0')<<persp_<<"_"<<shot_;
	file_name.data = file_name_stream.str();

	cv::Mat img,dm;

	//read bmp image
	img=cv::imread(bmp_stream.str().c_str());
	//img=cv::imread("/home/stefan/rgbd_db/007_2_c.bmp");
	img.convertTo(img,CV_8UC3);
	cv::resize(img,img,cv::Size(640,480));
	//std::cout << "img read... size: " << img.size() << " rows: " << img.rows << " cols: " << img.cols << std::endl;

	//read depth xml
	cv::FileStorage fs_read2(xyz_stream.str().c_str(),cv::FileStorage::READ);
	fs_read2["depth"]>> dm;
	fs_read2.release();
	//std::cout << "depthmap read... size: " << dm.size() << " rows: " << dm.rows << " cols: " << dm.cols << std::endl;

	//set parameters
	pc.width=640;
	pc.height=480;
	//cam mat found in scene publisher
	cv::Mat cam_mat =(cv::Mat_<double>(3,3)<< 524.90160178307269,0.0,320.13543361773458,0.0,525.85226379335393,240.73474482242005,0.0,0.0,1.0);
	//I don't know how they were determined. AAU was unable to locate the kinect used for this, so could not help solve the issue.
	//However, these values are very close to those found for the kinect rgb camera at nicolas.burrus.name/index.php/Research/KinectCalibration, so they should give good results if the deviation between devices is not great.

	// compensate for kinect offset
	cv::Vec3f rvec,tvec;
	cv::Mat dist_coeff;

	tvec=cv::Vec3f(0.03,0.0,0.0);
	rvec=cv::Vec3f(0.0,0.0,0);
	dist_coeff=cv::Mat::zeros(1,5,CV_32FC1);
	cv::Mat pc_trans=cv::Mat::zeros(640,480,CV_64FC1);

	//convert depth to xyz
	pcl::PointXYZRGB pt;
	int failctr=0;
	for(int r=0;r<dm.rows;r++)
	{
		for(int c=0;c<dm.cols;c++)
		{
			//estimate of point
			cv::Point3f pt_3f;
			pt_3f.x=(c-320);
			pt_3f.y=(r-240);
			pt_3f.z=525;

			double nrm=cv::norm(pt_3f);
			pt_3f.x/=nrm;
			pt_3f.y/=nrm;
			pt_3f.z/=nrm;

			pt_3f=pt_3f*dm.at<float>(r,c);
			//pt_3f=pt_3f*dm.at<double>(r,c);
			std::vector<cv::Point3f> pt3_vec;

			pt3_vec.push_back(pt_3f);
			std::vector<cv::Point2f> pt2_vec;

			cv::projectPoints(pt3_vec,rvec,tvec,cam_mat,dist_coeff,pt2_vec);

			int x_t,y_t;
			x_t=round(pt2_vec[0].x);
			y_t=round(pt2_vec[0].y);
			if(x_t<0 ||x_t>640 ||y_t<0 ||y_t>480)
				{
				failctr++;
				continue;
				}
			//pc_trans.at<double>(y_t,x_t)=dm.at<double>(r,c);
			pc_trans.at<float>(y_t,x_t)=dm.at<float>(r,c);

		}
	}

	//std::cout << "pc_trans created... points dropped: " << failctr << ", size: " << pc_trans.size() << " rows: " << pc_trans.rows << " cols: " << pc_trans.cols << std::endl;
	//std::cout << "start iterating to add points..." << std::endl;
	for(int j =0; j< dm.rows;j++)
	{
		for(int i =0; i< dm.cols; i++)
		{
			//create points from new depthmap pc_trans
			cv::Point3f pt_;
			pt_.x=(i-320);
			pt_.y=(j-240);
			pt_.z=525;

			double nrm=cv::norm(pt_);
			pt_.x/=nrm;
			pt_.y/=nrm;
			pt_.z/=nrm;

			//pt_=pt_*pc_trans.at<float>(j,i);
			pt_=pt_*pc_trans.at<float>(j,i);

			pt.x = pt_.x;
			pt.y = pt_.y;
			pt.z = pt_.z;

			//add color to points, not shifted
			uint32_t rgb = (static_cast<uint32_t>(img.at<cv::Vec3b>(j,i)[0]) << 0 |static_cast<uint32_t>(img.at<cv::Vec3b>(j,i)[1]) << 8 | static_cast<uint32_t>(img.at<cv::Vec3b>(j,i)[2]) << 16);
			pt.rgb = *reinterpret_cast<float*>(&rgb);
			pc.points.push_back(pt);
		}
	}
	cv_bridge::CvImage cv_ptr;
	cv_ptr.image = img;
	cv_ptr.encoding = sensor_msgs::image_encodings::BGR8;
	out_img = *(cv_ptr.toImageMsg());
	pcl::toROSMsg(pc,out_pc2);
}

void RgbdDbScenePublisher::Publish()
{
	out_pc2.header.frame_id="/camera/";
	out_pc2.header.stamp = ros::Time::now();
	scene_pub_.publish(out_pc2);

	out_img.header.frame_id="/camera/";
	out_img.header.stamp = ros::Time::now();
	img_pub_.publish(out_img);

	set_path.header.stamp = ros::Time::now();
	set_pub_.publish(set_path);

	file_name.header.stamp =ros::Time::now();
	file_name_pub_.publish(file_name);
	total_published_msgs_++;
}


int main (int argc, char** argv)
{
	// Initialize ROS, specify name of node
	ros::init(argc, argv, "rgbd_db_scene_publisher");

	// Create a handle for this node, initialize node
	ros::NodeHandle nh;

	// Create RgbdDbRecognitionTest class instance, loading parameters from .yaml
	RgbdDbScenePublisher rgbd_db_scene_publisher(nh);

	// Initiate Test
	rgbd_db_scene_publisher.PublishScenes();

	ros::Rate loop_rate(0.5);

	ros::spin();

	return 0;
}
