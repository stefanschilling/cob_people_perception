#ifdef __LINUX__
	#include "cob_people_detection/rgbd_db_manual_publisher.h"
	#include "cob_vision_utils/GlobalDefines.h"

	// getch() include copied from people_detection_action_client
	#include <termios.h>

    int getch();

    int getch()
    {
       static int ch = -1, fd = 0;
       struct termios neu, alt;
       fd = fileno(stdin);
       tcgetattr(fd, &alt);
       neu = alt;
       neu.c_lflag &= ~(ICANON|ECHO);
       tcsetattr(fd, TCSANOW, &neu);
       ch = getchar();
       tcsetattr(fd, TCSANOW, &alt);
       return ch;
    }
#else
#endif


using namespace ipa_PeopleDetector;

RgbdDbManualPublisher::RgbdDbManualPublisher(ros::NodeHandle nh)
: node_handle_(nh)
{
	//get parameters for this run
	XmlRpc::XmlRpcValue perspectives_list, sets_list;
	if(!node_handle_.getParam("/cob_people_detection/rgbd_db_directory", path_to_db_)) std::cout<<"PARAM NOT AVAILABLE"<<std::endl;
	std::cout << "rgbd_db_directory = " << path_to_db_ << "\n";

	scene_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/camera/depth_registered/points",1);
	img_pub_ = nh.advertise<sensor_msgs::Image>("/camera/rgb/image_color",1);
	set_pub_ = nh.advertise<cob_people_detection_msgs::stamped_string>("set_path",1);
	file_name_pub_ = nh.advertise<cob_people_detection_msgs::stamped_string>("file_name",1);

	this->shot_=1;
	this->persp_=1;
	this->set_=2;

	//get available Sets
	dirent* ent;
	DIR* dir;
	errno = 0;
	unsigned char isFolder = 0x4;
	dir = opendir( path_to_db_.empty() ? "." : path_to_db_.c_str() );
	if (dir)
	{
		while (true)
		{
			errno=0;
			ent = readdir(dir);
			if (ent == NULL) break;
			if (ent->d_type == isFolder) sets_.push_back( std::string ( ent->d_name));
		}
		closedir(dir);
		std::sort(sets_.begin(), sets_.end());
	}
	for (int i =0; i<sets_.size(); i++) std::cout<< sets_[i] << std::endl;
}

// Destructor
RgbdDbManualPublisher::~RgbdDbManualPublisher()
{
}

void RgbdDbManualPublisher::createPcl()
{
	pc.clear();

	std::cout << "Ready Messages for " << sets_[set_] << " - " << persp_ << " - " << shot_ <<std::endl;
	std::stringstream set_path_stream, file_name_stream;

	set_path_stream <<path_to_db_.c_str()<<sets_[set_];
	set_path.data = set_path_stream.str();
	file_name_stream << std::setw(3)<<std::setfill('0')<<persp_<<"_"<<shot_;
	file_name.data = file_name_stream.str();

	cv::Mat img,dm;

	//read bmp image
	img=cv::imread(bmp_stream_.str().c_str());
	//img=cv::imread("/home/stefan/rgbd_db/007_2_c.bmp");
	img.convertTo(img,CV_8UC3);
	cv::resize(img,img,cv::Size(640,480));
	//std::cout << "img read... size: " << img.size() << " rows: " << img.rows << " cols: " << img.cols << std::endl;

	//read depth xml
	cv::FileStorage fs_read2(xyz_stream_.str().c_str(),cv::FileStorage::READ);
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

// check if files for requested set, perspective and shot exist
bool RgbdDbManualPublisher::checkFile()
{
	bmp_stream_.str("");
	xyz_stream_.str("");
	// add requested perspective and shot to complete path, check i files exist
	std::string path;
	path.append(path_to_db_);
	path.append(sets_[set_]);
	bmp_stream_ << path.c_str() << "/" << std::setw(3)<<std::setfill('0')<<persp_<<"_"<<shot_;
	xyz_stream_ << bmp_stream_.rdbuf();

	bmp_stream_ << "_c.bmp";
	xyz_stream_ << "_d.xml";

	// check if files exists
	std::ifstream bmp_test(bmp_stream_.str().c_str());
	std::ifstream xyz_test(xyz_stream_.str().c_str());
	unsigned found = bmp_stream_.str().find_last_of("/\\");
	if (bmp_test.good()&& xyz_test.good()) return true;
	else return false;
}

bool RgbdDbManualPublisher::changeSet(int incr)
{
	bool valid = false;
	// offset counter by 2, to skip '.' and '..'
	for (int i=2; i< sets_.size(); i++)
	{
		set_+= incr;
		if (set_ > sets_.size()-1) set_=2;
		else if (set_ < 2) set_ = sets_.size()-1;

		if (checkFile()) return true;
		else if (changePersp(incr)) return true;
	}
	std::cout << "No valid sets, perspectives and shots found." << std::endl;
	return false;
}

bool RgbdDbManualPublisher::changePersp(int incr)
{
	bool valid = false;
	for (int i=0; i<13; i++)
	{
		persp_+=incr;
		if (persp_<1) persp_ = 13;
		else if (persp_ > 13) persp_= 1;

		if( checkFile()) return true;
		else if (changeShot(incr)) return true;
	}
	std::cout << "failed to find valid perspective for requested set after iterating through all shots: " << sets_[set_] << std::endl;
	return false;
}

bool RgbdDbManualPublisher::changeShot(int incr)
{
	bool valid = false;
	for (int i=0; i<3; i++)
	{
		shot_+=incr;
		if (shot_<1) shot_=3;
		else if (shot_>3) shot_=1;

		if (checkFile()) return true;
	}
	std::cout << "failed to find valid shot for current set and perspective: " << sets_[set_] << ", " << persp_ << std::endl;
	return false;
}


void RgbdDbManualPublisher::publish()
{
	std::cout << "Publish: " << set_path.data << ", file: " << file_name.data << std::endl;
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
}

void RgbdDbManualPublisher::keyLoop()
{
	char key = 'q';
	bool rdy=false;
	bool valid=false;
	int persp, shot;
	std::string set;
	std::cout << "\n7, 9: change set - 4, 6: previous - next perspective - 1, 3: prev - next shot \n5: Create Message - 2: publish \nq - Quit\n";
	do
	{
		std::cout << "Selection: " << sets_[set_] << " " << persp_ << " " << shot_;
		if (rdy) std::cout << " | Ready to publish: " << set << " " << persp << " " << shot;
		std::cout << "\n";
		key = getch();
		if (key == '7') valid=changeSet(-1);
		else if (key == '9') valid=changeSet(+1);
		else if (key == '4') valid=changePersp(-1);
		else if (key == '6') valid=changePersp(+1);
		else if (key == '1') valid=changeShot(-1);
		else if (key == '3') valid=changeShot(+1);
		else if (key == '5')
		{
			if (valid)
			{
				createPcl();
				set = sets_[set_];
				shot = shot_;
				persp = persp_;
				rdy=true;
			}
		}
		else if (key == '2') if (rdy) publish();

	}while (key != 'q');
	ros::shutdown();
}

int main (int argc, char** argv)
{
	// Initialize ROS, specify name of node
	ros::init(argc, argv, "rgbd_db_manual_publisher");

	// Create a handle for this node, initialize node
	ros::NodeHandle nh;

	// Create RgbdDbRecognitionTest class instance, loading parameters from .yaml
	RgbdDbManualPublisher rgbd_db_manual_publisher(nh);

	// Initiate Test
	rgbd_db_manual_publisher.keyLoop();

	ros::spin();

	return 0;
}
