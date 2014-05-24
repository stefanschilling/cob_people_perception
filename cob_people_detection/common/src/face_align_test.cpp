#include"cob_people_detection/face_normalizer.h"
#include<iostream>
#include<opencv/cv.h>
#include<opencv/highgui.h>
//
//int main(int argc, const char *argv[])
//{
//	//std::cout<<"[FaceNormalizer] reading trained data for label "<<argn<<"...\n";
//	FaceNormalizer::FNConfig cfg;
//	cfg.eq_ill=false;
//	cfg.align=true;
//	cfg.resize=true;
//	cfg.cvt2gray=false;
//	cfg.extreme_illumination_condtions=false;
//
//	std::string  class_path="/opt/ros/groovy/share/OpenCV/";
//
//	FaceNormalizer fn;
//	fn.init(class_path,cfg);
//	cv::Mat depth,img,xyz;
//	//else      i_path="/share/goa-tz/people_detection/eval/Kinect3DSelect/";
//	// i_path="/share/goa-tz/people_detection/eval/KinectIPA/";
//	// std::string training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
//	std::string training_path = "/home/rmb-ss/.ros/cob_people_detection/files/training_data/";
//	std::string tdata_path = training_path + "tdata.xml";
//
//	cv::FileStorage fileStorage(tdata_path, cv::FileStorage::READ);
//	if (!fileStorage.isOpened())
//	{
//		std::cout << "Error: load Training Data: Can't open " << tdata_path << ".\n" << std::endl;
//	}
//
//	// search tdata for scenes associated to label, store scene numbers in vector
//	std::vector<int> scene_numbers;
//	std::vector<float> scene_scores;
//	std::string label, src_label;
//	std::cout << "Enter label to work with: \n";
//	std::cin >> src_label;
//
//	int number_entries = (int)fileStorage["number_entries"];
//	scene_numbers.clear();
//	for(int i=0; i<number_entries; i++)
//	{
//		// labels
//		std::ostringstream tag_label;
//		tag_label << "label_" << i;
//		//std::cout<<"tag label: " << tag_label << " done\n";
//		label = (std::string)fileStorage[tag_label.str().c_str()];
//		//std::cout<<"label: " << label << std::endl;
//		if(label == src_label) scene_numbers.push_back(i);
//	}
//	fileStorage.release();
//
//	if (scene_numbers.size()==0)
//	{
//		std::cout << "No entries of that label in tdata.xml\n";
//		return 0;
//	}
//
//	// read scenes associated with label, rate for seeding viability
//	for(int i=0; i<scene_numbers.size(); i++)
//	{
//		//if (i==10)
//		//{
//			fn.read_scene_from_training(xyz,img,training_path, scene_numbers.at(i));
//			fn.frontFaceImage(img,xyz,scene_scores);
//		//}
//	}
//
//	// find best seed to created images from
//	int best_seed_id=-1;
//	float best_score =10;
//	for(int i=0; i<scene_scores.size();i++)
//	{
//		if (scene_scores[i]<best_score)
//		{
//			best_seed_id=i;
//			best_score=scene_scores[i];
//		}
//	}
//	//in case of data corruption, set ID of valid data manually
//	//best_seed_id = 10;
//
//	//std::cout<<"Best source image ID: " << best_seed_id << " Score: " << scene_scores[best_seed_id] << std::endl;
//	scene_numbers.clear();
//	scene_scores.clear();
//
//	// create synth images from source
//	cv::Size norm_size=cv::Size(100,100);
//	std::vector<cv::Mat> synth_images;
//	std::vector<cv::Mat> synth_depths;
//
//	//load source image, create training data
//	fn.read_scene_from_training(xyz,img,training_path, best_seed_id);
//	fn.synthFace(img,xyz,norm_size,synth_images,synth_depths,training_path, label);
//
//	//cv::FileStorage fileStorage(tdata_path, cv::FileStorage::WRITE);
//
//	// call member functions of FaceNormalizer
//
//	//fn.isolateFace(wmat1,xyz);
//	//fn.normalizeFace(wmat1,xyz,norm_size,depth);
//	//fn.recordFace(synth_images[4],synth_depths[4]);
//
//	//synth_images.push_back(wmat1);
//	// depth.convertTo(depth,CV_8UC1,255);
//	// cv::equalizeHist(depth,depth);
//
//	return 0;
//}
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include<sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>


//#include </opt/ros/groovy/include/pcl-1.6/pcl/point_types.h>
//#include </opt/ros/groovy/include/pcl-1.6/pcl/pcl_macros.h>

//#include <pcl/common/common.h>
//#include <pcl/common/eigen.h>


typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class scene_publisher
{
	public:

	// Constructor
	scene_publisher()
	{
		this->persp=1;
		this->shot=0;

		//file="-1";
		//n_.param("/cob_people_detection/face_recognizer/file",file,file);
		//std::cout<<"input file: "<<file<<std::endl;

		scene_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/camera/depth_registered/points",1);
		img_pub_ = nh.advertise<sensor_msgs::Image>("/camera/rgb/image_color",1);

	}

	// Destructor
	~scene_publisher()
	{
	}


	void process()
	{
		pc.clear();
		path = "/home/stefan/rgbd_db/";
		std::stringstream xyz_stream,jpg_stream,depth_stream;
		xyz_stream<<path.c_str()<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_d.xml";
		jpg_stream<<path.c_str()<<"set_01/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_c.bmp";
		depth_stream<<path.c_str()<<"set_01/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_d.xml";

		std::cout<<xyz_stream.str()<<"\n "<<jpg_stream.str()<<"\n "<<depth_stream.str()<<std::endl;

		//read bmp image
		cv::Mat img;
		img=cv::imread(jpg_stream.str().c_str());
		//img=cv::imread("/home/stefan/rgbd_db/007_2_c.bmp");
		img.convertTo(img,CV_8UC3);
		cv::resize(img,img,cv::Size(640,480));
		std::cout << "img gelesen... size: " << img.size() << " rows: " << img.rows << " cols: " << img.cols << std::endl;

		//read xyz or depth xml
		cv::Mat xyz,dm;
		cv::FileStorage fs_read(xyz_stream.str().c_str(),cv::FileStorage::READ);
		cv::FileStorage fs_read2(depth_stream.str().c_str(),cv::FileStorage::READ);
		//cv::FileStorage fs_read("/home/stefan/rgbd_db/007_2_d.xml",cv::FileStorage::READ);
		fs_read2["depth"]>> dm;
		fs_read2.release();
		fs_read["depth"]>> xyz;
		fs_read.release();
		std::cout << "xyz gelesen... size: " << xyz.size() << " rows: " << xyz.rows << " cols: " << xyz.cols << std::endl;

		//set parameters
		pc.width=640;
		pc.height=480;
		cv::Mat cam_mat =(cv::Mat_<double>(3,3)<< 524.90160178307269,0.0,320.13543361773458,0.0,525.85226379335393,240.73474482242005,0.0,0.0,1.0);
		// compensate for kinect offset
		int index=0;
		cv::Vec3f rvec,tvec;
		cv::Mat dist_coeff;

		tvec=cv::Vec3f(0.03,0.0,0.0);
		rvec=cv::Vec3f(0.0,0.0,0);
		dist_coeff=cv::Mat::zeros(1,5,CV_32FC1);
		cv::Mat pc_trans=cv::Mat::zeros(640,480,CV_64FC1);
		//convert depth to xyz
		pcl::PointXYZRGB pt;
		for(int r=0;r<dm.rows;r++)
		{
			for(int c=0;c<dm.cols;c++)
			{
				//pt  << c,r,1.0/525;
				//pt=cam_mat_inv*pt;

				cv::Point3f pt_3f;
				pt_3f.x=(c-320);
				pt_3f.y=(r-240);
				pt_3f.z=525;

				double nrm=cv::norm(pt_3f);
				pt_3f.x/=nrm;
				pt_3f.y/=nrm;
				pt_3f.z/=nrm;

				pt_3f=pt_3f*dm.at<float>(r,c);
				std::vector<cv::Point3f> pt3_vec;
				pt3_vec.push_back(pt_3f);
				/*
				pt.x=pt_3f.x;
				pt.y=pt_3f.y;
				pt.z=pt_3f.z;
				uint32_t rgb = (static_cast<uint32_t>(img.at<cv::Vec3b>(r,c)[0]) << 0 |static_cast<uint32_t>(img.at<cv::Vec3b>(r,c)[1]) << 8 | static_cast<uint32_t>(img.at<cv::Vec3b>(r,c)[2]) << 16);
				pt.rgb = *reinterpret_cast<float*>(&rgb);
				pc.points.push_back(pt);
				*/
				std::vector<cv::Point2f> pt2_vec;

				cv::projectPoints(pt3_vec,rvec,tvec,cam_mat,dist_coeff,pt2_vec);

				int x_t,y_t;
				x_t=round(pt2_vec[0].x);
				y_t=round(pt2_vec[0].y);
				if(x_t<0 ||x_t>640 ||y_t<0 ||y_t>480) continue;
				pc_trans.at<float>(y_t,x_t)=dm.at<float>(r,c);
			}
		}
		std::cout << "start iterating to add points..." << std::endl;
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

				pt_=pt_*pc_trans.at<float>(j,i);

				pt.x = pt_.x;
				pt.y = pt_.y;
				pt.z = pt_.z;
				//end transform

				pt.x = xyz.at<cv::Vec3f>(j,i)[0];
				pt.y = xyz.at<cv::Vec3f>(j,i)[1];
				pt.z = xyz.at<cv::Vec3f>(j,i)[2];
				if (i<619&&j<469)
				{
					//uint32_t rgb = (static_cast<uint32_t>(img.at<cv::Vec3b>(j+10,i+15)[0]) << 0 |static_cast<uint32_t>(img.at<cv::Vec3b>(j+10,i+15)[1]) << 8 | static_cast<uint32_t>(img.at<cv::Vec3b>(j+10,i+15)[2]) << 16);
					uint32_t rgb = (static_cast<uint32_t>(img.at<cv::Vec3b>(j,i+18)[0]) << 0 |static_cast<uint32_t>(img.at<cv::Vec3b>(j,i+18)[1]) << 8 | static_cast<uint32_t>(img.at<cv::Vec3b>(j,i+18)[2]) << 16);

					pt.rgb = *reinterpret_cast<float*>(&rgb);
				}
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

		std::cout << "done with point cloud: " << pc.size() << std::endl;
	}

	void publish()
	{
		out_pc2.header.frame_id="/camera/";
		out_pc2.header.stamp = ros::Time::now();
		scene_pub_.publish(out_pc2);

		out_img.header.frame_id="/camera/";
		out_img.header.stamp = ros::Time::now();
		img_pub_.publish(out_img);
	}

	ros::NodeHandle nh;
	int persp;
	int shot;

	protected:
		ros::Publisher scene_pub_;
		ros::Publisher img_pub_;
		sensor_msgs::PointCloud2 out_pc2;
		pcl::PointCloud<pcl::PointXYZRGB> pc;
		sensor_msgs::Image out_img;
		std::string file,path;
};



int main (int argc, char** argv)
{
	ros::init (argc, argv, "scene_publisher");
	int persp_filter=17;

	scene_publisher sp;

	ros::Rate loop_rate(1);
	while (ros::ok())
	{
		sp.shot++;
		std::cout<<"PERSPECTIVE:"<<sp.persp<<std::endl;
		if(sp.shot==4)
		{
			sp.shot=1;
			sp.persp++;
		}
		if(sp.persp==18) break;

		if(sp.persp<=persp_filter)
		{
			sp.process();
			sp.publish();
			ros::spinOnce ();
			loop_rate.sleep();
		}
	}
}
