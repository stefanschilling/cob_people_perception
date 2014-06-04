#include"cob_people_detection/face_normalizer.h"
#include<iostream>
#include<opencv/cv.h>
#include<opencv/highgui.h>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/thread/mutex.hpp>
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem.hpp"

bool create_file_lists(std::string& path, std::vector<std::string>& xmls,std::vector<std::string>& bmps)
{
	boost::filesystem::path boost_path;
	boost_path=path;
	if (!exists(boost_path))
	{
		std::cout<<"bad path, doesn't exist!\n";
		return false;
	}
	boost::filesystem::directory_iterator end_itr;
	for(boost::filesystem::directory_iterator itr(boost_path); itr!=end_itr;++itr)
	{
		if (boost::filesystem::is_regular_file(itr->status()))
		{
			if (itr->path().extension()==".xml")
			{
				xmls.push_back(itr->path().c_str());
			}
//			if (itr->path().extension()==".bmp")
//			{
//				bmps.push_back(itr->path().c_str());
//			}
		}
	}
	if (xmls.size()>0)
	{
		return true;
	}
	else return false;
}

void rewrite_tdata(std::string& training_path, std::string& added_label, int& img_count)
{
//	 TODO
//	read out tdata, re-write xml with newly added content.

	cv::FileStorage fileStorageRead(training_path + "tdata2.xml", cv::FileStorage::READ);
	if (!fileStorageRead.isOpened())
	{
		std::cout << "Error: load Training Data: Can't open " << training_path+"tdata.xml" << ".\n" << std::endl;
	}

	// labels
	std::vector<std::string> face_labels, face_images, face_depths;
	face_labels.clear();
	face_images.clear();
	face_depths.clear();
	int old_number_entries = (int)fileStorageRead["number_entries"];
	// save current entries
	for(int i=0; i<old_number_entries; i++)
	{
		// labels
		std::ostringstream tag_label1, tag_label2, tag_label3;
		tag_label1 << "label_" << i;
		std::string label = (std::string)fileStorageRead[tag_label1.str().c_str()];
		face_labels.push_back(label);
		tag_label2 << "image_" << i;
		std::string image = (std::string)fileStorageRead[tag_label2.str().c_str()];
		face_images.push_back(image);
		tag_label3 << "depthmap_" << i;
		std::string depth = (std::string)fileStorageRead[tag_label3.str().c_str()];
		face_depths.push_back(depth);
	}

	//create tdata2.xml with newly added images.
	cv::FileStorage fileStorageWrite(training_path+"tdata2.xml", cv::FileStorage::WRITE);
	if (fileStorageWrite.isOpened())
	{
		fileStorageWrite << "number_entries" << (int)img_count;
		for(int i=0; i<img_count; i++)
		{
			std::ostringstream tag, tag2, tag3;
			std::ostringstream shortname_img, shortname_depth;
			if (i<old_number_entries)
			{
				// face label
				tag << "label_" << i;
				fileStorageWrite << tag.str().c_str() << face_labels[i].c_str();

				// face images
				shortname_img << "img/" << i << ".bmp";
				tag2 << "image_" << i;
				fileStorageWrite << tag2.str().c_str() << face_images[i].c_str();

				// depth images
				shortname_depth << "depth/" << i << ".xml";
				tag3 << "depthmap_" << i;
				fileStorageWrite << tag3.str().c_str() << face_depths[i].c_str();

			}
			else if (i>= old_number_entries)
			{
				// labels
				//std::cout << "writing new entry number " << i << " to tdata2.xml\n";
				tag << "label_" << i;
				fileStorageWrite << tag.str().c_str() << added_label;

				// face images
				shortname_img << "synth_img/" << i << ".bmp";
				tag2 << "image_" << i;
				fileStorageWrite << tag2.str().c_str() << shortname_img.str().c_str();

				shortname_depth << "synth_depth/" << i << ".xml";
				tag3 << "depthmap_" << i;
				fileStorageWrite << tag3.str().c_str() << shortname_depth.str().c_str();
			}
		}
		fileStorageWrite.release();
		face_labels.clear();
		face_images.clear();
		face_depths.clear();
	}
}

int main(int argc, const char *argv[])
{
	std::stringstream xyz_stream;
	std::string src_set, src_persp;
	std::cout << "Enter set (set_01-31): \n";
	std::cin >> src_set;
	std::cout << "Enter perspective to use(1-13): ";
	std::cin >> src_persp;
	std::string path = "/home/stefan/rgbd_db_heads/"+src_set;
	std::vector<std::string> xmls,bmps;
	create_file_lists(path, xmls, bmps);
	std::cout << xmls.size() << std::endl;
	std::cout << bmps.size() << std::endl;

	//std::cout<<"[FaceNormalizer] reading trained data for label "<<argn<<"...\n";
	FaceNormalizer::FNConfig cfg;
	cfg.eq_ill=false;
	cfg.align=true;
	cfg.resize=true;
	cfg.cvt2gray=false;
	cfg.extreme_illumination_condtions=false;

	std::string  class_path="/opt/ros/groovy/share/OpenCV/";

	FaceNormalizer fn;
	fn.init(class_path,cfg);
	cv::Mat depth, img;
	cv::Size norm_size=cv::Size(100,100);
	std::string bmp_path,training_path, label;
	label = src_set.append("_synth");
	training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";

	// read number of entries from tdata
	cv::FileStorage fileStorageRead(training_path + "tdata2.xml", cv::FileStorage::READ);
	if (!fileStorageRead.isOpened())
	{
		std::cout << "Error: load Training Data: Can't open " << training_path+"tdata.xml" << ".\n" << std::endl;
		return 0;
	}
	int img_count_current = (int)fileStorageRead["number_entries"];
	int img_count = img_count_current;


	xyz_stream<<std::setw(3)<<std::setfill('0')<<src_persp;
	for (int i=1;i<xmls.size();i++)
	{
		//std::cout << xmls[i] << std::endl;
		unsigned perspective = xmls[i].find_last_of("/\\");
		std::string substring = xmls[i].substr(perspective+1, 3);
		if(xyz_stream.str() == substring)
		{
			//std::cout << bmps[i] << std::endl;
			bmp_path = xmls[i];
			bmp_path.erase(xmls[i].length()-5,5);
			bmp_path.append("c.bmp");
			//std::cout << bmp_path << std::endl;
			cv::FileStorage fs(xmls[i], cv::FileStorage::READ);
			fs["depthmap"]>> depth;
			img = cv::imread(bmp_path, CV_LOAD_IMAGE_COLOR);
			std::cout << "start synth for bmp and xml: " << img.size() << ", " << depth.size() << std::endl;
			if(fn.synthFace(img,depth,norm_size,training_path,img_count))
			{
				std::cout << "images added from source image " << bmp_path << ": " << img_count-img_count_current <<std::endl;
			}
			fs.release();
		}
	}
	xmls.clear();
	bmps.clear();
	rewrite_tdata(training_path, label, img_count);
	std::cout << "synth succesful, new image total: " << img_count << std::endl;

//	std::string training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
//	// std::string training_path = "/home/rmb-ss/.ros/cob_people_detection/files/training_data/";
//	std::string src_path = "/home/stefan/rgbd_db/";
//	std::string tdata_path = training_path + "tdata.xml";
//
//	//quality rating and picking best quality front-shot from training data
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
//		fn.read_scene_from_training(xyz,img,training_path, scene_numbers.at(i));
//		fn.frontFaceImage(img,xyz,scene_scores);
//
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

//	// create synth images from source
//	// norm_size for training data: 100x100
//	cv::Size norm_size=cv::Size(100,100);
//
//	//load source image, create training data
//	//int perspective;
//	std::string src_set, src_persp, src_label, shot_no, depth_path, img_path;
//	//perspective = 7;
//	std::cout << "Enter set to work with: \n";
//	std::cin >> src_set;
//	std::cout << "Enter perspective to work with: \n";
//	std::cin >> src_persp;
//	src_label = "set_"+src_set;
//	std::cout << "Enter shot number \n";
//	std::cin >> shot_no;
//
//	img_path = src_path + src_label + "/0"+ src_persp +"_" + shot_no +"_c.bmp";
//	depth_path = src_path + src_label + "/0"+ src_persp +"_" + shot_no +"_d.xml";
//	std::cout <<  img_path << std::endl << depth_path <<std::endl;
//
//	//read scene from disk
//	cv::FileStorage fs(depth_path,FileStorage::READ);
//	fs["depthmap"]>> depth;
//	fs.release();
//	img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
//	//fn.read_scene_from_training(xyz,img,source_image_path, perspective);
//	//resize image
//	//img.convertTo(img,CV_8UC3);
//	//cv::resize(img,img,cv::Size(640,480));
//	cv::imshow("read img", img);
//	cv::waitKey();
//
//	//create xyz mat from depth
//	//xyz_from_depth(depth, xyz);
//	//crop image and xyz
//
//	//create training data
//	//fn.synthFace(img,xyz,norm_size,training_path, src_label);
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

	return 0;
}


#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include<sensor_msgs/image_encodings.h>
#include "std_msgs/String.h"

#include <cv_bridge/cv_bridge.h>

#include<iostream>
#include<fstream>


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

		file="-1";
		nh.param("/cob_people_detection/face_recognizer/file",file,file);
		std::cout<<"input file: "<<file<<std::endl;

		scene_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/camera/depth_registered/points",1);
		img_pub_ = nh.advertise<sensor_msgs::Image>("/camera/rgb/image_color",1);

		set_pub_ = nh.advertise<std_msgs::String>("set_path",1);
		file_name_pub_ = nh.advertise<std_msgs::String>("file_name",1);

	}

	// Destructor
	~scene_publisher()
	{
	}

	void process()
	{
		pc.clear();
		path = "/home/stefan/rgbd_db/";

		std::stringstream xyz_stream,jpg_stream,depth_stream, set_path_stream, file_name_stream;

		set_path_stream <<path.c_str()<<file.c_str();
		set_path.data = set_path_stream.str();
		file_name_stream << std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot;
		file_name.data = file_name_stream.str();

		xyz_stream<<path.c_str()<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_d.xml";
		//jpg_stream<<path.c_str()<<"set_03/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_c.bmp";
		//depth_stream<<path.c_str()<<"set_03/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_d.xml";

		depth_stream<<path<<file.c_str()<<"/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_d.xml";
		jpg_stream<<path<<file.c_str()<<"/"<<std::setw(3)<<std::setfill('0')<<persp<<"_"<<shot<<"_c.bmp";

		std::cout<<xyz_stream.str()<<"\n "<<jpg_stream.str()<<"\n "<<depth_stream.str()<<std::endl;

		//read bmp image
		cv::Mat img;
		img=cv::imread(jpg_stream.str().c_str());
		//img=cv::imread("/home/stefan/rgbd_db/007_2_c.bmp");
		img.convertTo(img,CV_8UC3);
		cv::resize(img,img,cv::Size(640,480));
		//std::cout << "img read... size: " << img.size() << " rows: " << img.rows << " cols: " << img.cols << std::endl;

		//read depth xml
		cv::Mat dm;
		cv::FileStorage fs_read2(depth_stream.str().c_str(),cv::FileStorage::READ);
		fs_read2["depth"]>> dm;
		fs_read2.release();
		std::cout << "depthmap read... size: " << dm.size() << " rows: " << dm.rows << " cols: " << dm.cols << std::endl;

//		cv::Mat xyz;
//		cv::FileStorage fs_read(xyz_stream.str().c_str(),cv::FileStorage::READ);
//		fs_read["depth"]>> xyz;
//		fs_read.release();
		//std::cout << "xyz read... size: " << xyz.size() << " rows: " << xyz.rows << " cols: " << xyz.cols << std::endl;

		//set parameters
		pc.width=640;
		pc.height=480;
		//cam mat found in scene publisher
		cv::Mat cam_mat =(cv::Mat_<double>(3,3)<< 524.90160178307269,0.0,320.13543361773458,0.0,525.85226379335393,240.73474482242005,0.0,0.0,1.0);
		//f_x and f_y do NOT agree with my fov based calculation. I don't know how they were determined. AAU was unable to locate the kinect used for this, so could not help solve the issue.
		//However, these values are very close to those found for the kinect rgb camera at nicolas.burrus.name/index.php/Research/KinectCalibration

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
		int failctr=0;
		for(int r=0;r<dm.rows;r++)
		{
			for(int c=0;c<dm.cols;c++)
			{
				//pt  << c,r,1.0/525;
				//pt=cam_mat_inv*pt;

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

		//std::cout << "done with point cloud: " << pc.size() << std::endl;
	}

	void publish()
	{
		out_pc2.header.frame_id="/camera/";
		out_pc2.header.stamp = ros::Time::now();
		scene_pub_.publish(out_pc2);

		out_img.header.frame_id="/camera/";
		out_img.header.stamp = ros::Time::now();
		img_pub_.publish(out_img);

		set_pub_.publish(set_path);
		file_name_pub_.publish(file_name);
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

		ros::Publisher set_pub_;
		std_msgs::String set_path;

		ros::Publisher file_name_pub_;
		std_msgs::String file_name;
};



//int main (int argc, char** argv)
//{
//	std::string line;
//	std::vector<int> perspective_filter;
//	std::ifstream persp_file("/home/stefan/rgbd_db_tools/perspectives.txt");
//	int filter_it=0;
//	if (persp_file.is_open())
//	{
//		int persp;
//		while (getline(persp_file, line))
//		{
//			std::cout << line << std::endl;
//			std::istringstream (line) >> persp;
//			perspective_filter.push_back(persp);
//		}
//	}
//
//	ros::init (argc, argv, "scene_publisher");
//	int persp_filter=7;
//
//	scene_publisher sp;
//	ros::Rate loop_rate(0.4);
//	sp.persp=perspective_filter[filter_it];
//
//	while (ros::ok())
//	{
//		sp.shot++;
//		if(sp.shot==4)
//		{
//			sp.shot=1;
//			filter_it++;
//			if(filter_it < perspective_filter.size())
//			{
//				sp.persp=perspective_filter[filter_it];
//			}
//			//sp.persp++;
//		}
//		if(filter_it == perspective_filter.size())
//		{
//			loop_rate.sleep();
//			break;
//		}
//		if(sp.persp==14)
//		{
//			loop_rate.sleep();
//			break;
//		}
//		std::cout<<"PERSPECTIVE:"<<sp.persp<<std::endl;
//		sp.process();
//		sp.publish();
//		ros::spinOnce ();
//		loop_rate.sleep();
//	}
//}
