#include "cob_people_detection/face_normalizer.h"
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"

bool create_file_lists(std::string& path, std::vector<std::string>& xmls)
{
	boost::filesystem::path boost_path;
	boost_path=path;
	if (!exists(boost_path))
	{
		std::cout<<"requested path does not exist: " << path << std::endl;
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
		}
	}
	if (xmls.size()>0)
	{
		return true;
	}
	else return false;
}

void rewrite_tdata(std::string& training_path, std::string& added_label, int& img_count, std::vector<int>& imgs_per_axis, bool separation)
{
	std::cout << "write new tdata";
	//write new tdata2 file for newly created image files
	//keep old tdata2 entries, if they exist.
	cv::FileStorage fileStorageRead(training_path + "tdata2.xml", cv::FileStorage::READ);
	if (!fileStorageRead.isOpened())
	{
		std::cout << "Error: load Training Data: Can't open " << training_path+"tdata2.xml" << ".\n" << std::endl;
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
	int k=0;
	int done=0;
	std::string stamp,label;
	std::cout << "Status:" <<std::endl;
	cv::FileStorage fileStorageWrite(training_path+"tdata2.xml", cv::FileStorage::WRITE);
	if (fileStorageWrite.isOpened())
	{
		fileStorageWrite << "number_entries" << (int)img_count;
		//for(int i=0; i<img_count; i++)
		for(int i=0; i<old_number_entries; i++)
		{
			std::ostringstream tag, tag2, tag3;
			std::ostringstream shortname_img, shortname_depth;

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

		int done = old_number_entries;
		for (int i =0; i < imgs_per_axis.size();i++)
		{
			for (int j=0; j < imgs_per_axis[i]; j++)
			{
				std::ostringstream tag, tag2, tag3;
				std::ostringstream shortname_img, shortname_depth;
				std::ostringstream convert;
				if (j<2) convert << 0;
				if (j>=2) convert << (i%8)+1;
				stamp = convert.str();
				label = added_label;
				//stamp label with axis+orientation if separation is required
				if (separation)	label = added_label +"_"+ stamp;
				//std::cout << "Number: " << i+old_number_entries << ", Done: " << done << ", Axis: " << (i%8) +1 << ", images in this axis: " << imgs_per_axis[i] << ", stamp: " << stamp << ", Label: " << label_axis_stamped<< std::endl;

				// labels
				tag << "label_" << done;
				fileStorageWrite << tag.str().c_str() << label;

				// face images
				shortname_img << "synth_img_test/" << done << ".bmp";
				tag2 << "image_" << done;
				fileStorageWrite << tag2.str().c_str() << shortname_img.str().c_str();

				// depth xmls
				shortname_depth << "synth_depth_test/" << done << ".xml";
				tag3 << "depthmap_" << done;
				fileStorageWrite << tag3.str().c_str() << shortname_depth.str().c_str();
				done++;
			}

		}
		fileStorageWrite.release();
		face_labels.clear();
		face_images.clear();
		face_depths.clear();
		imgs_per_axis.clear();
	}
	std::cout << " - writing completed, imgs_per_axis cleared: " << imgs_per_axis.size() <<std::endl;
}

int main(int argc, const char *argv[])
{
	//set up variables
	int img_count, img_count_current, step_size, step_no;
	float step_size_rad;
	std::vector<int> imgs_per_axis;
	step_size=step_no=0;
	std::vector<std::string> xmls;
	std::string features,set, set_list, src_set, src_persp, bmp_path,training_path, label, path;
	training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
	std::vector<std::string> set_filter,perspectives;
	cv::Mat depth, img;
	cv::Size norm_size=cv::Size(100,100);

	//create directories for synth images and xyz-mats if they do not exist
	boost::filesystem::path synth_depth_path = training_path + "synth_depth_test";
	boost::filesystem::path synth_img_path = training_path + "synth_img_test";
	if (!boost::filesystem::exists (synth_depth_path)||!boost::filesystem::exists ( synth_img_path))
	{
		std::cout << "creating img and xyz directories for additional data" << std::endl;
		if (boost::filesystem::create_directory(synth_depth_path) && boost::filesystem::create_directory(synth_img_path))
		{
			std::cout << "directories ready" <<std::endl;
		}
		else
		{
			std::cout << "creating directories failed" <<std::endl;
			return 0;
		}
	}

	//ask set list
	std::cout << "set list to use: ";
	std::cin >> set_list;
	set_list = "/home/stefan/rgbd_db_tools/sets/set"+set_list+".txt";
	std::ifstream set_file(set_list.c_str());

	//read list of sets to create images from
	if (set_file.is_open())
	{
		std::cout << "sets in this Set List: ";
		while (std::getline(set_file, set))
		{
			std::cout << set << ", ";
			set_filter.push_back(set);
		}
		std::cout << std::endl;
	}
	//ask size and number of steps for synth poses
	std::cout << "Rotation step size [rad*100] (0 = train source images only):"<<std::endl;
	std::cin >> step_size;
	if (step_size > 0)
	{
		std::cout << "Number of steps per rotation (total attempted images: 8x step number)" <<std::endl;
		std::cin >> step_no;
	}
	//step_size_rad = (float)step_size*0.0174533;
	step_size_rad = (float)step_size/100;

	//use features for rotation axis?
	std::cout << "use face features for rotation axis? [y,n]";
	std::cin >> features;

	bool separation=false;
	int sep;
	//1 direction at a time?
	std::cout << "Create seperate labels per axis and orientation of rotation? [0:no, 1: yes]" << std::endl;
	std::cin >> sep;

	if (sep !=0)
	{
		std::cout << "separation = true: ";
		separation = true;
		std::cout << separation << std::endl;
	}
	//read list of perspectives used for image creation
	std::ifstream persp_file("/home/stefan/rgbd_db_tools/training_perspectives.txt");
	if (persp_file.is_open())
	{
		std::cout << "Perspectives used for training: ";
		while (std::getline(persp_file, set))
		{
			std::cout << set << ", ";
			perspectives.push_back(set);
		}
		std::cout << std::endl;
	}

	//initialize facenormalizer
	FaceNormalizer::FNConfig cfg;
	cfg.eq_ill=false;
	cfg.align=false;
	if (features == "y") cfg.align=true;
	cfg.resize=true;
	cfg.cvt2gray=false;
	cfg.extreme_illumination_condtions=false;

	std::string  class_path="/opt/ros/groovy/share/OpenCV/";

	FaceNormalizer fn;
	fn.init(class_path,cfg);

	// read number of entries from tdata or start from scratch
	cv::FileStorage fileStorageRead(training_path + "tdata2.xml", cv::FileStorage::READ);
	if (fileStorageRead.isOpened())
	{
		img_count_current = (int)fileStorageRead["number_entries"];
		img_count = img_count_current;
	}
	else
	{
		std::cout << "unable to read tdata2.xml, starting new file at entry 0" <<std::endl;
		img_count = img_count_current= 0;
	}

	//iterate over sets and perspectives to find the required files for requested perspectives and sets,
	//images are saved by synth_face -> synth_head_poses, and the amount of created images is returned
	//from the returned number, their entries are written to tdata2.xml
	for (int j=0; j <set_filter.size();j++)
	{
		src_set = set_filter[j];
		path = "/home/stefan/rgbd_db_heads/"+src_set;
		create_file_lists(path, xmls);
		std::cout << xmls.size() << std::endl;

		label = src_set;

		for (int k=0;k<perspectives.size();k++)
		{
			int count = 0;
			src_persp = perspectives[k];
			std::stringstream xyz_stream;
			xyz_stream<<std::setw(3)<<std::setfill('0')<<src_persp;
			for (int i=1;i<xmls.size();i++)
			{
				unsigned perspective = xmls[i].find_last_of("/\\");
				std::string substring = xmls[i].substr(perspective+1, 3);
				if(count < 1 && xyz_stream.str() == substring)
				{
					std::cout << count << std::endl;
					count++;
					//std::cout << bmps[i] << std::endl;
					bmp_path = xmls[i];
					bmp_path.erase(xmls[i].length()-5,5);
					bmp_path.append("c.bmp");
					//std::cout << bmp_path << std::endl;
					cv::FileStorage fs(xmls[i], cv::FileStorage::READ);
					fs["depthmap"]>> depth;
					img = cv::imread(bmp_path, CV_LOAD_IMAGE_COLOR);
					//cv::imshow("img sent",img);
					//cv::waitKey();
					if(fn.synthFace(img,depth,norm_size,training_path,img_count,step_size_rad, step_no,imgs_per_axis))
					{
						std::cout << "images added from source image " << bmp_path << ": " << img_count-img_count_current <<std::endl;
					}
					img_count_current = img_count;
					fs.release();
				}
			}
		}
		//checkpoint for imgs_per_axis content
//		std::cout << imgs_per_axis.size() << std::endl;
//		for (int iter = 0; iter < imgs_per_axis.size(); iter++)
//		{
//			if (iter >0) std::cout << ", ";
//			std::cout << imgs_per_axis[iter] ;
//		}
//		std::cout << std::endl;
		//std::cout << imgs_per_axis[0] << ", "<< imgs_per_axis[1] << ", "<< imgs_per_axis[2] << ", "<< imgs_per_axis[3] << ", "<< imgs_per_axis[4] << ", "<< imgs_per_axis[5] << ", "<< imgs_per_axis[6] << ", "<< imgs_per_axis[7];

		rewrite_tdata(training_path, label, img_count,imgs_per_axis,separation);
		xmls.clear();
	}

	std::cout << "synth succesful, new image total: " << img_count << std::endl;

	return 0;
}
