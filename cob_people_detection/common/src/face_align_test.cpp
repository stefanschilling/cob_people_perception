#include "cob_people_detection/face_normalizer.h"
//#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "boost/filesystem/operations.hpp"
//#include "boost/filesystem/convenience.hpp"
//#include <boost/thread/mutex.hpp>
#include "boost/filesystem/path.hpp"
//#include "boost/lexical_cast.hpp"
#include "boost/filesystem.hpp"

bool create_file_lists(std::string& path, std::vector<std::string>& xmls)
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
				shortname_img << "synth_img_test/" << i << ".bmp";
				tag2 << "image_" << i;
				fileStorageWrite << tag2.str().c_str() << shortname_img.str().c_str();

				shortname_depth << "synth_depth_test/" << i << ".xml";
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

//alternate main to use fn.orientation with (calculates orientation of person)
//int main (int argc, const char *argv[])
//{
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
//	std::vector<std::string> xmls,bmps;
//	std::string src_set, src_persp, bmp_path,training_path, label, path;
//	training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
//	std::vector<std::string> set_filter;
//	std::string set_list;
//	std::cin >> set_list;
//	//std::string set_list = "x";
//	set_list = "/home/stefan/rgbd_db_tools/sets/set"+set_list+".txt";
//	std::ifstream set_file(set_list.c_str());
//	int set_it=0;
//	std::string set;
//	if (set_file.is_open())
//	{
//		std::cout << "sets in this Set List: ";
//		while (std::getline(set_file, set))
//		{
//			std::cout << set << ", ";
//			//std::stringstream (line) >> set;
//			set_filter.push_back(set);
//		}
//		std::cout << std::endl;
//	}
//	std::vector<std::string> perspectives;
//	std::ifstream persp_file("/home/stefan/rgbd_db_tools/perspectives.txt");
//	int filter_it=0;
//	if (persp_file.is_open())
//	{
//		std::cout << "Perspectives used to check recognition: ";
//		while (std::getline(persp_file, set))
//		{
//			std::cout << set << ", ";
//			perspectives.push_back(set);
//		}
//		std::cout << std::endl;
//	}
//
//	std::vector<std::string> bad_persp;
//	std::ofstream output_textfile;
//	output_textfile.open("/home/stefan/rgbd_db_tools/feat_and_persp.txt");
//	output_textfile.clear();
//	output_textfile << "Feature Detection across Sets" <<std::endl;
//	output_textfile << "=============================" << std::endl;
//
//	for (int j=0; j <set_filter.size();j++)
//	{
//		src_set = set_filter[j];
//		path = "/home/stefan/rgbd_db_heads/"+src_set;
//		create_file_lists(path, xmls, bmps);
//		std::cout << xmls.size() << std::endl;
//		std::cout << bmps.size() << std::endl;
//
//		label = src_set;
//		output_textfile << "Set: " << src_set <<std::endl;
//
//		int feat_counter=0;
//		int head_counter=0;
//
//		for (int k=0;k<perspectives.size();k++)
//		{
//			cv::Mat img,xyz;
//			src_persp = perspectives[k];
//			std::stringstream xyz_stream;
//			xyz_stream<<std::setw(3)<<std::setfill('0')<<src_persp;
//			int feat_in_persp=0;
//			for (int i=1;i<xmls.size();i++)
//			{
//				unsigned perspective = xmls[i].find_last_of("/\\");
//				std::string substring = xmls[i].substr(perspective+1, 3);
//				int count = 0;
//				//output_textfile << substring << " ";
//				if(count < 3 && xyz_stream.str() == substring)
//				{
//					head_counter++;
//					//output_textfile << xmls[i].substr(perspective+1) << ": ";
//					bmp_path = xmls[i];
//					bmp_path.erase(xmls[i].length()-5,5);
//					bmp_path.append("c.bmp");
//					//std::cout << bmp_path << std::endl;
//					cv::FileStorage fs(xmls[i], cv::FileStorage::READ);
//					fs["depthmap"]>> xyz;
//					img = cv::imread(bmp_path, CV_LOAD_IMAGE_COLOR);
//					if(fn.orientations(img, xyz))
//					{
//						feat_in_persp++;
//						//output_textfile << "True" << std::endl;
//					}
//					//else output_textfile << "False" << std::endl;
//				}
//
//			}
//			output_textfile << "Perspective " << src_persp << " has "<< feat_in_persp << " images available for synth!" << std::endl;
//			if (feat_in_persp == 0) bad_persp.push_back(src_persp);
//			feat_counter += feat_in_persp;
//		}
//		output_textfile << "Features detected in " << feat_counter << " of " << head_counter << "tested files" << std::endl;
//		output_textfile << "In percent: " << (float)feat_counter/head_counter << std::endl;
//		if (bad_persp.size() >0)
//		{
//			output_textfile << "Perspectives unavailable for synth: ";
//			for (int k = 0; k< bad_persp.size(); k++)
//			{
//				output_textfile << bad_persp[k] << " ";
//			}
//		}
//		bad_persp.clear();
//		output_textfile << std::endl;
//		output_textfile << std::endl;
//		xmls.clear();
//		bmps.clear();
//	}
//	output_textfile.close();
//
//	return 0;
//}

int main(int argc, const char *argv[])
{
	//set up variables
	std::vector<std::string> xmls;
	std::string src_set, src_persp, bmp_path,training_path, label, path;
	training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
	std::vector<std::string> set_filter;

	//ask set list
	std::string set_list;
	std::cin >> set_list;
	set_list = "/home/stefan/rgbd_db_tools/sets/set"+set_list+".txt";
	std::ifstream set_file(set_list.c_str());
	std::string set;

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

	std::vector<std::string> perspectives;
	std::ifstream persp_file("/home/stefan/rgbd_db_tools/perspectives.txt");
	//read list of perspectives used for image creation
	if (persp_file.is_open())
	{
		std::cout << "Perspectives used to check recognition: ";
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
	cfg.align=true;
	cfg.resize=true;
	cfg.cvt2gray=false;
	cfg.extreme_illumination_condtions=false;

	std::string  class_path="/opt/ros/groovy/share/OpenCV/";

	FaceNormalizer fn;
	fn.init(class_path,cfg);
	cv::Mat depth, img;
	cv::Size norm_size=cv::Size(100,100);

	// read number of entries from tdata
	cv::FileStorage fileStorageRead(training_path + "tdata2.xml", cv::FileStorage::READ);
	int img_count, img_count_current;
	if (fileStorageRead.isOpened())
	{
		int img_count_current = (int)fileStorageRead["number_entries"];
		img_count = img_count_current;
	}
	else
	{
		std::cout << "unable to read tdata2.xml, creating new one starting at 0 entries" <<std::endl;
		img_count = img_count_current= 0;
	}

	//iterate over sets and perspectives to find the required perspectives per set,
	//create images for them, and write their entries to tdata2.xml
	for (int j=0; j <set_filter.size();j++)
	{
		src_set = set_filter[j];
		path = "/home/stefan/rgbd_db_heads/"+src_set;
		create_file_lists(path, xmls);
		std::cout << xmls.size() << std::endl;

		label = src_set;

		for (int k=0;k<perspectives.size();k++)
		{
				src_persp = perspectives[k];
			std::stringstream xyz_stream;
			xyz_stream<<std::setw(3)<<std::setfill('0')<<src_persp;
			for (int i=1;i<xmls.size();i++)
			{
				unsigned perspective = xmls[i].find_last_of("/\\");
				std::string substring = xmls[i].substr(perspective+1, 3);
				int count = 0;
				if(count < 3 && xyz_stream.str() == substring)
				{
					count++;
					//std::cout << bmps[i] << std::endl;
					bmp_path = xmls[i];
					bmp_path.erase(xmls[i].length()-5,5);
					bmp_path.append("c.bmp");
					//std::cout << bmp_path << std::endl;
					cv::FileStorage fs(xmls[i], cv::FileStorage::READ);
					fs["depthmap"]>> depth;
					img = cv::imread(bmp_path, CV_LOAD_IMAGE_COLOR);
					if(fn.synthFace(img,depth,norm_size,training_path,img_count))
					{
						std::cout << "images added from source image " << bmp_path << ": " << img_count-img_count_current <<std::endl;
					}
					fs.release();
				}
			}
		}
		rewrite_tdata(training_path, label, img_count);
		xmls.clear();
	}
	std::cout << "synth succesful, new image total: " << img_count << std::endl;

	return 0;
}
