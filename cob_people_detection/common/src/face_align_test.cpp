#include"cob_people_detection/face_normalizer.h"
#include<iostream>
#include<opencv/cv.h>
#include<opencv/highgui.h>

int main(int argc, const char *argv[])
{
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
	cv::Mat depth,img,xyz;
	//else      i_path="/share/goa-tz/people_detection/eval/Kinect3DSelect/";
	// i_path="/share/goa-tz/people_detection/eval/KinectIPA/";
	// std::string training_path = "/home/stefan/.ros/cob_people_detection/files/training_data/";
	std::string training_path = "/home/rmb-ss/.ros/cob_people_detection/files/training_data/";
	std::string tdata_path = training_path + "tdata.xml";

	cv::FileStorage fileStorage(tdata_path, cv::FileStorage::READ);
	if (!fileStorage.isOpened())
	{
		std::cout << "Error: load Training Data: Can't open " << tdata_path << ".\n" << std::endl;
	}

	// search tdata for scenes associated to label, store scene numbers in vector
	std::vector<int> scene_numbers;
	std::vector<float> scene_scores;
	std::string label, src_label;
	std::cout << "Enter label to work with: \n";
	std::cin >> src_label;

	int number_entries = (int)fileStorage["number_entries"];
	scene_numbers.clear();
	for(int i=0; i<number_entries; i++)
	{
		// labels
		std::ostringstream tag_label;
		tag_label << "label_" << i;
		//std::cout<<"tag label: " << tag_label << " done\n";
		label = (std::string)fileStorage[tag_label.str().c_str()];
		//std::cout<<"label: " << label << std::endl;
		if(label == src_label) scene_numbers.push_back(i);
	}
	fileStorage.release();

	if (scene_numbers.size()==0)
	{
		std::cout << "No entries of that label in tdata.xml\n";
		return 0;
	}

	// read scenes associated with label, rate for seeding viability
	for(int i=0; i<scene_numbers.size(); i++)
	{
		//if (i==10)
		//{
			fn.read_scene_from_training(xyz,img,training_path, scene_numbers.at(i));
			fn.frontFaceImage(img,xyz,scene_scores);
		//}
	}

	// find best seed to created images from
	int best_seed_id=-1;
	float best_score =10;
	for(int i=0; i<scene_scores.size();i++)
	{
		if (scene_scores[i]<best_score)
		{
			best_seed_id=i;
			best_score=scene_scores[i];
		}
	}
	//in case of data corruption, set ID of valid data manually
	//best_seed_id = 10;

	//std::cout<<"Best source image ID: " << best_seed_id << " Score: " << scene_scores[best_seed_id] << std::endl;
	scene_numbers.clear();
	scene_scores.clear();

	// create synth images from source
	cv::Size norm_size=cv::Size(100,100);
	std::vector<cv::Mat> synth_images;
	std::vector<cv::Mat> synth_depths;

	//load source image, create training data
	fn.read_scene_from_training(xyz,img,training_path, best_seed_id);
	fn.synthFace(img,xyz,norm_size,synth_images,synth_depths,training_path, label);

	//cv::FileStorage fileStorage(tdata_path, cv::FileStorage::WRITE);

	// call member functions of FaceNormalizer

	//fn.isolateFace(wmat1,xyz);
	//fn.normalizeFace(wmat1,xyz,norm_size,depth);
	//fn.recordFace(synth_images[4],synth_depths[4]);

	//synth_images.push_back(wmat1);
	// depth.convertTo(depth,CV_8UC1,255);
	// cv::equalizeHist(depth,depth);

	return 0;
}
