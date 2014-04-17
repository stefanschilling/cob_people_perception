
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
	std::string training_path = "/home/rmb-ss/.ros/cob_people_detection/files/training_data/";
	std::string tdata_path = training_path + "tdata.xml";

	// search tdata for scenes associated to label, store scene numbers in vector
	std::vector<int> scene_numbers;
	std::vector<float> scene_scores;

	cv::FileStorage fileStorage(tdata_path, cv::FileStorage::READ);
	if (!fileStorage.isOpened())
	{
		std::cout << "Error: load Training Data: Can't open " << tdata_path << ".\n" << std::endl;
	}

	int number_entries = (int)fileStorage["number_entries"];
	scene_numbers.clear();
	for(int i=0; i<number_entries; i++)
	{
		// labels
		std::ostringstream tag_label;
		tag_label << "label_" << i;
		std::cout<<"tag label: " << tag_label << " done\n";
		std::string label = (std::string)fileStorage[tag_label.str().c_str()];
		std::cout<<"label: " << label << std::endl;
		if(label == "Stefan")
			scene_numbers.push_back(i);
	}

	// read scenes associated with label, rate for seeding viability
	for(int i=0; i<scene_numbers.size(); i++)
	{
		fn.read_scene_from_training(xyz,img,training_path, scene_numbers.at(i));
		fn.frontFaceImage(img,xyz,scene_scores);
	}

	// pick and display best seed
	int best_seed_id=-1;
	float best_score =10;
	for(int i=0; i<scene_numbers.size();i++)
	{
		if (scene_scores[i]<best_score)
		{
			best_seed_id=i;
			best_score=scene_scores[i];
		}
	}
	std::cout<<"found best seed. ID: " << best_seed_id << " Score: " << scene_scores[best_seed_id] << std::endl;


	// create synth images from seed
	cv::Size norm_size=cv::Size(100,100);
	cv::Mat wmat1,wmat2;
	std::vector<cv::Mat> synth_images;
	std::vector<cv::Mat> synth_depths;

	fn.read_scene_from_training(xyz,img,training_path, best_seed_id);
	img.copyTo(wmat1);
	xyz.copyTo(wmat2);

	fn.synthFace(wmat1,wmat2,norm_size,synth_images,synth_depths);


	// call member functions of FaceNormalizer

	//fn.isolateFace(wmat1,xyz);
	//fn.normalizeFace(wmat1,xyz,norm_size,depth);
	//fn.recordFace(synth_images[4],synth_depths[4]);


	//synth_images.push_back(wmat1);
	// depth.convertTo(depth,CV_8UC1,255);
	// cv::equalizeHist(depth,depth);


//	for(int j = 0;j<synth_images.size();j++)
//	{
//		cv::imshow("NORMALIZED",synth_images[j]);
//		std::cout<<j<<std::endl;
//		cv::waitKey();
//	}

	return 0;
}
