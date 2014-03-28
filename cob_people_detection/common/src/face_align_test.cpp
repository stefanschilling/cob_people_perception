
#include"cob_people_detection/face_normalizer.h"
#include<iostream>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int main(int argc, const char *argv[])
{


	std::cout<<"[FaceNormalizer] running scene no. "<<argv[1]<<"...\n";
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
	std::string i_path;
	std::cout << "test :)";
	//else      i_path="/share/goa-tz/people_detection/eval/Kinect3DSelect/";
	// i_path="/share/goa-tz/people_detection/eval/KinectIPA/";
	std::string depth_path=i_path;

	std::string training_path = "/home/rmb-ss/.ros/cob_people_detection/files/training_data";
	fn.read_scene_from_training(xyz,img,training_path, argv[1]);
	//fn.read_scene(xyz,img,xml_path);
	cv::Mat wmat1,wmat2;
	img.copyTo(wmat1);
	img.copyTo(wmat2);
	cv::Size norm_size=cv::Size(100,100);
	//cv::cvtColor(wmat1,wmat1,CV_RGB2BGR);

	//cv::imshow("ORIGINAL",wmat1);
	float score;
	fn.frontFaceImage(img,xyz,score);
	std::cout << score <<std::endl;

	cv::Mat depth_res;
	std::vector<cv::Mat> synth_images;
	std::vector<cv::Mat> synth_depths;

	// call member functions of FaceNormalizer
	fn.synthFace(wmat1,xyz,norm_size,synth_images,synth_depths);
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
	//cv::waitKey();
	std::cout<<"..done\n";
	return 0;
}
