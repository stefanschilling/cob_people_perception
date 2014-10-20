#ifdef __LINUX__
	#include "cob_people_detection/rgbd_db_recognition_test.h"
	#include "cob_vision_utils/GlobalDefines.h"
#else
#endif

using namespace ipa_PeopleDetector;

RgbdDbRecognitionTest::RgbdDbRecognitionTest(ros::NodeHandle nh)
: node_handle_(nh)
{
	// Parameters for Face Detector
	double faces_increase_search_scale;		// The factor by which the search window is scaled between the subsequent scans
	int faces_drop_groups;					// Minimum number (minus 1) of neighbor rectangles that makes up an object.
	int faces_min_search_scale_x;			// Minimum search scale x
	int faces_min_search_scale_y;			// Minimum search scale y
	bool reason_about_3dface_size;			// if true, the 3d face size is determined and only faces with reasonable size are accepted
	double face_size_max_m;					// the maximum feasible face diameter [m] if reason_about_3dface_size is enabled
	double face_size_min_m;					// the minimum feasible face diameter [m] if reason_about_3dface_size is enabled
	double max_face_z_m;					// maximum distance [m] of detected faces to the sensor
	bool debug;								// enables some debug outputs
	std::cout << "\n--------------------------\nFace Detector Parameters:\n--------------------------\n";
	node_handle_.getParam("/cob_people_detection/data_directory", data_directory_);
	std::cout << "data_directory = " << data_directory_ << "\n";
	node_handle_.param("faces_increase_search_scale", faces_increase_search_scale, 1.1);
	std::cout << "faces_increase_search_scale = " << faces_increase_search_scale << "\n";
	node_handle_.param("faces_drop_groups", faces_drop_groups, 68);
	std::cout << "faces_drop_groups = " << faces_drop_groups << "\n";
	node_handle_.param("faces_min_search_scale_x", faces_min_search_scale_x, 20);
	std::cout << "faces_min_search_scale_x = " << faces_min_search_scale_x << "\n";
	node_handle_.param("faces_min_search_scale_y", faces_min_search_scale_y, 20);
	std::cout << "faces_min_search_scale_y = " << faces_min_search_scale_y << "\n";
	node_handle_.param("reason_about_3dface_size", reason_about_3dface_size, true);
	std::cout << "reason_about_3dface_size = " << reason_about_3dface_size << "\n";
	node_handle_.param("face_size_max_m", face_size_max_m, 0.35);
	std::cout << "face_size_max_m = " << face_size_max_m << "\n";
	node_handle_.param("face_size_min_m", face_size_min_m, 0.1);
	std::cout << "face_size_min_m = " << face_size_min_m << "\n";
	node_handle_.param("max_face_z_m", max_face_z_m, 8.0);
	std::cout << "max_face_z_m = " << max_face_z_m << "\n";
	node_handle_.param("debug", debug, false);
	std::cout << "debug = " << debug << "\n";
	node_handle_.param("display_timing", display_timing_, false);
	std::cout << "display_timing = " << display_timing_ << "\n";

	// initialize face detector
	face_detector_.init(data_directory_, faces_increase_search_scale, faces_drop_groups, faces_min_search_scale_x, faces_min_search_scale_y,
			reason_about_3dface_size, face_size_max_m, face_size_min_m, max_face_z_m, debug);

	std::cout << "Face Detector initialized." << std::endl;


	// Parameters for Face Recognizer
	bool norm_illumination;
	bool norm_align;
	bool norm_extreme_illumination;
	int  norm_size;						// Desired width and height of the Eigenfaces (=eigenvectors).
	int feature_dimension;			// Number of eigenvectors per person to identify -> controls the total number of eigenvectors
	double threshold_facespace;				// Threshold to facespace
	double threshold_unknown;				// Threshold to detect unknown faces
	int metric; 							// metric for nearest neighbor search in face space: 0 = Euklidean, 1 = Mahalanobis, 2 = Mahalanobis Cosine
	int recognition_method;           // choose subspace method
	bool use_unknown_thresh; // use threshold for unknown faces
	bool use_depth;         // use depth for recognition
	std::cout << "\n--------------------------\nFace Recognizer Parameters:\n--------------------------\n";
	//if(!node_handle_.getParam("~data_directory", data_directory_)) std::cout<<"PARAM NOT AVAILABLE"<<std::endl;
	if(!node_handle_.getParam("/cob_people_detection/data_storage_directory", data_directory_)) std::cout<<"PARAM NOT AVAILABLE"<<std::endl;
	std::cout << "data_directory = " << data_directory_ << "\n";
	node_handle_.param("enable_face_recognition", enable_face_recognition_, true);
	std::cout << "enable_face_recognition = " << enable_face_recognition_ << "\n";
	node_handle_.param("feature_dimension", feature_dimension, 10);
	std::cout << "feature dimension = " << feature_dimension << "\n";
	node_handle_.param("threshold_facespace", threshold_facespace, 10000.0);
	std::cout << "threshold_facespace = " << threshold_facespace << "\n";
	node_handle_.param("threshold_unknown", threshold_unknown, 1000.0);
	std::cout << "threshold_unknown = " << threshold_unknown << "\n";
	node_handle_.param("metric", metric, 0);
	std::cout << "metric = " << metric << "\n";
	node_handle_.param("debug", debug, false);
	std::cout << "debug = " << debug << "\n";
	node_handle_.param("recognition_method", recognition_method, 3);
	std::cout << "recognition method: " << recognition_method << "\n";
	node_handle_.param("use_unknown_thresh", use_unknown_thresh, true);
	std::cout << " use use unknown thresh: " << use_unknown_thresh << "\n";
	node_handle_.param("use_depth", use_depth, true);
	std::cout << " use depth: " << use_depth << "\n";
	node_handle_.param("display_timing", display_timing_, false);
	std::cout << "display_timing = " << display_timing_ << "\n";
	node_handle_.param("norm_size", norm_size, 100);
	std::cout << "norm_size = " << norm_size << "\n";
	node_handle_.param("norm_illumination", norm_illumination, true);
	std::cout << "norm_illumination = " << norm_illumination << "\n";
	node_handle_.param("norm_align", norm_align, false);
	std::cout << "norm_align = " << norm_align << "\n";
	node_handle_.param("norm_extreme_illumination", norm_extreme_illumination, false);
	std::cout << "norm_extreme_illumination = " << norm_extreme_illumination << "\n";
	node_handle_.param("debug", debug, false);
	std::cout << "debug = " << debug << "\n";
	node_handle_.param("use_depth",use_depth,false);
	std::cout<< "use depth: "<<use_depth<<"\n";
	// todo: make parameters for illumination and alignment normalization on/off
	node_handle_.param("transform_labels", trans_labels_, false);
	std::cout << "transform labels to split axis: " <<trans_labels_ << "\n";

	std::cout << "identification_labels_to_recognize: \n";
	XmlRpc::XmlRpcValue identification_labels_to_recognize_list;
	node_handle_.getParam("identification_labels_to_recognize", identification_labels_to_recognize_list);
	if (identification_labels_to_recognize_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
	{
		if (!trans_labels_) identification_labels_to_recognize_.resize(identification_labels_to_recognize_list.size());
		else if (trans_labels_) identification_labels_to_recognize_.resize(9*identification_labels_to_recognize_list.size());
		for (int i = 0; i < identification_labels_to_recognize_list.size(); i++)
		{
			ROS_ASSERT(identification_labels_to_recognize_list[i].getType() == XmlRpc::XmlRpcValue::TypeString);
			if (!trans_labels_) identification_labels_to_recognize_[i] = static_cast<std::string>(identification_labels_to_recognize_list[i]);
			else
			{
				for (int j =0; j<9; j++)
				{
					std::stringstream convert;
					convert << "_" << (j);
					identification_labels_to_recognize_[9*i+j] = static_cast<std::string>(identification_labels_to_recognize_list[i]).append(convert.str());
				}
			}
		}
	}
	std::cout << "Labels to recognize: " << std::endl;
	for (int j = 0; j<identification_labels_to_recognize_.size(); j++) std::cout << identification_labels_to_recognize_[j] << std::endl;

	// initialize face recognizer
	face_recognizer_.init(data_directory_, norm_size,norm_illumination,norm_align,norm_extreme_illumination, metric, debug, identification_labels_to_recognize_,recognition_method, feature_dimension, use_unknown_thresh, use_depth);

	//Parameters for Test: database directory, sets and perspectives to use
	XmlRpc::XmlRpcValue perspectives_list, sets_list;
	if(!node_handle_.getParam("/cob_people_detection/rgbd_db_heads_directory", rgbd_db_directory_)) std::cout<<"PARAM NOT AVAILABLE"<<std::endl;
	std::cout << "rgbd_db_heads_directory = " << rgbd_db_directory_ << "\n";
	node_handle_.getParam("sets_list", sets_list);
	std::cout << "sets_list = " << sets_list << "\n";
	node_handle_.getParam("perspectives_list",perspectives_list);
	std::cout<< "perspectives_list: "<<perspectives_list<<"\n";
	node_handle_.param("detail",detail_,false);
	std::cout<< "output in full detail: "<<detail_<<"\n";

	// translate set and perspective arrays from xml into string vectors
	if (sets_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
	{
		sets_.resize(sets_list.size());
		for (int i = 0; i < sets_list.size(); i++)
		{
			ROS_ASSERT(sets_list[i].getType() == XmlRpc::XmlRpcValue::TypeString);
			sets_[i] = static_cast<std::string>(sets_list[i]);
		}
	}

	if (perspectives_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
	{
		perspectives_.resize(perspectives_list.size());
		for (int i = 0; i < perspectives_list.size(); i++)
		{
			ROS_ASSERT(perspectives_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt);
			perspectives_[i] = static_cast<int>(perspectives_list[i]);
		}
	}

	this->face_found_=0;
}

RgbdDbRecognitionTest::~RgbdDbRecognitionTest(void)
{
}

bool RgbdDbRecognitionTest::loadModel(std::vector<std::string>& identification_labels_to_recognize)
{
	// load the corresponding recognition model
	bool result_state = face_recognizer_.loadRecognitionModel(identification_labels_to_recognize);

	return result_state;
}

bool RgbdDbRecognitionTest::checkOrientation(char label_tag, int p)
{
	// compare found orientation with perspective
	switch (label_tag)
	{
	case '0':
		if (p == 7)
		{
			return true;
		}
		break;
	case '1':
		if (p == 11 || p == 13)
		{
			return true;
		}
		break;
	case '2':
		if (p == 10)
		{
			return true;
		}
		break;
	case '3':
		if (p == 5 || p ==6 )
		{
			return true;
		}
		break;
	case '4':
		if (p == 2)
		{
			return true;
		}
		break;
	case '5':
		if (p == 1 || p == 3)
		{
			return true;
		}
		break;
	case '6':
		if (p == 4)
		{
			return true;
		}
		break;
	case '7':
		if (p == 8 || p == 9)
		{
			return true;
		}
		break;
	case '8':
		if (p == 12)
		{
			return true;
		}
		break;
	}

	return false;
}

void RgbdDbRecognitionTest::TestRecognition()
{
	std::string out_path;
	std::stringstream bmp_stream,xyz_stream;
	cv::Mat bmp,xyz;

	//ints for recognition percentage
	int set_rec, total_rec, set_imgs, total_imgs;
	set_rec=total_rec=set_imgs=total_imgs=0;

	//ints for perspective/orientation test
	int set_persp, set_persp_rec, total_persp, total_persp_rec;
	set_persp=set_persp_rec=total_persp=total_persp_rec = 0;

	//get current data and time to stamp test output file with
	out_path.append(rgbd_db_directory_);
	time_t now;
	char date_c[24];
	now = time(NULL);
	if (now != -1) strftime(date_c, 24, "%d_%m_%Y_%H_%M_%S", gmtime(&now));
	std::string date(date_c);
	date.append(".txt");
	out_path.append(date);
	std::cout << "output file for this test: " << out_path << std::endl;

    std::ofstream output_textfile(out_path.c_str(), std::ios_base::app);
    writeSetup(output_textfile);

    //iterate through requested sets in requested perspectives, through all shots (maximum of 3 shots per perspective in database)
	for (int set=0; set < sets_.size(); set++)
	{
		//build path to set
		std::string path;
		path.append(rgbd_db_directory_);
		path.append(sets_[set]);
		std::cout << path << std::endl;
		output_textfile << sets_[set] << "\n";
        for (int persp=0; persp < perspectives_.size(); persp++)
		{
			for (int shot=1; shot < 4; shot++)
			{
				//vectors for detector and recognizer
				std::vector<cv::Mat> bmp_v,xyz_v;
				std::vector<std::vector<cv::Rect> > face_rect_v;
				std::vector< std::vector<std::string> > label_v, labels;
				std::vector< std::vector < float> > scores;

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
					bmp_test.close();
					xyz_test.close();
					//read bmp image
					bmp=cv::imread(bmp_stream.str().c_str());
					bmp.convertTo(bmp,CV_8UC3);
					//read depth xml
					cv::FileStorage fs_read2(xyz_stream.str().c_str(),cv::FileStorage::READ);
					fs_read2["depthmap"]>> xyz;
					fs_read2.release();
					bmp_v.push_back(bmp);
					xyz_v.push_back(xyz);

					// detect face
					face_detector_.detectColorFaces(bmp_v, xyz_v, face_rect_v);
					//std::cout << "detected faces: " << face_rect_v.size();
					if (face_rect_v.size() >0 )
						{
						face_found_++;
						//std::cout << " and " << face_rect_v[0].size();
						}
					//std::cout << " for a total of " << face_found_ << "faces" <<std::endl;

					// if single face detection, recognize face
					if(face_rect_v.size()==1 && face_rect_v[0].size()==1)
					{
						bool rec, ori;
						rec=ori=false;
						set_imgs++;
						face_recognizer_.recognizeFaces(bmp_v, xyz_v, face_rect_v, label_v, labels, scores);
						std::cout << "compare " << sets_[set] << " with " << label_v[0][0].substr(0,6) << std::endl;
						if (label_v[0][0].substr(0,6) == sets_[set])
						{
							set_rec++;
							rec=true;
						}
						if (detail_)
						{
							output_textfile << bmp_stream.str().substr(found+1,bmp_stream.str().size()-3) << " - " << label_v[0][0] << "\n";
						}
						if (trans_labels_)
						{
							bool right_ori =checkOrientation(label_v[0][0][7], perspectives_[persp]);
							if (right_ori)
							{
								set_persp++;
								if(rec) set_persp_rec++;
							}
						}
					}
					else if (detail_)
					{
						set_imgs++;
						output_textfile << bmp_stream.str().substr(found+1,bmp_stream.str().size()-3) << " - faulty detection - " << face_rect_v[0].size() << " faces found in img \n";
					}
				}
				else if (bmp_test.good() != xyz_test.good()) continue; //bmp or xml went missing
			}
		}
        //output for set totals
        std::cout << "Images tested: " << set_imgs << " - Images correctly labeled: " << set_rec << "\n Percentage correct: " << (float) set_rec/set_imgs << "\n";
        if (trans_labels_) std::cout << "Orientation correct: " << set_persp << " - Orientation and label correct: " << set_persp_rec << "\n Percentages - orientation: " << (float)set_persp/set_imgs << ", label and orientation: " << (float) set_persp_rec/set_imgs;
        std::cout << std::endl;

        output_textfile << "Images tested: " << set_imgs << " - Images correctly labeled: " << set_rec << "\n Percentage correct: " << (float) set_rec/set_imgs << "\n";
        if (trans_labels_) output_textfile<< "Orientation correct: " << set_persp << " - Orientation and label correct: " << set_persp_rec << "\n Percentages - orientation: " << (float)set_persp/set_imgs << ", label and orientation: " << (float) set_persp_rec/set_imgs << "\n";
        output_textfile << "\n";
        total_rec+=set_rec;
		total_imgs+=set_imgs;
		total_persp+=set_persp;
		total_persp_rec+=set_persp_rec;

		set_rec=set_imgs=0;
		set_persp=set_persp_rec=0;
	}
	//output for overall totals
	std::cout << "Totals: \nImages tested: " << total_imgs << " - Images correctly labeled: " << total_rec << "\n Percentage correct: " << (float) total_rec/total_imgs << "\n";
	if (trans_labels_) std::cout << "Orientation correct: " << total_persp << " - Orientation and label correct: " << total_persp_rec << "\n Percentages - orientation: " << (float)total_persp/total_imgs << ", label and orientation: " << (float) total_persp_rec/total_imgs;

	output_textfile << "\n";
	output_textfile << "Totals: \nImages tested: " << total_imgs << " - Images correctly labeled: " << total_rec << "\n Percentage correct: " << (float) total_rec/total_imgs << "\n";
	output_textfile << "Orientation correct: " << total_persp << " - Orientation and label correct: " << total_persp_rec << "\n Percentages - orientation: " << (float)total_persp/total_imgs << ", label and orientation: " << (float) total_persp_rec/total_imgs;
	output_textfile.close();
	std::cout << "test completed" << std::endl;
	ros::shutdown();
	return;
}

void RgbdDbRecognitionTest::writeSetup(std::ofstream& output_textfile)
{
	output_textfile << "Trained Labels: ";
	for (int i=0; i<identification_labels_to_recognize_.size();i++)
	{
		output_textfile << identification_labels_to_recognize_[i] << " ";
	}
	output_textfile << "\n \nTested Labels: ";
	for (int i=0; i<sets_.size();i++)
	{
		output_textfile << sets_[i] << " ";
	}
	output_textfile << "\n \nTested Perspectives: ";
	for (int i=0;i<perspectives_.size(); i++)
	{
		output_textfile << perspectives_[i] << " ";
	}
	output_textfile << "\n \n \n";
	return;
}


int main(int argc, char** argv)
{
	// Initialize ROS, specify name of node
	ros::init(argc, argv, "rgbd_db_recognition_test");

	// Create a handle for this node, initialize node
	ros::NodeHandle nh;

	// Create RgbdDbRecognitionTest class instance
	RgbdDbRecognitionTest rgbd_db_recognition_test(nh);

	// Initiate test
	rgbd_db_recognition_test.TestRecognition();

	ros::spin();

	return 0;
}
