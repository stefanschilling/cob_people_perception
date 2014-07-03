#include<cob_people_detection/face_normalizer.h>
#include<pcl/common/common.h>
#include<pcl/common/eigen.h>

#if !defined(PCL_VERSION_COMPARE)
	#include<pcl/common/transform.h>
#else
	#if PCL_VERSION_COMPARE(<,1,2,0)
		#include<pcl/common/transform.h>
	#endif
#endif

#include<fstream>

#include<limits>

using namespace cv;

void FaceNormalizer::init(FNConfig& i_config)
{
  std::string def_classifier_directory=     "/opt/ros/groovy/share/OpenCV/";
  this->init(def_classifier_directory,"",i_config,0,false,false);
}

void FaceNormalizer::init(std::string i_classifier_directory,FNConfig& i_config)
{
  this->init(i_classifier_directory,"",i_config,0,false,false);
}

//call initialize function with set default values
// -> no debug
// -> no storage of results
// -> epoch ctr set to zero
// -> classifier directory set to standard opencv installation dir
void FaceNormalizer::init()
{

  // set default values for face normalization
  FNConfig def_config;
  def_config.eq_ill=  true;
  def_config.align=   false;
  def_config.resize=  true;
  def_config.cvt2gray=true;
  def_config.extreme_illumination_condtions=false;

  std::string def_classifier_directory=   "/opt/ros/groovy/share/OpenCV/"  ;

  this->init(def_classifier_directory,"",def_config,0,false,false);
}



void FaceNormalizer::init(std::string i_classifier_directory,std::string i_storage_directory,FNConfig& i_config,int i_epoch_ctr,bool i_debug,bool i_record_scene)
{
  classifier_directory_=i_classifier_directory;
  storage_directory_=   i_storage_directory;
  config_=              i_config;
  epoch_ctr_=            i_epoch_ctr;
  debug_=               i_debug;
  record_scene_=        i_record_scene;

  if (config_.align)
  {
    std::string eye_r_path,eye_path,eye_l_path,nose_path,mouth_path;
    eye_r_path=    classifier_directory_+ "haarcascades/haarcascade_mcs_righteye.xml";
    eye_l_path=    classifier_directory_+ "haarcascades/haarcascade_mcs_lefteye.xml";
    nose_path=     classifier_directory_+ "haarcascades/haarcascade_mcs_nose.xml";

    eye_r_cascade_=(CvHaarClassifierCascade*) cvLoad(eye_r_path.c_str(),0,0,0);
    eye_r_storage_=cvCreateMemStorage(0);

    eye_l_cascade_=(CvHaarClassifierCascade*) cvLoad(eye_l_path.c_str(),0,0,0);
    eye_l_storage_=cvCreateMemStorage(0);

    nose_cascade_=(CvHaarClassifierCascade*) cvLoad(nose_path.c_str(),0,0,0);
    nose_storage_=cvCreateMemStorage(0);

    //intrinsics
    cv::Vec2f focal_length;
    cv::Point2f pp;
    focal_length[0]=524.90160178307269 ;
    focal_length[1]=525.85226379335393;
    pp.x=312.13543361773458;
    pp.y=254.73474482242005;
    cam_mat_=(cv::Mat_<double>(3,3) << focal_length[0] , 0.0 , pp.x  , 0.0 , focal_length[1] , pp.y  , 0.0 , 0.0 , 1);
    dist_coeffs_=(cv::Mat_<double>(1,5) << 0.25852454045259377, -0.88621162461930914, 0.0012346117737001144, 0.00036377459304633028, 1.0422813597203011);
  }

  initialized_=true;
}


FaceNormalizer::~FaceNormalizer()
{
  if(config_.align)
  {
	cvReleaseHaarClassifierCascade(&nose_cascade_);
	cvReleaseMemStorage(&nose_storage_);
	cvReleaseHaarClassifierCascade(&eye_l_cascade_);
	cvReleaseMemStorage(&eye_l_storage_);
	cvReleaseHaarClassifierCascade(&eye_r_cascade_);
	cvReleaseMemStorage(&eye_r_storage_);
  }
};

bool FaceNormalizer::isolateFace(cv::Mat& RGB,cv::Mat& XYZ)
{
  cv::Vec3f middle_pt=XYZ.at<cv::Vec3f>(round(XYZ.rows/2),round(XYZ.cols/2));

  float background_threshold=middle_pt[2]+0.25;

  eliminate_background(RGB,XYZ,background_threshold);
  interpolate_head(RGB, XYZ);
  return true;
}

bool FaceNormalizer::recordFace(cv::Mat&RGB,cv::Mat& XYZ)
{
  std::string filepath=storage_directory_;
  filepath.append("scene");
  save_scene(RGB,XYZ,filepath);
}

bool FaceNormalizer::synthFace(cv::Mat &RGB,cv::Mat& XYZ, cv::Size& norm_size, std::string training_path, int& img_count, int& step_size, int& step_no)
{
	//isolate face?
	//isolateFace();
	//cv::imshow ("received img", RGB);
	//cv::waitKey();
	bool valid = true; // Flag only returned true if all steps have been completed successfully

	//epoch_ctr_++;
	//geometric normalization
	if(config_.align)
	{
		cv::Mat GRAY,GRAY2;
		cv::cvtColor(RGB,GRAY,CV_RGB2GRAY);
		//cv::imshow ("img after gray", GRAY);
		// remove background
		//isolateFace(GRAY,XYZ);

		//adjust img and XYZ mat to size which their normalized versions will have
		//TODO
		//input images are currently square, so this could be done in one step
		// - any plans to change from square head/face rectangles?
		if( GRAY.rows&(2) != 0 )
		{
			XYZ=XYZ(cv::Rect(0,0,XYZ.cols,XYZ.rows-1));
			GRAY=GRAY(cv::Rect(0,0,GRAY.cols,GRAY.rows-1));
		}
		if( GRAY.cols&(2) != 0 )
		{
			XYZ=XYZ(cv::Rect(0,0,XYZ.cols-1,XYZ.rows));
			GRAY=GRAY(cv::Rect(0,0,GRAY.cols-1,GRAY.rows));
		}
		//dont normalize here, normalization works on histograms and is used for face images for best end-results.
		//normalize_radiometry(GRAY);
		//cv::imshow ("gray after radiometry", GRAY);
		//cv::waitKey();

		if(!synth_head_poses(GRAY,XYZ,training_path,img_count,norm_size,step_size,step_no))
		{
			std::cout<< "synth failed \n";
			return false;
		}
		//std::cout << "synth succesful, new img count: " << img_count << "\n";
		valid = true;
	}
	return valid;
}

bool FaceNormalizer::normalizeFace( cv::Mat& RGB, cv::Mat& XYZ, cv::Size& norm_size, cv::Mat& DM)
{
  bool valid=true;
  valid =normalizeFace(RGB,XYZ,norm_size);
  XYZ.copyTo(DM);
  //reducing z coordinate and performing whitening transformation
  create_DM(DM,DM);
  return valid;
}


bool FaceNormalizer::normalizeFace( cv::Mat& img,cv::Mat& depth,cv::Size& norm_size)
{

  norm_size_=norm_size;
  input_size_=cv::Size(img.cols,img.rows);

  bool valid = true; // Flag only returned true if all steps have been completed successfully

  epoch_ctr_++;

  //geometric normalization
  if(config_.align)
  {
    if(!normalize_geometry_depth(img,depth)) valid=false;
  }

  if(config_.cvt2gray)
  {
    if(img.channels()==3)cv::cvtColor(img,img,CV_RGB2GRAY);
  }

  if(config_.eq_ill)
  {
    // radiometric normalization
    if(!normalize_radiometry(img)) valid=false;
    if(debug_)dump_img(img,"radiometry");
  }

  if(debug_ && valid)dump_img(img,"geometry");

  if(config_.resize)
  {
  //resizing
  cv::resize(img,img,norm_size_,0,0);
  cv::resize(depth,depth,norm_size_,0,0);
  }

  if(debug_)dump_img(img,"size");

  epoch_ctr_++;
  normalize_img_type(img,img);
  return valid;
}


bool FaceNormalizer::normalizeFace( cv::Mat& img,cv::Size& norm_size)
{

  if(!initialized_)
  {
    std::cout<<"[FaceNormalizer] not initialized - use init() first"<<std::endl;
  }
  // set members to current values
  norm_size_=norm_size;
  input_size_=cv::Size(img.cols,img.rows);

  bool valid = true; // Flag only returned true if all steps have been completed successfully

  if(config_.cvt2gray)
  {
  if(img.channels()==3)cv::cvtColor(img,img,CV_RGB2GRAY);
  }

  if(config_.align)
  {
    std::cout<<"[FaceNormalizer] geometrical normalization only with 3D data"<<std::endl;
  //geometric normalization for rgb image only disabled
  }

  if(config_.eq_ill)
  {
  // radiometric normalization
  if(!normalize_radiometry(img)) valid=false;
  if(debug_)dump_img(img,"1_radiometry");
  }

  if(config_.resize)
  {
  //resizing
  cv::resize(img,img,norm_size_,0,0);
  if(debug_)dump_img(img,"2_resized");
  }

  epoch_ctr_++;
  normalize_img_type(img,img);
  return valid;
}

bool FaceNormalizer::normalize_radiometry(cv::Mat& img)
{
  
  if(config_.extreme_illumination_condtions==true)GammaDoG(img);
  else  GammaDCT(img);
  return true;
}

void FaceNormalizer::extractVChannel( cv::Mat& img,cv::Mat& V)
{
  if(debug_)cv::cvtColor(img,img,CV_RGB2HSV);
  else cv::cvtColor(img,img,CV_BGR2HSV);

  std::vector<cv::Mat> channels;
  cv::split(img,channels);
  channels[2].copyTo(V);

  if(debug_)cv::cvtColor(img,img,CV_HSV2RGB);
  else cv::cvtColor(img,img,CV_HSV2BGR);

  return;

}

void FaceNormalizer::subVChannel(cv::Mat& img,cv::Mat& V)
{

   if(debug_)cv::cvtColor(img,img,CV_RGB2HSV);
  else cv::cvtColor(img,img,CV_BGR2HSV);

  std::vector<cv::Mat> channels;
  cv::split(img,channels);
  channels[2]=V;
  cv::merge(channels,img);

  if(debug_)cv::cvtColor(img,img,CV_HSV2RGB);
  else cv::cvtColor(img,img,CV_HSV2BGR);

  return;
}



void FaceNormalizer::GammaDoG(cv::Mat& input_img)
{
  cv::Mat img=cv::Mat(input_img.rows,input_img.cols,CV_8UC1);
  if(input_img.channels()==3)
  {
    extractVChannel(input_img,img);
    std::cout<<"extracting"<<std::endl;
  }
  else
  {
    img=input_img;
  }


  img.convertTo(img,CV_32FC1);

  // gamma correction
  cv::pow(img,0.2,img);
  //dog
  cv::Mat g2,g1;
  cv::GaussianBlur(img,g1,cv::Size(9,9),1);
  cv::GaussianBlur(img,g2,cv::Size(9,9),2);
  cv::subtract(g2,g1,img);
  //cv::normalize(img,img,0,255,cv::NORM_MINMAX);
  img.convertTo(img,CV_8UC1,255);
  cv::equalizeHist(img,img);

  input_img=img;
}

void FaceNormalizer::GammaDCT(cv::Mat& input_img)
{
  cv::Mat img=cv::Mat(input_img.rows,input_img.cols,CV_8UC1);
  if(input_img.channels()==3)
  {
    extractVChannel(input_img,img);
  }
  else
  {
    img=input_img;
  }

  // Dct conversion on logarithmic image
  if( img.rows&2!=0 )
  {
    img=img(cv::Rect(0,0,img.cols,img.rows-1));
    input_img=img(cv::Rect(0,0,img.cols,img.rows-1));
  }
  if( img.cols&2!=0 )
  {
    img=img(cv::Rect(0,0,img.cols-1,img.rows));
    input_img=img(cv::Rect(0,0,img.cols-1,img.rows));
  }

  cv::equalizeHist(img,img);
  img.convertTo(img,CV_32FC1);
  cv::Scalar mu,sigma;
  cv::meanStdDev(img,mu,sigma);

  double C_00=log(mu.val[0])*sqrt(img.cols*img.rows);

//----------------------------
  cv::pow(img,0.2,img);
  cv::Mat imgdummy;
  img.convertTo(imgdummy,CV_8UC1);
  cv::dct(img,img);

  //---------------------------------------
  img.at<float>(0,0)=C_00;
  img.at<float>(0,1)/=10;
  img.at<float>(0,2)/=10;

  img.at<float>(1,0)/=10;
  img.at<float>(1,1)/=10;

  //--------------------------------------

  cv::idct(img,img);
  cv::normalize(img,img,0,255,cv::NORM_MINMAX);

  img.convertTo(img,CV_8UC1);
  //cv::blur(img,img,cv::Size(3,3));

  if(input_img.channels()==3)
  {
    subVChannel(input_img,img);
  }
  else
  {
    input_img=img;
  }
}

bool FaceNormalizer::synth_head_poses(cv::Mat& img,cv::Mat& depth, std::string training_path, int& img_count,cv::Size& norm_size, int& step_size, int& step_no)
{
	//variables for face detection
	double faces_increase_search_scale = 1.1;		// The factor by which the search window is scaled between the subsequent scans
	int faces_drop_groups = 2;					// Minimum number (minus 1) of neighbor rectangles that makes up an object.
	int faces_min_search_scale_x = 20;			// Minimum search scale x
	int faces_min_search_scale_y = 20;			// Minimum search scale y
	//std::string faceCascadePath = ros::package::getPath("cob_people_detection") + "/common/files/" + "haarcascades/haarcascade_frontalface_alt2.xml";
	std::string faceCascadePath = "/home/stefan/git/cob_people_perception/cob_people_detection/common/files/haarcascades/haarcascade_frontalface_alt2.xml";
	CvHaarClassifierCascade* m_face_cascade = (CvHaarClassifierCascade*)cvLoad(faceCascadePath.c_str(), 0, 0, 0 );	//"ConfigurationFiles/haarcascades/haarcascade_frontalface_alt2.xml", 0, 0, 0 );
	CvMemStorage* m_storage = cvCreateMemStorage(0);
	IplImage imgPtr = (IplImage)img;

	//add source img if face is found in it with required face.
	//face image is normalized and resized as in normalizeFace
	CvSeq* faces = cvHaarDetectObjects(&imgPtr,	m_face_cascade,	m_storage, faces_increase_search_scale, faces_drop_groups, CV_HAAR_DO_CANNY_PRUNING, cv::Size(faces_min_search_scale_x, faces_min_search_scale_y));
	if (faces->total ==1)
	{
		cv::Rect* face = (cv::Rect*)cvGetSeqElem(faces, 0);
		//exclude faces that are too small for the head bounding box
		if (face->width > 0.4*img.cols && face->height > 0.4*img.rows)
		{
			cv::Mat synth_img;
			img(cv::Rect(face->x,face->y, face->width,face->height)).copyTo(synth_img);
			cv::Mat synth_dm = depth(cv::Rect(face->x,face->y, face->width,face->height));
			normalize_radiometry(synth_img);
			//TODO
			//test resizing vs roi around nose used in other functions of face_normalizer
			// DONE. Using ROI has negative influence on recognition.
			cv::resize(synth_img,synth_img,norm_size,0,0);
			cv::resize(synth_dm,synth_dm,norm_size,0,0);
			//save resulting image and depth mat
			std::ostringstream synth_id_str;
			synth_id_str << img_count;
			std::string synth_id(synth_id_str.str());
			std::string synth_image_path = training_path + "synth_img_test/" + synth_id + ".bmp";
			std::string synth_depth_path = training_path + "synth_depth_test/" + synth_id + ".xml";
			//std::cout << "imwrite to: " << synth_image_path << std::endl;
			//std::cout << "dmwrite to: " << synth_depth_path << std::endl;
			cv::imwrite(synth_image_path, synth_img);
			cv::FileStorage fs(synth_depth_path, FileStorage::WRITE);
			fs << "depthmap" << synth_dm;
			fs.release();
			img_count++;
			std::cout << "added source img and depth\n";
		}
	}

	// feature detection runs better on a normalized image
	cv::Mat norm_head_img;
	img.copyTo(norm_head_img);
	normalize_radiometry(norm_head_img);

	// detect features
	if(!features_from_color(norm_head_img))
	{
		std::cout <<"features in color missing"<<std::endl;
		return false;
	}
	if(!features_from_depth(depth))
	{
		std::cout <<"features in depth missing"<<std::endl;
		return false;
	}
	int img_added=0;

	Eigen::Vector3f temp,x_new,y_new,z_new,lefteye,nose,righteye,eye_middle;

	nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
	lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
	righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;
	eye_middle=lefteye+((righteye-lefteye)*0.5);

	//TZ coordinates based on nose tip to middle of eyes vector
	//x_new<<f_det_xyz_.righteye.x-f_det_xyz_.lefteye.x,f_det_xyz_.righteye.y-f_det_xyz_.lefteye.y,f_det_xyz_.righteye.z-f_det_xyz_.lefteye.z;
	//temp<<f_det_xyz_.nose.x-eye_middle[0],f_det_xyz_.nose.y-eye_middle[1],(f_det_xyz_.nose.z-eye_middle[2]);
	//z_new=x_new.cross(temp);
	//x_new.normalize();
	//z_new.normalize();
	//y_new=x_new.cross(z_new);

	//SS coordinates, z vector direction = vector to nose
	z_new<<nose[0],nose[1],nose[2];
	z_new.normalize();
	//vectors of plane
	Eigen::Vector3f x_plane, y_plane,eye_to_eye,temporary;;
	x_plane<<nose[2],nose[0],-nose[1];
	y_plane=z_new.cross(x_plane);
	//distance of plane to origin
	double d = nose[0]*nose[0]+nose[1]*nose[1]+nose[2]*nose[2];
	d = sqrt(d);

	//eye-to-eye vector
	eye_to_eye<<f_det_xyz_.righteye.x-f_det_xyz_.lefteye.x,f_det_xyz_.righteye.y-f_det_xyz_.lefteye.y,f_det_xyz_.righteye.z-f_det_xyz_.lefteye.z;
	//move vector to start from nose
	eye_to_eye += nose;

	d = eye_to_eye[0]*z_new[0]+eye_to_eye[1]*z_new[1]+eye_to_eye[2]*z_new[2]- d;
	//Point on Plane from distance and z_new
	temporary = eye_to_eye + d * z_new;
	//new x vector = nose to point on plane
	x_new = nose - temporary;
	x_new.normalize();

	//perpendicular 3rd vector for y
	y_new = x_new.cross(z_new);
	y_new.normalize();

	//show base vectors...
	//std::cout << "x_plane: " << x_plane << "\n eye_to_eye + nose: " << eye_to_eye << "\n distance of vector end to plane: " << d << "\n";
	//std::cout << "new base vectors: \n z " << z_new << "\n y " << y_new << "\n x " << x_new << "\n";

	if(y_new[1]<0) y_new*=-1;

	Eigen::Vector3f origin;
	origin=nose;

	Eigen::Affine3f T_norm, T_rot;
	//pcl::getTransformationFromTwoUnitVectorsAndOrigin(y_new,z_new,origin,T_norm);
	T_norm= pcl::getTransformation(float(-nose[0]), float(-nose[1]), float(-nose[2]), 0.0, 0.0, 0.0);
	// viewing offset of normalized perspective
	double view_offset=nose[2];
	view_offset = nose[2];
	Eigen::Translation<float,3> translation=Eigen::Translation<float,3>(0, 0, view_offset);

	Eigen::AngleAxis<float> alpha;

	cv::Mat dmres,imgres;
	cv::Rect roi;
	cv::Mat workmat=cv::Mat(depth.rows,depth.cols,CV_32FC3);

	//number and distance between poses
	int N=step_no*8;
	float rot_step = step_size;
	//std::cout<<"Synthetic POSES"<<std::endl;

	if (N>0)
		{
		for(int i=0;i<N;i++)
		{
			//std::cout << "synth pose no: " << i ;
			Eigen::Vector3f xy_new = x_new+y_new;
			Eigen::Vector3f yx_new = x_new-y_new;
			if(i>=0&&i<N/8)alpha=Eigen::AngleAxis<float>(((float)(i+1)*rot_step)*M_PI, x_new);
			else if(i>=N/8&&i<2*N/8)alpha=Eigen::AngleAxis<float>((((float)(i+1)-N/8)*rot_step)*M_PI, y_new);
			else if(i>=2*N/8&&i<=3*N/8)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-2*N/8)*rot_step)*M_PI, xy_new);
			else if(i>=3*N/8&&i<=4*N/8)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-3*N/8)*rot_step)*M_PI, yx_new);
			else if(i>=4*N/8&&i<=5*N/8)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-4*N/8)*-rot_step)*M_PI, x_new);
			else if(i>=5*N/8&&i<=6*N/8)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-5*N/8)*-rot_step)*M_PI, y_new);
			else if(i>=6*N/8&&i<=7*N/8)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-6*N/8)*-rot_step)*M_PI, xy_new);
			else if(i>=7*N/8&&i<=N)alpha=Eigen::AngleAxis<float> ((((float)(i+1)-7*N/8)*-rot_step)*M_PI, yx_new);

			// ----- artificial head pose rotation
			T_rot.setIdentity();
			T_rot=alpha*T_rot;

			dmres=cv::Mat::zeros(480,640,CV_32FC3);
			if(img.channels()==3)imgres=cv::Mat::zeros(480,640,CV_8UC3);
			if(img.channels()==1)imgres=cv::Mat::zeros(480,640,CV_8UC1);

			depth.copyTo(workmat);
			cv::Vec3f* ptr=workmat.ptr<cv::Vec3f>(0,0);
			Eigen::Vector3f pt;

			for(int j=0;j<img.total();j++)
			{
				pt<<(*ptr)[0],(*ptr)[1],(*ptr)[2];
				pt=T_norm*pt;
				pt=T_rot*pt;
				pt=translation*pt;

				(*ptr)[0]=pt[0];
				(*ptr)[1]=pt[1];
				(*ptr)[2]=pt[2];
				ptr++;
			}
			//std::cout << "project pcl" << std::endl;
			projectPointCloudSynth(img,workmat,imgres,dmres);

			//TODO:
			//confirm if created image is recognized as face? - DONE
			//TODO:
			//decide if we want to detect actual face area from result picture or
			//use a roi determined by geometrical position of eyes and nose after transform
			// - DONE. Using detected face in created image produced better results

			//detect face in rotated head image.
			//normalize detections of sufficient size as in normalizeFace
			bool detect_face =true;
			//std::cout << "detect face" << std::endl;
			if (detect_face)
			{
				IplImage imgPtr = (IplImage)imgres;

				CvSeq* faces = cvHaarDetectObjects(&imgPtr,	m_face_cascade,	m_storage, faces_increase_search_scale, faces_drop_groups, CV_HAAR_DO_CANNY_PRUNING, cv::Size(faces_min_search_scale_x, faces_min_search_scale_y));
				//std::cout << "Face in synth image: " << faces->total << std::endl;
				if (faces->total ==1)
				{

					cv::Rect* face = (cv::Rect*)cvGetSeqElem(faces, 0);
					//exclude faces that are too small for the head bounding box
					if (face->width > 0.4*img.cols && face->height > 0.4*img.rows)
					{
						cv::Mat synth_img=imgres(cv::Rect(face->x,face->y, face->width,face->height));
						cv::Mat synth_dm = dmres(cv::Rect(face->x,face->y, face->width,face->height));
						normalize_radiometry(synth_img);
						cv::resize(synth_img,synth_img,norm_size,0,0);
						cv::resize(synth_dm,synth_dm,norm_size,0,0);
						//save resulting image and depth mat
						std::ostringstream synth_id_str;
						synth_id_str << img_count+img_added;
						std::string synth_id(synth_id_str.str());
						std::string synth_image_path = training_path + "synth_img_test/" + synth_id + ".bmp";
						std::string synth_depth_path = training_path + "synth_depth_test/" + synth_id + ".xml";
						//std::cout << "imwrite to: " << synth_image_path << std::endl;
						//std::cout << "dmwrite to: " << synth_depth_path << std::endl;
						cv::imwrite(synth_image_path, synth_img);
						cv::FileStorage fs(synth_depth_path, FileStorage::WRITE);
						fs << "depthmap" << synth_dm;
						fs.release();
						img_added++;
						//std::cout << "cut out face size: " << face_img.size() << std::endl;
						//std::cout << face->height << "," << face->width<<","<<face->x<<","<<face->y <<std::endl;
						//cv::imshow("Face Detection",face_img);
						//cv::waitKey();
					}
				}
			}
		}
	}
	img_count = img_count+img_added;
	std::cout << "additional images created for training: " << img_added << std::endl;
	std::cout << "new image total: " << img_count << std::endl;
	return true;
}

bool FaceNormalizer::projectPointCloud(cv::Mat& img, cv::Mat& depth, cv::Mat& img_res, cv::Mat& depth_res)
{
	int channels=img.channels();

	cv::Mat pc_xyz,pc_rgb;
	depth.copyTo(pc_xyz);
	img.copyTo(pc_rgb);

	//make point_list
	if(pc_xyz.rows>1 && pc_xyz.cols >1)
	{
		pc_xyz=pc_xyz.reshape(3,1);
	}

   //project 3d points to virtual camera
   //TODO temporary triy
   //cv::Mat pc_proj(pc_xyz.rows*pc_xyz.cols,1,CV_32FC2);
   cv::Mat pc_proj(pc_xyz.cols,1,CV_32FC2);

   cv::Vec3f rot=cv::Vec3f(0.0,0.0,0.0);
   cv::Vec3f trans=cv::Vec3f(0.0,0.0,0.0);
   cv::Size sensor_size=cv::Size(640,480);
   cv::projectPoints(pc_xyz,rot,trans,cam_mat_,dist_coeffs_,pc_proj);

   cv::Vec3f* pc_ptr=pc_xyz.ptr<cv::Vec3f>(0,0);
   cv::Vec2f* pc_proj_ptr=pc_proj.ptr<cv::Vec2f>(0,0);
   int ty,tx;

   if(channels==3)
   {
    cv::add(img_res,0,img_res);
    cv::add(depth_res,0,depth_res);
   // assign color values to calculated image coordinates
   cv::Vec3b* pc_rgb_ptr=pc_rgb.ptr<cv::Vec3b>(0,0);


   cv::Mat occ_grid=cv::Mat::ones(sensor_size,CV_32FC3);
   cv::Mat img_cum=cv::Mat::zeros(sensor_size,CV_32FC3);
   cv::Vec3f occ_inc=cv::Vec3f(1,1,1);
   for(int i=0;i<pc_proj.rows;++i)
     {
       cv::Vec2f txty=*pc_proj_ptr;
       tx=(int)round(txty[0]);
       ty=(int)round(txty[1]);


       if (ty>1 && tx>1 && ty<sensor_size.height-1 && tx<sensor_size.width-1 && !isnan(ty) && !isnan(tx) )
       {
            img_cum.at<cv::Vec3b>(ty,tx)+=(*pc_rgb_ptr);
            img_cum.at<cv::Vec3f>(ty+1,tx)+=(*pc_rgb_ptr);
            img_cum.at<cv::Vec3f>(ty-1,tx)+=(*pc_rgb_ptr);
            img_cum.at<cv::Vec3f>(ty,tx-1)+=(*pc_rgb_ptr);
            img_cum.at<cv::Vec3f>(ty,tx+1)+=(*pc_rgb_ptr);

            occ_grid.at<cv::Vec3f>(ty,tx)+=  occ_inc;
            occ_grid.at<cv::Vec3f>(ty+1,tx)+=occ_inc;
            occ_grid.at<cv::Vec3f>(ty-1,tx)+=occ_inc;
            occ_grid.at<cv::Vec3f>(ty,tx+1)+=occ_inc;
            occ_grid.at<cv::Vec3f>(ty,tx-1)+=occ_inc;

            depth_res.at<cv::Vec3f>(ty,tx)=((*pc_ptr));
       }
       pc_rgb_ptr++;
       pc_proj_ptr++;
       pc_ptr++;
      }
   img_cum=img_cum / occ_grid;
   img_cum.convertTo(img_res,CV_8UC3);
   }


   if(channels==1)
   {
   // assign color values to calculated image coordinates
    cv::add(img_res,0,img_res);
    cv::add(depth_res,0,depth_res);
   unsigned char* pc_rgb_ptr=pc_rgb.ptr<unsigned char>(0,0);

   cv::Mat occ_grid=cv::Mat::ones(sensor_size,CV_32FC1);
   cv::Mat img_cum=cv::Mat::zeros(sensor_size,CV_32FC1);
   cv::Mat occ_grid2=cv::Mat::ones(sensor_size,CV_32FC1);
   for(int i=0;i<pc_proj.rows;++i)
     {
       cv::Vec2f txty=*pc_proj_ptr;
       tx=(int)round(txty[0]);
       ty=(int)round(txty[1]);


       if (ty>1 && tx>1 && ty<sensor_size.height-1 && tx<sensor_size.width-1 && !isnan(ty) && !isnan(tx) )
       {
            //if((depth_map.at<cv::Vec3f>(ty,tx)[2]==0) || (depth_map.at<cv::Vec3f>(ty,tx)[2]>(*pc_ptr)[2]))
            //{
            img_res.at<unsigned char>(ty,tx)=(*pc_rgb_ptr);
            occ_grid2.at<float>(ty,tx)=0.0;
            img_cum.at<float>(ty+1,tx)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty-1,tx)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty,tx-1)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty,tx+1)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty+1,tx+1)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty-1,tx-1)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty-1,tx+1)+=(*pc_rgb_ptr);
            img_cum.at<float>(ty+1,tx-1)+=(*pc_rgb_ptr);

            occ_grid.at<float>(ty,tx)+=  1;
            occ_grid.at<float>(ty+1,tx)+=1;
            occ_grid.at<float>(ty-1,tx)+=1;
            occ_grid.at<float>(ty,tx+1)+=1;
            occ_grid.at<float>(ty,tx-1)+=1;
            occ_grid.at<float>(ty+1,tx+1)+=1;
            occ_grid.at<float>(ty-1,tx-1)+=1;
            occ_grid.at<float>(ty-1,tx+1)+=1;
            occ_grid.at<float>(ty+1,tx-1)+=1;

            depth_res.at<cv::Vec3f>(ty,tx)=((*pc_ptr));
       }
       pc_rgb_ptr++;
       pc_proj_ptr++;
       pc_ptr++;
      }

   occ_grid=occ_grid;
   img_cum=img_cum / (occ_grid.mul(occ_grid2)-1);
   img_cum.convertTo(img_cum,CV_8UC1);
   cv::add(img_res,img_cum,img_res);
   }



//   // refinement
//
//   //unsigned char* img_res_ptr=img_res.ptr<unsigned char>(0,0);
//   cv::Mat img_res2=cv::Mat::zeros(img_res.rows,img_res.cols,img_res.type());
//   for(int i=1;i<img_res.total();i++)
//   {
//     if(img_res.at<unsigned char>(i)==0 && img_res.at<unsigned char>(i-1)!=0)
//     {
//       //calc position
//       cv::Vec2f pos= pc_proj.at<cv::Vec2f>(i-1);
//       std::cout<<"POSITION"<<pos<<std::endl;
//
//       unsigned char val=pc_rgb.at<unsigned char>(round(pos[0]),round(pos[1]+1));
//       img_res2.at<unsigned char>(i)=val;
//     }
//   }
//
//   cv::imshow("IMG_RES",img_res2);
//   cv::waitKey(0);
//
   return true;
}
// construct path to depth and color image data, read/load xml and bmp.

bool FaceNormalizer::projectPointCloudSynth(cv::Mat& img, cv::Mat& depth, cv::Mat& img_res_fltr, cv::Mat& depth_res)
{
	//std::cout << "received: " <<std::endl;
	//std::cout << img.size() << std::endl;
	//std::cout << depth.size() << std::endl;
	int channels=img.channels();

	cv::Mat pc_xyz,pc_rgb;
	depth.copyTo(pc_xyz);
	img.copyTo(pc_rgb);

	//make point_list
	if(pc_xyz.rows>1 && pc_xyz.cols >1)
	{
		pc_xyz=pc_xyz.reshape(3,1);
	}

	//project 3d points to virtual camera
	cv::Mat pc_proj(pc_xyz.cols,1,CV_32FC2);
	cv::Vec3f rot=cv::Vec3f(0.0,0.0,0.0);
	cv::Vec3f trans=cv::Vec3f(0.0,0.0,0.0);
	cv::Size sensor_size=cv::Size(640,480);
	cv::projectPoints(pc_xyz,rot,trans,cam_mat_,dist_coeffs_,pc_proj);
	cv::Vec3f* pc_ptr=pc_xyz.ptr<cv::Vec3f>(0,0);
	cv::Vec2f* pc_proj_ptr=pc_proj.ptr<cv::Vec2f>(0,0);
	int ty,tx;
	if(channels==1)
	{
		unsigned char* pc_rgb_ptr=pc_rgb.ptr<unsigned char>(0,0);

		// Assign Color, no spread, singular assignment (lowest z-value gets the point)
		cv::Mat img_res_zwin=cv::Mat::zeros(640,480,CV_8UC1);
		cv::Mat depth_res_zwin=cv::Mat::zeros(640,480,CV_32FC3);
		std::map <std::pair<int,int> ,std::vector<std::pair<cv::Vec3f, unsigned char > > > proj_map;
		std::vector<std::vector<cv::Vec3f> > proj_points;
		for(int i=0;i<pc_proj.rows;++i)
		{
			cv::Vec2f txty=*pc_proj_ptr;
			tx=(int)round(txty[0]);
			ty=(int)round(txty[1]);
			if (ty>1 && tx>1 && ty<sensor_size.height-1 && tx<sensor_size.width-1 && !isnan(ty) && !isnan(tx) )
			{
				// fill map with point coordinates and color values, key: x,y values they are projected to
				cv::Vec3f txyz = *pc_ptr;
				proj_map[std::pair<int, int>(tx,ty)].push_back(std::make_pair(txyz, *pc_rgb_ptr));
			}
			pc_rgb_ptr++;
			pc_proj_ptr++;
			pc_ptr++;
		}
		//std::cout << "points on sensor added to proj map pairs: " << proj_map.size() << std::endl;
		//assign values based on map
		for(std::map <std::pair<int,int>,std::vector<std::pair<cv::Vec3f, unsigned char > > >::iterator it=proj_map.begin(); it!=proj_map.end(); ++it)
		{
			if( it->second.size() >0)
			{
				// add closest depth point for each projected point to result depth matrix
				// add color of closest depth point to result color matrix
				int a=4;
				for (int i = 0; i < it->second.size(); i++)
				{
					// pick closest point, restricted to points belonging to facial area (remove background, threshhold picked at random)
					if (it->second[i].first[2] < a)
					{
						a = it->second[i].first[2];
						img_res_zwin.at<unsigned char>(it->first.second,it->first.first) = it->second[i].second;
						depth_res_zwin.at<Vec3f>(it->first.second,it->first.first) = it->second[i].first;
						// std::cout << "assigned new value to contested point at " << it->first.second << " , " << it->first.first << " z value winner: " << it->second[i].first[2] << std::endl;
					}
				}
			}
		}

		//std::cout << " z-based assignment done \n ";
		proj_map.clear();

		img_res_fltr = img_res_zwin.clone();
		depth_res = depth_res_zwin.clone();
		bool filter = true;
		if (filter)
		{
			// fill in gaps for points inside of head contour, remove straggler points from color and depth matrix

			// blur to fill gaps in face image, erode and dilate to cut out any "straggler" points
			// and other parts drifting away from the face for lack of depth point cohesion
			int erosion_size = 2;
			cv::Mat contour_mat=cv::Mat::zeros(480,640,CV_8UC1);
			cv::Mat element = cv::getStructuringElement( MORPH_RECT, Size( 2*erosion_size + 1, 2*erosion_size+1 ));
			cv::GaussianBlur(img_res_zwin,contour_mat, cv::Size(3,3),0,0);
			cv::erode(contour_mat,contour_mat,element);
			cv::erode(contour_mat,contour_mat,element);
			cv::dilate(contour_mat, contour_mat, element);

			// Outer contour of dilated image
			int thresh = 5;
			int outside_contour_index=0;
			RNG rng(12345);
			cv::Mat threshold_output;
			std::vector<vector<Point> > contours;
			std::vector<Vec4i> hierarchy;

			// Detect edges using Threshold
			cv::threshold( contour_mat, threshold_output, thresh, 255, THRESH_BINARY );

			// Find contours, find outside contour (assuming it has most points)
			cv::findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
			for( int i = 0; i< contours.size(); i++ )
			{
				if (contours[i].size() > contours[outside_contour_index].size())
					outside_contour_index = i;
			}

			// Draw contours
			cv::Mat outside_drawing = cv::Mat::zeros( threshold_output.size(), CV_8UC1 );
			cv::drawContours( outside_drawing, contours, outside_contour_index, 255, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			contours.clear();
			hierarchy.clear();

			//set up kernel for gap filling. 3x3 without center pixel (since operation is only applied to empty fields)
			std::pair<int,int> kernel_array[8] = {std::make_pair(1,0), std::make_pair(1,1), std::make_pair(0,1), std::make_pair(-1,1), std::make_pair(-1,0), std::make_pair(-1,-1), std::make_pair(0,-1), std::make_pair(1,-1) };

			//fill gaps in head image contour
			//leave out some rows and columns, since the head area is (about) centered
			for (int i = 50; i<img_res_fltr.cols-50; i++)
			{
				for (int j = 50; j<img_res_fltr.rows-50; j++)
				{
					//idea: contour based filler instead,
					//working around edge of contour to fill evenly from all sides?
					if (outside_drawing.at<uchar>(i,j) > 0 && img_res_fltr.at<uchar>(i,j) == 0)
					{
						int res_color_divisor=0;
						int res_color=0;
						cv::Vec3f res_depth;
						res_depth[0]=res_depth[1]=res_depth[2]=0;
						for (int k = 0; k < 8; k++)
						{
							if (img_res_fltr.at<uchar>(i+kernel_array[k].first,j+kernel_array[k].second) !=0)
							{
								//res_depth += depth_res.at<cv::Vec3f>(i+kernel_array[k].first,j+kernel_array[k].second);
								res_color += int(img_res_fltr.at<uchar>(i+kernel_array[k].first,j+kernel_array[k].second));
								res_color_divisor++;
							}
						}
						if (res_color_divisor > 0)
						{
							img_res_fltr.at<uchar>(i,j) = res_color / res_color_divisor;
							//add point coordinates to depth with same procedure
							//depth_res.at<cv::Vec3f>(i,j)[0] = res_depth[0]/res_color_divisor;
							//depth_res.at<cv::Vec3f>(i,j)[1] = res_depth[1]/res_color_divisor;
							//depth_res.at<cv::Vec3f>(i,j)[2] = res_depth[2]/res_color_divisor;
						}
					}
					if (outside_drawing.at<uchar>(i,j) == 0)
					{
						img_res_fltr.at<uchar>(i,j) = 255;
						//TODO get .NaN without causing /0 warning?
						depth_res.at<cv::Vec3f>(i,j) = cv::Vec3f(0.0/0,0.0/0,0.0);
					}
				}
			}
		}

		cv::GaussianBlur(img_res_fltr,img_res_fltr, cv::Size(3,3),0,0);
		cv::GaussianBlur(img_res_fltr,img_res_fltr, cv::Size(3,3),0,0);

		//cv::imshow("original", img);
		//cv::imshow("eroded zwin", src_erode);
		//cv::imshow("dilated zwin", src_dilate);
		//cv::imshow("zwin", img_res_zwin);
		//cv::imshow("filtered zwin", img_res_fltr);
		//cv::waitKey();
	}

	if(channels==3)
	{
		std::cout << "please use the single channel / gray image version!\n";
		/*
		cv::add(img_res,0,img_res);
		cv::add(depth_res,0,depth_res);
		// assign color values to calculated image coordinates
		cv::Vec3b* pc_rgb_ptr=pc_rgb.ptr<cv::Vec3b>(0,0);


		cv::Mat occ_grid=cv::Mat::ones(sensor_size,CV_32FC3);
		cv::Mat img_cum=cv::Mat::zeros(sensor_size,CV_32FC3);
		cv::Vec3f occ_inc=cv::Vec3f(1,1,1);
		for(int i=0;i<pc_proj.rows;++i)
		{
			cv::Vec2f txty=*pc_proj_ptr;
			tx=(int)round(txty[0]);
			ty=(int)round(txty[1]);


			if (ty>1 && tx>1 && ty<sensor_size.height-1 && tx<sensor_size.width-1 && !isnan(ty) && !isnan(tx) )
			{
				img_cum.at<cv::Vec3b>(ty,tx)+=(*pc_rgb_ptr);
				img_cum.at<cv::Vec3f>(ty+1,tx)+=(*pc_rgb_ptr);
				img_cum.at<cv::Vec3f>(ty-1,tx)+=(*pc_rgb_ptr);
				img_cum.at<cv::Vec3f>(ty,tx-1)+=(*pc_rgb_ptr);
				img_cum.at<cv::Vec3f>(ty,tx+1)+=(*pc_rgb_ptr);

				occ_grid.at<cv::Vec3f>(ty,tx)+=  occ_inc;
				occ_grid.at<cv::Vec3f>(ty+1,tx)+=occ_inc;
				occ_grid.at<cv::Vec3f>(ty-1,tx)+=occ_inc;
				occ_grid.at<cv::Vec3f>(ty,tx+1)+=occ_inc;
				occ_grid.at<cv::Vec3f>(ty,tx-1)+=occ_inc;

				depth_res.at<cv::Vec3f>(ty,tx)=((*pc_ptr));
			}
			pc_rgb_ptr++;
			pc_proj_ptr++;
			pc_ptr++;
		}
		img_cum=img_cum / occ_grid;
		img_cum.convertTo(img_res,CV_8UC3);
		*/
	}

	return true;
}

bool FaceNormalizer::read_scene_from_training(cv::Mat& depth,cv::Mat& color,std::string path, int image_id)
{
	//std::cout<<"[FaceNormalizer]Reading from training_data from "<<path<<std::endl;
	std::string depth_path = path;

	path.append("img/");
	path = path + boost::lexical_cast<std::string>(image_id);
	path.append(".bmp");
	//std::cout << "img path: " << path<<std::endl;

	depth_path.append("depth/");
	depth_path = depth_path + boost::lexical_cast<std::string>(image_id);
	depth_path.append(".xml");
	//std::cout << "depth path: " << depth_path<<std::endl;

	cv::FileStorage fs(depth_path,FileStorage::READ);
	fs["depthmap"]>> depth;

	color = cv::imread(path, CV_LOAD_IMAGE_COLOR);
	fs.release();

	return true;
}

bool FaceNormalizer::frontFaceImage(cv::Mat& img,cv::Mat& depth, std::vector<float>& scores)
{
	//TODO
	// adjust for looking at camera (face-plane perpendicular to detected Nose Vector?)
	// detect features
	if(!features_from_color(img))
	{
		scores.push_back(10);
		return false;
	}
	if(debug_)dump_features(img);
	if(!features_from_depth(depth))
	{
		scores.push_back(10);
		return false;
	}
	Eigen::Vector3f temp,x_new,y_new,z_new,lefteye,nose,righteye,eye_middle;
	float score;

	nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
	lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
	righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;
	eye_middle=lefteye+((righteye-lefteye)*0.5);

	//std::cout << "Detected coordinates of features: \nNose: " << nose[0] << " " << nose[1] << " " << nose[2] << "\nLeft Eye " << lefteye[0] << " " << lefteye[1] << " " << lefteye[2] << "\nRight Eye " << righteye[0] << " " << righteye[1] << " " << righteye[2]<< "\nMiddle Eye " << eye_middle[0] << " " << eye_middle[1] << " " << eye_middle[2]<< "\n";
	score = (lefteye[1]-righteye[1])*(lefteye[1]-righteye[1]) + (lefteye[2]-righteye[2])*(lefteye[2]-righteye[2]) + (eye_middle[0] - nose [0])*(eye_middle[0] - nose [0]);
	score = sqrt(score);
	scores.push_back(score);
	// ^ score 0 for ideal image. eyes same height, same distance from camera and nose centered between eyes
}

bool FaceNormalizer::rotate_head(cv::Mat& img,cv::Mat& depth)
{

  // detect features
  if(!features_from_color(img))return false;
  if(debug_)dump_features(img);


   if(!features_from_depth(depth)) return false;

   Eigen::Vector3f temp,x_new,y_new,z_new,lefteye,nose,righteye,eye_middle;

   nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
   lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
   righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;
   eye_middle=lefteye+((righteye-lefteye)*0.5);

   x_new<<f_det_xyz_.righteye.x-f_det_xyz_.lefteye.x,f_det_xyz_.righteye.y-f_det_xyz_.lefteye.y,f_det_xyz_.righteye.z-f_det_xyz_.lefteye.z;
   temp<<f_det_xyz_.nose.x-eye_middle[0],f_det_xyz_.nose.y-eye_middle[1],(f_det_xyz_.nose.z-eye_middle[2]);
   z_new=x_new.cross(temp);
   x_new.normalize();
   z_new.normalize();
   y_new=x_new.cross(z_new);

   if(y_new[1]<0) y_new*=-1;


   Eigen::Vector3f origin;
   origin=nose;
   //origin=nose+(0.1*z_new);

   Eigen::Affine3f T_norm;
   std::cout<<"X-Axis:\n "<<x_new<<std::endl;
   std::cout<<"Y-Axis:\n "<<y_new<<std::endl;
   std::cout<<"Z-Axis:\n "<<z_new<<std::endl;

   pcl::getTransformationFromTwoUnitVectorsAndOrigin(y_new,z_new,origin,T_norm);


   // viewing offset of normalized perspective
    double view_offset=0.6;
    Eigen::Translation<float,3> translation=Eigen::Translation<float,3>(0, 0, view_offset);

    // modify T_norm by angle for nose incline compensation
    Eigen::AngleAxis<float> roll(0.0, x_new);
    //Eigen::AngleAxis<float> roll(-0.78, x_new);
    T_norm=roll*T_norm;


  Eigen::Affine3f T_rot;
  cv::Mat workmat=cv::Mat(depth.rows,depth.cols,CV_32FC3);
  Eigen::AngleAxis<float> alpha;

  cv::Mat dmres;  cv::Mat imgres;
  cv::Rect roi;
  for(int i=0;i<60;i++)
  {
  // ----- artificial head pose rotation
  if(i<20)alpha=Eigen::AngleAxis<float>((float)i*0.1*M_PI, Eigen::Vector3f(1,0,0));
  else if(i>=20&&i<40)alpha=Eigen::AngleAxis<float>(((float)i-20)*0.1*M_PI, Eigen::Vector3f(0,1,0));
  else if(i>=40&&i<=60)alpha=Eigen::AngleAxis<float> (((float)i-40)*0.1*M_PI, Eigen::Vector3f(0,0,1));

    T_rot.setIdentity();
  T_rot=alpha*T_rot;
  // ----- artificial head pose rotation

  dmres=cv::Mat::zeros(480,640,CV_32FC3);
  if(img.channels()==3)imgres=cv::Mat::zeros(480,640,CV_8UC3);
  if(img.channels()==1)imgres=cv::Mat::zeros(480,640,CV_8UC1);


  depth.copyTo(workmat);
   cv::Vec3f* ptr=workmat.ptr<cv::Vec3f>(0,0);
   Eigen::Vector3f pt;
   for(int i=0;i<img.total();i++)
   {

     pt<<(*ptr)[0],(*ptr)[1],(*ptr)[2];
     pt=T_norm*pt;
     pt=T_rot*pt;
     pt=translation*pt;

    (*ptr)[0]=pt[0];
    (*ptr)[1]=pt[1];
    (*ptr)[2]=pt[2];
     ptr++;
   }

   nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
   lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
   righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;

   lefteye=translation*T_rot*T_norm*lefteye;
   righteye=translation*T_rot*T_norm*righteye;
   nose=translation*T_rot*T_norm*nose;

   //transform norm coordinates separately to  determine roi
   cv::Point2f lefteye_uv,righteye_uv,nose_uv;
   cv::Point3f lefteye_xyz,righteye_xyz,nose_xyz;


   lefteye_xyz = cv::Point3f(lefteye[0],lefteye[1],lefteye[2]);
   righteye_xyz = cv::Point3f(righteye[0],righteye[1],righteye[2]);
   nose_xyz = cv::Point3f(nose[0],nose[1],nose[2]);

   projectPoint(lefteye_xyz,lefteye_uv);
   projectPoint(righteye_xyz,righteye_uv);
   projectPoint(nose_xyz,nose_uv);

   //determine bounding box

   float s=2;
   int dim_x=(righteye_uv.x-lefteye_uv.x)*s;
   //int off_x=((righteye_uv.x-lefteye_uv.x)*s -(righteye_uv.x-lefteye_uv.x))/2;
   //int off_y=off_x;
   int dim_y=dim_x;

   //roi=cv::Rect(round(nose_uv.x-dim_x*0.5),round(nose_uv.y-dim_y*0.5),dim_x,dim_y);
   roi=cv::Rect(round(lefteye_uv.x-dim_x*0.25),round(lefteye_uv.y-dim_y*0.25),dim_x,dim_y);

   if(img.channels()==3)cv::cvtColor(img,img,CV_RGB2GRAY);


   //filter out background

  

  projectPointCloud(img,workmat,imgres,dmres);
  //TODO temporary
  // add debug oiints
  cv::circle(imgres,nose_uv,5,cv::Scalar(255,255,255),-1,8);
  cv::circle(imgres,lefteye_uv,5,cv::Scalar(255,255,255),-1,8);
  cv::circle(imgres,righteye_uv,5,cv::Scalar(255,255,255),-1,8);
  cv::imshow("FULL",imgres);

  int key;
  key=cv::waitKey(200);
  if(key ==1048608)
  {
    std::cout<<"PAUSED--confirm usage with ENTER - continue with other key"<<std::endl;
    int enter;
    enter=cv::waitKey(0);
    if(enter==1048586)
    {
      std::cout<<"ENTER-- using current view for normalization"<<std::endl;
      break;
    }
    else
    {
      std::cout<<"CONTINUE"<<std::endl;
    }
  }
  }



  if(debug_)dump_img(imgres,"uncropped");

  if(roi.height<=1 ||roi.width<=0 || roi.x<0 || roi.y<0 ||roi.x+roi.width >imgres.cols || roi.y+roi.height>imgres.rows)
  {
    std::cout<<"[FaceNormalizer]image ROI out of limits"<<std::endl;
    return false;
  }
  imgres(roi).copyTo(img);
  //TODO TEMPorary switch ogg
  dmres(roi).copyTo(depth);
  //despeckle<unsigned char>(img,img);
  //only take central region
  img=img(cv::Rect(2,2,img.cols-4,img.rows-4));

  return true;
}

bool FaceNormalizer::normalize_geometry_depth(cv::Mat& img,cv::Mat& depth)
{

  // detect features
  if(!features_from_color(img))return false;
  if(debug_)dump_features(img);


   if(!features_from_depth(depth)) return false;

   Eigen::Vector3f temp,x_new,y_new,z_new,lefteye,nose,righteye,eye_middle;

   nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
   lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
   righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;
   eye_middle=lefteye+((righteye-lefteye)*0.5);

   x_new<<f_det_xyz_.righteye.x-f_det_xyz_.lefteye.x,f_det_xyz_.righteye.y-f_det_xyz_.lefteye.y,f_det_xyz_.righteye.z-f_det_xyz_.lefteye.z;
   temp<<f_det_xyz_.nose.x-eye_middle[0],f_det_xyz_.nose.y-eye_middle[1],(f_det_xyz_.nose.z-eye_middle[2]);
   z_new=x_new.cross(temp);
   x_new.normalize();
   z_new.normalize();
   y_new=x_new.cross(z_new);

   if(y_new[1]<0) y_new*=-1;

   Eigen::Vector3f nose_flat=nose;
   nose_flat[2]+=0.03;


   Eigen::Vector3f origin;
   origin=nose;

   Eigen::Affine3f T_norm;

   pcl::getTransformationFromTwoUnitVectorsAndOrigin(y_new,z_new,origin,T_norm);


   // viewing offset of normalized perspective
    double view_offset=0.8;
    Eigen::Translation<float,3> translation=Eigen::Translation<float,3>(0, 0, view_offset);

    // modify T_norm by angle for nose incline compensation
    Eigen::AngleAxis<float> roll(-0.78, x_new);
    //Eigen::AngleAxis<float> roll(0.0, x_new);
    T_norm=translation*roll*T_norm;

   cv::Vec3f* ptr=depth.ptr<cv::Vec3f>(0,0);
   Eigen::Vector3f pt;
   for(int i=0;i<img.total();i++)
   {
     pt<<(*ptr)[0],(*ptr)[1],(*ptr)[2];
     pt=T_norm*pt;

    (*ptr)[0]=pt[0];
    (*ptr)[1]=pt[1];
    (*ptr)[2]=pt[2];
     ptr++;
   }

   lefteye=T_norm*lefteye;
   righteye=T_norm*righteye;
   nose=T_norm*nose;

   //transform norm coordinates separately to  determine roi
   cv::Point2f lefteye_uv,righteye_uv,nose_uv;
   cv::Point3f lefteye_xyz,righteye_xyz,nose_xyz;

   lefteye_xyz = cv::Point3f(lefteye[0],lefteye[1],lefteye[2]);
   righteye_xyz = cv::Point3f(righteye[0],righteye[1],righteye[2]);
   nose_xyz = cv::Point3f(nose[0],nose[1],nose[2]);

   projectPoint(lefteye_xyz,lefteye_uv);
   projectPoint(righteye_xyz,righteye_uv);
   projectPoint(nose_xyz,nose_uv);

   //determine bounding box
   float s=3.3;
   int dim_x=(righteye_uv.x-lefteye_uv.x)*s;
   //int off_x=((righteye_uv.x-lefteye_uv.x)*s -(righteye_uv.x-lefteye_uv.x))/2;
   //int off_y=off_x;
   int dim_y=dim_x;

   cv::Rect roi=cv::Rect(round(nose_uv.x-dim_x*0.5),round(nose_uv.y-dim_y*0.5),dim_x,dim_y);

   if(img.channels()==3)cv::cvtColor(img,img,CV_RGB2GRAY);

  cv::Mat imgres;
  if(img.channels()==3)imgres=cv::Mat::zeros(480,640,CV_8UC3);
  if(img.channels()==1)imgres=cv::Mat::zeros(480,640,CV_8UC1);
  cv::Mat dmres=cv::Mat::zeros(480,640,CV_32FC3);

  projectPointCloud(img,depth,imgres,dmres);
  if(debug_)dump_img(imgres,"uncropped");

  if(roi.height<=1 ||roi.width<=0 || roi.x<0 || roi.y<0 ||roi.x+roi.width >imgres.cols || roi.y+roi.height>imgres.rows)
  {
    std::cout<<"[FaceNormalizer]image ROI out of limits"<<std::endl;
    return false;
  }
  imgres(roi).copyTo(img);
  dmres(roi).copyTo(depth);
  despeckle<unsigned char>(img,img);
  //only take central region
  img=img(cv::Rect(2,2,img.cols-4,img.rows-4));

  return true;
}



bool FaceNormalizer::features_from_color(cv::Mat& img_color)
{
  if(!detect_feature(img_color,f_det_img_.nose,FACE::NOSE))
  {
    //std::cout<<"[FaceNormalizer] detected no nose"<<std::endl;
    f_det_img_.nose.x=round(img_color.cols*0.5);
    f_det_img_.nose.y=round(img_color.rows*0.5);
    return false;
  }
  if(!detect_feature(img_color,f_det_img_.lefteye,FACE::LEFTEYE))
  {
    //std::cout<<"[FaceNormalizer] detected no eye_l"<<std::endl;
     return false;
  }
  if(!detect_feature(img_color,f_det_img_.righteye,FACE::RIGHTEYE))
  {
    //std::cout<<"[FaceNormalizer] detected no eye_r"<<std::endl;
     return false;
  }

  if(debug_)
  {
    std::cout<<"[FaceNormalizer] detected detected image features:\n";
    std::cout<<"detected lefteye "<<f_det_img_.lefteye.x<<" - " << f_det_img_.lefteye.y<<std::endl;
    std::cout<<"detected righteye "<<f_det_img_.righteye.x<<" - " << f_det_img_.righteye.y<<std::endl;
    std::cout<<"detected nose "<<f_det_img_.nose.x<<" - " << f_det_img_.nose.y<<std::endl;
    //std::cout<<"detected mouth "<<f_det_img_.mouth.x<<" - " << f_det_img_.mouth.y<<std::endl;
  }
  return true;
}

bool FaceNormalizer::features_from_depth(cv::Mat& depth)
{
  //pick 3D points from pointcloud
  f_det_xyz_.nose=      depth.at<cv::Vec3f>(f_det_img_.nose.y,f_det_img_.nose.x)               ;
  //f_det_xyz_.mouth=     depth.at<cv::Vec3f>(f_det_img_.mouth.y,f_det_img_.mouth.x)            ;
  f_det_xyz_.lefteye=   depth.at<cv::Vec3f>(f_det_img_.lefteye.y,f_det_img_.lefteye.x)      ;
  f_det_xyz_.righteye=  depth.at<cv::Vec3f>(f_det_img_.righteye.y,f_det_img_.righteye.x)   ;

  if(debug_)
  {
    std::cout<<"[FaceNormalizer] detected Coordinates of features in pointcloud:"<<std::endl;
    std::cout<<"LEFTEYE: "<<f_det_xyz_.lefteye.x<<" "<<f_det_xyz_.lefteye.y<<" "<<f_det_xyz_.lefteye.z<<std::endl;
    std::cout<<"RIGTHEYE: "<<f_det_xyz_.righteye.x<<" "<<f_det_xyz_.righteye.y<<" "<<f_det_xyz_.righteye.z<<std::endl;
    std::cout<<"NOSE: "<<f_det_xyz_.nose.x<<" "<<f_det_xyz_.nose.y<<" "<<f_det_xyz_.nose.z<<std::endl;
    //std::cout<<"MOUTH: "<<f_det_xyz_.mouth.x<<" "<<f_det_xyz_.mouth.y<<" "<<f_det_xyz_.mouth.z<<std::endl;
  }
  if(!f_det_xyz_.valid()) return false;

  return true;
}

bool FaceNormalizer::detect_feature(cv::Mat& img,cv::Point2f& coords,FACE::FEATURE_TYPE type)
{

  //  determine scale of search pattern
  double scale=img.cols/160.0;

  CvSeq* seq;
  cv::Vec2f offset;

  switch(type)
  {

    case FACE::NOSE:
  {
    offset =cv::Vec2f(0,0);
    IplImage ipl_img=(IplImage)img;
     seq=cvHaarDetectObjects(&ipl_img,nose_cascade_,nose_storage_,1.1,1,0,cv::Size(20*scale,20*scale));
     //seq=cvHaarDetectObjects(&ipl_img,nose_cascade_,nose_storage_,1.3,2,CV_HAAR_DO_CANNY_PRUNING,cv::Size(15*scale,15*scale));
     break;
  }

    case FACE::LEFTEYE:
  {
    offset[0]=0;
    offset[1]=0;
    cv::Mat sub_img=img.clone();
    sub_img=sub_img(cvRect(0,0,f_det_img_.nose.x,f_det_img_.nose.y));
    IplImage ipl_img=(IplImage)sub_img;
     seq=cvHaarDetectObjects(&ipl_img,eye_l_cascade_,eye_l_storage_,1.1,1,0,cvSize(20*scale,10*scale));
     break;
  }

    case FACE::RIGHTEYE:
  {
    offset[0]=((int)f_det_img_.nose.x);
    offset[1]=0;
    cv::Mat sub_img=img.clone();
    sub_img=sub_img(cvRect(f_det_img_.nose.x,0,img.cols-f_det_img_.nose.x-1,f_det_img_.nose.y));
    IplImage ipl_img=(IplImage)sub_img;
     seq=cvHaarDetectObjects(&ipl_img,eye_r_cascade_,eye_r_storage_,1.1,1,0,cvSize(20*scale,10*scale));
     break;
  }

    case FACE::MOUTH:
  {
  //  offset[0]=0;
  //  offset[1]=(int)f_det_img_.nose.y;
  //  cv::Mat sub_img=img.clone();
  //  sub_img=sub_img(cvRect(0,f_det_img_.nose.y,img.cols,img.rows-f_det_img_.nose.y-1));
  //  IplImage ipl_img=(IplImage)sub_img;
  //   seq=cvHaarDetectObjects(&ipl_img,mouth_cascade_,mouth_storage_,1.3,4,CV_HAAR_DO_CANNY_PRUNING,cvSize(30*scale,15*scale));
  //   break;
  }
  }

    if(seq->total ==0) return false;
    Rect* seq_det=(Rect*)cvGetSeqElem(seq,0);
    coords.x=(float)seq_det->x+seq_det->width/2+offset[0];
    coords.y=(float)seq_det->y+seq_det->height/2+offset[1];

    return true;
}


void FaceNormalizer::dump_img(cv::Mat& data,std::string name)
{
  if(!debug_)
  {

    std::cout<<"[FaceNomalizer] dump_img() only with set debug flag and path."<<std::endl;
  }
  std::string filename =storage_directory_;
  filename.append(boost::lexical_cast<std::string>(epoch_ctr_));
  filename.append("_");
  filename.append(name);
  filename.append(".jpg");

  cv::imwrite(filename,data);
  return;
}

void FaceNormalizer::dump_features(cv::Mat& img)
{

  cv::Mat img2;
  img.copyTo(img2);
  //IplImage ipl_img=(IplImage)img2;
   cv::circle(img2,cv::Point(f_det_img_.nose.x, f_det_img_.nose.y),5,CV_RGB(255,0,0));
   //cv::circle(img2,cv::Point(f_det_img_.mouth.x,f_det_img_.mouth.y),5,CV_RGB(0,255,0));
   cv::circle(img2,cv::Point(f_det_img_.lefteye.x,f_det_img_.lefteye.y),5,CV_RGB(255,255,0));
   cv::circle(img2,cv::Point(f_det_img_.righteye.x,f_det_img_.righteye.y),5,CV_RGB(255,0,255));
   if(debug_)dump_img(img2,"features");
}

bool FaceNormalizer::save_scene(cv::Mat& RGB,cv::Mat& XYZ,std::string path)
{
  path.append(boost::lexical_cast<std::string>(epoch_ctr_));
  std::cout<<"[FaceNormalizer]Saving to "<<path<<std::endl;
  std::string depth_path,color_path;
  color_path=path;
  color_path.append(".jpg");
  depth_path=path;
  depth_path.append(".xml");
  cv::FileStorage fs(depth_path,FileStorage::WRITE);
  fs << "depth"<<XYZ;
  fs << "color"<<RGB;
  fs.release();
  imwrite(color_path,RGB);

  return true;
}

bool FaceNormalizer::read_scene(cv::Mat& depth, cv::Mat& color,std::string path)
{
  std::cout<<"[FaceNormalizer]Reading from "<<path<<std::endl;
  cv::FileStorage fs(path,FileStorage::READ);
  std::cout<<"read depth 1.xml";
  fs["depthmap"]>> depth;

  std::cout<<"got it";
  //fs["color"]>> color;
  fs.release();
  std::cout<<"released";
  return true;
}


void FaceNormalizer::create_DM(cv::Mat& XYZ,cv::Mat& DM)
{
  //reducing to depth ( z - coordinate only)
  if(XYZ.channels()==3)
  {
  std::vector<cv::Mat> cls;
  cv::split(XYZ,cls);
  DM=cls[2];
  }
  else if (XYZ.channels()==1)
  {
    DM=XYZ;
  }

  //reduce z values
  float minval=std::numeric_limits<float>::max();
  for(int r=0;r<DM.rows;r++)
  {
    for(int c=0;c<DM.cols;c++)
    {
      if(DM.at<float>(r,c) < minval &&DM.at<float>(r,c)!=0) minval =DM.at<float>(r,c);
    }
  }

  for(int r=0;r<DM.rows;r++)
  {
    for(int c=0;c<DM.cols;c++)
    {
      if(DM.at<float>(r,c)!=0)DM.at<float>(r,c)-=minval;
      //if(DM.at<float>(r,c)==0)DM.at<float>(r,c)=255;
    }
  }
  //despeckle<float>(DM,DM);
  cv::medianBlur(DM,DM,5);
  //cv::blur(DM,DM,cv::Size(3,3));
}

bool FaceNormalizer::projectPoint(cv::Point3f& xyz,cv::Point2f& uv)
{
    std::vector<cv::Point3f> m_xyz;
    std::vector<cv::Point2f> m_uv;
    m_xyz.push_back(xyz);

    cv::Vec3f  rot=cv::Vec3f(0.0,0.0,0.0);
    cv::Vec3f  trans=cv::Vec3f(0.0,0.0,0.0);
    cv::projectPoints(m_xyz,rot,trans,cam_mat_,dist_coeffs_,m_uv);
    uv=m_uv[0];
}

bool FaceNormalizer::eliminate_background(cv::Mat& RGB,cv::Mat& XYZ,float background_thresh)
{
  // eliminate background
  cv::Vec3f* xyz_ptr=XYZ.ptr<cv::Vec3f>(0,0);

  if(RGB.channels()==3)
  {
  cv::Vec3b* rgb_ptr=RGB.ptr<cv::Vec3b>(0,0);

  for(int r=0;r<XYZ.total();r++)
  {
    if((*xyz_ptr)[2]>background_thresh ||(*xyz_ptr)[2]<0)
    {
      // set this to invalid value
      *xyz_ptr=cv::Vec3f(-1000,-1000,-1000);
      *rgb_ptr=cv::Vec3b(0,0,0);
    }
    xyz_ptr++;
    rgb_ptr++;
  }
  }

  else if(RGB.channels()==1)
  {
  unsigned char* rgb_ptr=RGB.ptr<unsigned char>(0,0);
  for(int r=0;r<XYZ.total();r++)
  {
    if((*xyz_ptr)[2]>background_thresh ||(*xyz_ptr)[2]<0)
    {
      // set this to invalid value
      *xyz_ptr=cv::Vec3f(-1000,-1000,-1000);
      *rgb_ptr=0;
    }
    xyz_ptr++;
    rgb_ptr++;
  }
  }
}

bool FaceNormalizer::interpolate_head(cv::Mat& RGB, cv::Mat& XYZ)
{
  std::vector<cv::Mat> xyz_vec;
  cv::split(XYZ,xyz_vec);

  //cv::imshow("Z",xyz_vec[2]);
  //cv::imshow("RGB",RGB);

  //cv::waitKey(0);

}
bool FaceNormalizer::normalize_img_type(cv::Mat& in,cv::Mat& out)
{
  in.convertTo(out,CV_64FC1);
  return true;
}

bool FaceNormalizer::orientations(cv::Mat& img, cv::Mat& xyz)
{
	// normalize
	if( img.rows&2!=0 )
	{
		xyz=xyz(cv::Rect(0,0,xyz.cols,xyz.rows-1));
	}
	if( img.cols&2!=0 )
	{
		xyz=xyz(cv::Rect(0,0,xyz.cols-1,xyz.rows));
	}
	normalize_radiometry(img);

	// detect features
	if(!features_from_color(img))
	{
		//std::cout <<"features missing in color"<<std::endl;
		return false;
	}
	if(!features_from_depth(xyz))
	{
		//std::cout <<"features missing in xyz"<<std::endl;
		return false;
	}


	Eigen::Vector3d lefteye,nose,righteye,eye_middle,vec1,vec2,vec3;

	nose<<f_det_xyz_.nose.x,f_det_xyz_.nose.y,f_det_xyz_.nose.z;
	lefteye<<f_det_xyz_.lefteye.x,f_det_xyz_.lefteye.y,f_det_xyz_.lefteye.z;
	righteye<<f_det_xyz_.righteye.x,f_det_xyz_.righteye.y,f_det_xyz_.righteye.z;
	eye_middle=lefteye+((righteye-lefteye)*0.5);

	vec1 = 100*nose - 100*lefteye;
	vec1.normalize();
	vec2 = 100*nose - 100*righteye;
	vec2.normalize();
	vec3 = vec1.cross(vec2);
	//std::cout << "orientation vector x: " << vec3[0] << ", y: " << vec3[1] << ", z: " << vec3[2];
	vec3=10*vec3;
	//std::cout << "10*orientation vector x: " << vec3[0] << ", y: " << vec3[1] << ", z: " << vec3[2];
	vec3.normalize();
	//std::cout << "normalized orientation vector x: " << vec3[0] << ", y: " << vec3[1] << ", z: " << vec3[2];

	std::cout << std::endl;
	return true;

}
