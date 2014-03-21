/*!
*****************************************************************
* \file
*
* \note
* Copyright (c) 2012 \n
* Fraunhofer Institute for Manufacturing Engineering
* and Automation (IPA) \n\n
*
*****************************************************************
*
* \note
* Project name: Care-O-bot
* \note
* ROS stack name: cob_people_perception
* \note
* ROS package name: cob_people_detection
*
* \author
* Author: Richard Bormann
* \author
* Supervised by:
*
* \date Date of creation: 08.08.2012
*
* \brief
* functions for tracking detections, e.g. recognized faces
*
*****************************************************************
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* - Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer. \n
* - Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution. \n
* - Neither the name of the Fraunhofer Institute for Manufacturing
* Engineering and Automation (IPA) nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission. \n
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License LGPL as
* published by the Free Software Foundation, either version 3 of the
* License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License LGPL for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License LGPL along with this program.
* If not, see <http://www.gnu.org/licenses/>.
*
****************************************************************/

//##################
//#### includes ####

#include <cob_people_detection/detection_tracker_node.h>
#include <munkres/munkres.h>

// standard includes
//--

// ROS includes
#include <ros/ros.h>
#include <ros/package.h>

// ROS message includes
#include <sensor_msgs/Image.h>
//#include <sensor_msgs/PointCloud2.h>
#include <cob_people_detection_msgs/DetectionArray.h>
#include <cob_people_detection_msgs/ColorDepthImageArray.h>


// services
//#include <cob_people_detection/DetectPeople.h>

// topics
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// opencv
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// boost
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>

// external includes
#include "cob_vision_utils/GlobalDefines.h"

// Timer
#include "cob_people_detection/timer.h"

#include <sstream>
#include <string>
#include <vector>


using namespace ipa_PeopleDetector;

DetectionTrackerNode::DetectionTrackerNode(ros::NodeHandle nh)
: node_handle_(nh)
{
	it_ = 0;
	sync_input_2_ = 0;
	sync_input_2a_ = 0;

	// parameters
	std::cout << "\n---------------------------\nPeople Detection Parameters:\n---------------------------\n";
	node_handle_.param("debug", debug_, false);
	std::cout << "debug = " << debug_ << "\n";
	node_handle_.param("rosbag_mode", rosbag_mode_, false);
	std::cout << "use rosbag mode:  " << rosbag_mode_ << "\n";
	node_handle_.param("use_people_segmentation", use_people_segmentation_, true);
	std::cout << "use_people_segmentation = " << use_people_segmentation_ << "\n";
	node_handle_.param("face_redetection_time", face_redetection_time_, 2.0);
	std::cout << "face_redetection_time = " << face_redetection_time_ << "\n";
	double publish_currently_not_visible_detections_timespan = face_redetection_time_;
	node_handle_.param("publish_currently_not_visible_detections_timespan", publish_currently_not_visible_detections_timespan, face_redetection_time_);
	std::cout << "publish_currently_not_visible_detections_timespan = " << publish_currently_not_visible_detections_timespan << "\n";
	publish_currently_not_visible_detections_timespan_ = ros::Duration(publish_currently_not_visible_detections_timespan);
	node_handle_.param("min_segmented_people_ratio_face", min_segmented_people_ratio_face_, 0.7);
	std::cout << "min_segmented_people_ratio_face = " << min_segmented_people_ratio_face_ << "\n";
	node_handle_.param("min_segmented_people_ratio_head", min_segmented_people_ratio_head_, 0.2);
	std::cout << "min_segmented_people_ratio_head = " << min_segmented_people_ratio_head_ << "\n";
	node_handle_.param("tracking_range_m", tracking_range_m_, 0.3);
	std::cout << "tracking_range_m = " << tracking_range_m_ << "\n";
	node_handle_.param("face_identification_score_decay_rate", face_identification_score_decay_rate_, 0.9);
	std::cout << "face_identification_score_decay_rate = " << face_identification_score_decay_rate_ << "\n";
	node_handle_.param("min_face_identification_score_to_publish", min_face_identification_score_to_publish_, 0.9);
	std::cout << "min_face_identification_score_to_publish = " << min_face_identification_score_to_publish_ << "\n";
	node_handle_.param("fall_back_to_unknown_identification", fall_back_to_unknown_identification_, true);
	std::cout << "fall_back_to_unknown_identification = " << fall_back_to_unknown_identification_ << "\n";
	node_handle_.param("display_timing", display_timing_, false);
	std::cout << "display_timing = " << display_timing_ << "\n";


	// subscribers
	it_ = new image_transport::ImageTransport(node_handle_);
	people_segmentation_image_sub_.subscribe(*it_, "people_segmentation_image", 1);
	face_position_subscriber_.subscribe(node_handle_, "face_position_array_in", 1);

	// TODO: explore use of original image and head position values from face_positions to compare similar images at all times.
	// due to how the program works before this point, received images come as either cut-out faces or including background
	face_image_subscriber_.subscribe(node_handle_, "/cob_people_detection/face_detector/face_positions", 1);

	// input synchronization
	sensor_msgs::Image::ConstPtr nullPtr;
	cob_people_detection_msgs::ColorDepthImageArray::ConstPtr nullPtrCDI;


	if (use_people_segmentation_ == true)
	{
		sync_input_2_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_people_detection_msgs::DetectionArray, sensor_msgs::Image> >(2);
		sync_input_2_->connectInput(face_position_subscriber_, people_segmentation_image_sub_);
		sync_input_2_->registerCallback(boost::bind(&DetectionTrackerNode::inputCallback, this, _1, _2, nullPtrCDI));
	}
//	else
//	{
//		face_position_subscriber_.registerCallback(boost::bind(&DetectionTrackerNode::inputCallback, this, _1, nullPtr));
//	}
	else
	{
		sync_input_2a_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_people_detection_msgs::DetectionArray, cob_people_detection_msgs::ColorDepthImageArray> >(2);
		sync_input_2a_->connectInput(face_position_subscriber_, face_image_subscriber_);
		sync_input_2a_->registerCallback(boost::bind(&DetectionTrackerNode::inputCallback,this, _1, nullPtr, _2));
	}

	// publishers
	face_position_publisher_ = node_handle_.advertise<cob_people_detection_msgs::DetectionArray>("face_position_array", 1);

	std::cout << "DetectionTrackerNode initialized." << std::endl;
}

DetectionTrackerNode::~DetectionTrackerNode()
{
	if (it_ != 0) delete it_;
	if (sync_input_2_ != 0) delete sync_input_2_;
	if (sync_input_2a_ != 0) delete sync_input_2a_;

}

/// Converts a color image message to cv::Mat format.
unsigned long DetectionTrackerNode::convertColorImageMessageToMat(const sensor_msgs::Image::ConstPtr& image_msg, cv_bridge::CvImageConstPtr& image_ptr, cv::Mat& image)
{
	try
	{
		image_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("PeopleDetection: cv_bridge exception: %s", e.what());
		return ipa_Utils::RET_FAILED;
	}
	image = image_ptr->image;

	return ipa_Utils::RET_OK;
}


/// Copies the data from src to dest.
/// @param src The new data which shall be copied into dest
/// @param dst The new data src is copied into dest
/// @param update If update is true, dest must contain the data which shall be updated
/// @param updateIndex The index in face_identification_votes_ corresponding to the previous detection dest. Only necessary if update is true.
/// @return Return code.
unsigned long DetectionTrackerNode::copyDetection(const cob_people_detection_msgs::Detection& src, cob_people_detection_msgs::Detection& dest, bool update, unsigned int updateIndex)
{
	// 2D image coordinates
	dest.mask.roi.x = src.mask.roi.x; dest.mask.roi.y = src.mask.roi.y;
	dest.mask.roi.width = src.mask.roi.width; dest.mask.roi.height = src.mask.roi.height;

	// 3D world coordinates
//	const geometry_msgs::Point* pointIn = &(src.pose.pose.position);
//	geometry_msgs::Point* pointOut = &(dest.pose.pose.position);
//	pointOut->x = pointIn->x; pointOut->y = pointIn->y; pointOut->z = pointIn->z;
	dest.pose.pose = src.pose.pose;

	// person ID
	if (update==true)
	{
		// update label history for recognized detections
		if (src.label != "UnknownHead")
		{
			for (int i=0; i < src.label_distribution.size(); i++)
			{
				//std::cout << "increasing  " << face_identification_votes_[updateIndex][src.label_distribution[i].label] << " by " << src.label_distribution[i].score;
				face_identification_votes_[updateIndex][src.label_distribution[i].label] += src.label_distribution[i].score;
				//std::cout << " new score " << face_identification_votes_[updateIndex][src.label_distribution[i].label] << "\n";
			}
		}

		//std::cout << src.label_distribution[0].label << "\n";

		// apply voting decay with time and find most voted label
		double max_score = 0;

		// ? it is no longer necessary to copy the updated score of "UnknownHead" to the dest.label
		if (src.label == "UnknownHead")
		{
			std::cout << "copy " << src.label << " votes ( " << face_identification_votes_[updateIndex][src.label] << " ) to " << dest.label << " ( " << face_identification_votes_[updateIndex][dest.label] <<" )\n";
			face_identification_votes_[updateIndex][dest.label] = face_identification_votes_[updateIndex][src.label];
		}


		dest.label = src.label;
		bool only_decay_unknown=false;
		if (src.label == "UnknownHead")
			only_decay_unknown = true;

		// decay applied here
		for (std::map<std::string, double>::iterator face_identification_votes_it=face_identification_votes_[updateIndex].begin(); face_identification_votes_it!=face_identification_votes_[updateIndex].end(); face_identification_votes_it++)
		{
			// todo: make the decay time-dependend - otherwise faster computing = faster decay. THIS IS ACTUALLY WRONG as true detections occur with same rate as decay -> so computing power only affects response time to a changed situation
			// Votes for UnknownHead will be preserved rather than decayed, to keep tracking without continuous face recognition.
			if (!(only_decay_unknown))
			{
				face_identification_votes_it->second *= face_identification_score_decay_rate_;
			}
			else
			{
				if (face_identification_votes_it->first == "Unknown" || face_identification_votes_it->first == "UnknownHead")
					face_identification_votes_it->second *= face_identification_score_decay_rate_;
			}
			std::string label = face_identification_votes_it->first;
			if (face_identification_votes_it->second > max_score && (fall_back_to_unknown_identification_==true || (label!="Unknown" && fall_back_to_unknown_identification_==false)) && label!="UnknownHead" /*&& label!="No face"*/)
			{
				max_score = face_identification_votes_it->second;
				dest.label = label;
			}
		}

		// if the score for the assigned label is higher than the score for UnknownHead increase the score for UnknownHead to the label's score (allows smooth transition if only the head detection is available after recognition)
		// TODO: how does this help?
		if (face_identification_votes_[updateIndex][dest.label] > face_identification_votes_[updateIndex]["UnknownHead"])
		{
			face_identification_votes_[updateIndex]["UnknownHead"] = face_identification_votes_[updateIndex][dest.label];
		}

		if (fall_back_to_unknown_identification_==false)
		{
			if (face_identification_votes_[updateIndex]["Unknown"] > face_identification_votes_[updateIndex]["UnknownHead"])
			{
				face_identification_votes_[updateIndex]["UnknownHead"] = face_identification_votes_[updateIndex]["Unknown"];
			}

		}
	}
	else
		dest.label = src.label;

	if (dest.label=="UnknownHead")
		dest.label = "Unknown";

	// time stamp, detector (color or range)
	// if (update==true)
	// {
	// if (src.detector=="color") dest.detector = "color";
	// }
	// else dest.detector = src.detector;
	dest.detector = src.detector;

	dest.header.stamp = src.header.stamp; //ros::Time::now();

	return ipa_Utils::RET_OK;
}

inline double abs(double x) { return ((x<0) ? -x : x); }


/// Computes the Euclidean distance of a recent faces detection to a current face detection.
/// If the current face detection is outside the neighborhood of the previous detection, DBL_MAX is returned.
/// @return The Euclidian distance of both faces or DBL_MAX.
double DetectionTrackerNode::computeFacePositionDistanceTrackingRange(const cob_people_detection_msgs::Detection& previous_detection, const cob_people_detection_msgs::Detection& current_detection)
{
	const geometry_msgs::Point* point_1 = &(previous_detection.pose.pose.position);
	const geometry_msgs::Point* point_2 = &(current_detection.pose.pose.position);

	double dx = abs(point_1->x - point_2->x);
	double dy = abs(point_1->y - point_2->y);
	double dz = abs(point_1->z - point_2->z);

	// return a huge distance if the current face position is too far away from the recent
	if (dx>tracking_range_m_ || dy>tracking_range_m_ || dz>tracking_range_m_)
	{
		// new face position is too distant to recent position
		return DBL_MAX;
	}

	// return Euclidian distance if both faces are sufficiently close
	double dist = dx*dx+dy*dy+dz*dz;
	return dist;
}


double DetectionTrackerNode::computeFacePositionDistance(const cob_people_detection_msgs::Detection& previous_detection, const cob_people_detection_msgs::Detection& current_detection)
{
	const geometry_msgs::Point* point_1 = &(previous_detection.pose.pose.position);
	const geometry_msgs::Point* point_2 = &(current_detection.pose.pose.position);

	double dx = point_1->x - point_2->x;
	double dy = point_1->y - point_2->y;
	double dz = point_1->z - point_2->z;

	return sqrt(dx*dx+dy*dy+dz*dz);
}

// alternative rosmsg to cv mat converter, preserving the original encoding
unsigned long DetectionTrackerNode::convertColorImageMessageToMatAlt(const sensor_msgs::Image::ConstPtr& image_msg, cv_bridge::CvImageConstPtr& image_ptr, cv::Mat& image)
{
	try
	{
		image_ptr = cv_bridge::toCvShare(image_msg);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("PeopleDetection: cv_bridge exception: %s", e.what());
		return ipa_Utils::RET_FAILED;
	}
	image = image_ptr->image;

	return ipa_Utils::RET_OK;
}

void voidDeleter(const sensor_msgs::Image* const) {}

// receive color_image messages, convert to cvmat, compare cvmats
double DetectionTrackerNode::computeFacePositionImageSimilarity(const sensor_msgs::Image& previous_image_msg, const sensor_msgs::Image& current_image_msg)
{
	double histocomp_temp =0;
	cv_bridge::CvImageConstPtr previous_image_ptr;
	cv::Mat previous_image;
	cv_bridge::CvImageConstPtr current_image_ptr;
	cv::Mat current_image;

	cv::Mat current_image_grey, previous_image_grey;

	// convert old and current image to cvmats using alternative image converter (preserves image encoding of message)
	sensor_msgs::ImageConstPtr previous_image_msg_ptr = boost::shared_ptr<sensor_msgs::Image const>(&(previous_image_msg), voidDeleter);
	convertColorImageMessageToMatAlt(previous_image_msg_ptr, previous_image_ptr, previous_image);

	sensor_msgs::ImageConstPtr current_image_msg_ptr = boost::shared_ptr<sensor_msgs::Image const>(&(current_image_msg), voidDeleter);
	convertColorImageMessageToMatAlt(current_image_msg_ptr, current_image_ptr, current_image);

	// comparison, working on greyscale image
	// TODO: see if comparing color images improves results
	cv::cvtColor(current_image, current_image_grey, CV_BGR2GRAY);
	cv::cvtColor(previous_image, previous_image_grey, CV_BGR2GRAY);

	cv::Mat current_image_olbp;
	cv::Mat previous_image_olbp;
	OLBP_(current_image_grey, current_image_olbp);
	OLBP_(previous_image_grey, previous_image_olbp);
	//cv::imshow("OLBP Test",current_image_olbp);
	//cv::imshow("OLBP Test 2", previous_image_olbp);

	// fill histrogramvector with histograms of face regions. regions are square, number of regions must be a squared int.
	// histogramvect_ then is filled with image 1 and 2 histograms of regions, i.e.:
	//	1 	2 	3
	// 	4 	5 	6	current
	// 	7 	8 	9
	//	10	11	12
	// 	13	14	15	previous
	//	16	17	18

	//TODO: weighted comparison of histograms.
	// cv::compareHist(hist_image_curr, hist_image_prev, CV_COMP_CORREL);
	// Methods for Histogram Comparison:
	//    CV_COMP_CORREL Correlation
	//    CV_COMP_CHISQR Chi-Square
	//    CV_COMP_INTERSECT Intersection
	//    CV_COMP_BHATTACHARYYA Bhattacharyya distance
	OLBPHistogram(current_image_olbp, 16);
	OLBPHistogram(previous_image_olbp, 16);

	// defining weights for histogram regions in three pyramid layers
	int base_weight [] = {1, 3, 3, 1, 3, 8, 8, 3, 3, 8, 8, 3, 1, 3, 3, 1};
	int mid_weight [] = {1, 4, 1, 4, 8, 4, 1, 4, 1};
	int top_weight [] = {4, 4, 4, 4};

	for (int i=0; i<16; i++)
	{
		histocomp_temp += base_weight[i] * cv::compareHist(histogramvect_[i], histogramvect_[i+29], CV_COMP_CORREL);
		//std::cout << "histcomp_temp: " << histocomp_temp << "\n";
	}

	for (int i=16; i<25; i++)
	{
		histocomp_temp += mid_weight[i-16] * cv::compareHist(histogramvect_[i], histogramvect_[i+29], CV_COMP_CORREL);
	}

	for (int i=25; i<29; i++)
	{
		histocomp_temp += top_weight[i-25] * cv::compareHist(histogramvect_[i], histogramvect_[i+29], CV_COMP_CORREL);
	}

	// Divide result by sum of weights to receive normalized correlation.
	double histocomp_res = histocomp_temp / 104;
	//std::cout << "divided by number of regions: " << histocomp_res << "\n";
	histogramvect_.clear();

	return histocomp_res;
}

// Binary Pattern
// create new Mat, fill with 8 bit code for difference between center and surrounding pixels of source image
void DetectionTrackerNode::OLBP_(cv::Mat& src, cv::Mat& dst)
{

    dst = cv::Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
        	int center = src.at<unsigned char>(i,j);
            unsigned char code = 0;
            code |= (src.at<unsigned char>(i-1,j-1) > center) << 7;
            code |= (src.at<unsigned char>(i-1,j) > center) << 6;
            code |= (src.at<unsigned char>(i-1,j+1) > center) << 5;
            code |= (src.at<unsigned char>(i,j+1) > center) << 4;
            code |= (src.at<unsigned char>(i+1,j+1) > center) << 3;
            code |= (src.at<unsigned char>(i+1,j) > center) << 2;
            code |= (src.at<unsigned char>(i+1,j-1) > center) << 1;
            code |= (src.at<unsigned char>(i,j-1) > center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

// Binary Pattern
// Calculate histograms of regions in the src Mat.
// Fill histogramvect_ with resulting histograms.
unsigned long DetectionTrackerNode::OLBPHistogram(cv::Mat src, int regions)
{
	// Set histogram bins count
	int bins = 64;
	int histSize[] = {bins};
	// Set ranges for histogram bins
	float lranges[] = {0, 256};
	const float* ranges[] = {lranges};
	// create matrix for histogram
	cv::Mat hist;
	cv::Mat src_roi;
	int channels[] = {0};

	// calculate histogram over image, equally divided into regions
	// 3 layers: base, mid, top. 4x4, 3x3, 2x2.
	cv::Rect roi;
	int x = sqrt(regions);
	int width_base = src.cols/x;;
	int width_mid = ((src.cols - 2*width_base/3) / (x-1));
	int width_top = ((src.cols - 2*width_base/3 - 2*width_mid/3) / (x-2));

	//std::cout << "Pyramid ROI sizes - Base: " << width_base << " Mid: " << width_mid << " Top: " << width_top << "\n";

	// base layer
	for (int i =0; i<x; i++)
	{
		for (int j=0; j<x; j++)
		{
			roi = cv::Rect((width_base*i), (width_base*j), width_base, width_base);
			src_roi = src(roi);
			cv::calcHist(&src_roi, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
			histogramvect_.push_back(hist);
		}
	}
	//std::cout << "histogramvect_ size after base layer: " << histogramvect_.size() << "\n";

	// middle layer
	for (int i =0; i<x-1; i++)
	{
		for (int j=0; j<x-1; j++)
		{
			roi = cv::Rect((width_mid*i+(width_base/3)), (width_mid*j+(width_base/3)), width_mid, width_mid);
			src_roi = src(roi);
			cv::calcHist(&src_roi, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
			histogramvect_.push_back(hist);
		}
	}
	//std::cout << "histogramvect_ size after mid layer: " << histogramvect_.size() << "\n";

	// top layer
	for (int i =0; i<x-2; i++)
	{
		for (int j=0; j<x-2; j++)
		{
			roi = cv::Rect((width_top*i+((width_mid+width_base)/3)), (width_top*j+((width_mid+width_base)/3)), width_top, width_top);
			src_roi = src(roi);
			cv::calcHist(&src_roi, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
			histogramvect_.push_back(hist);
		}
	}
	//std::cout << "histogramvect_ size after top layer: " << histogramvect_.size() << "\n";

	return ipa_Utils::RET_OK;
}


/// Removes multiple instances of a label by renaming the detections with lower score to Unknown.
/// new functionality: if other labels are available with enough votes accumulated, change to new label instead of Unknown.
/// @return Return code.
unsigned long DetectionTrackerNode::removeMultipleInstancesOfLabel()
{
	// TODO: change to new label: either invert map and use 2nd Element of sorted-by-value map, or iterate through sorted-by-key map to find second highest (whats more efficient?)
	// TODO: repeat this Test if a label was changed for the reason above.
	// check this for each recognized face
	for (int i=0; i<(int)face_position_accumulator_.size(); i++)
	{
		// label of this detection
		std::string label = face_position_accumulator_[i].label;

		// check whether this label has multiple occurrences if it is a real name
		if (label!="Unknown" && label!="No face")
		{
			for (int j=0; j<(int)face_position_accumulator_.size(); j++)
			{
				if (j==i) continue; // do not check yourself

				if (face_position_accumulator_[j].label == label)
				{
					if (debug_)
						std::cout << "face_identification_votes_[i][" << label << "] = " << face_identification_votes_[i][label] << " face_identification_votes_[j][" << label << "] = " << face_identification_votes_[j][label] << "\n";
					// correct this label when some other instance has a higher score on this label
					if (face_identification_votes_[i][label] < face_identification_votes_[j][label])
					{
						face_position_accumulator_[i].label = "Unknown";
						// copy score to unknown if it is higher (this enables the display if either the unknown score or label's recognition score were high enough)
						face_identification_votes_[i]["Unknown"] = face_identification_votes_[i][label];
					}
				}
			}
		}
	}

	return ipa_Utils::RET_OK;
}


unsigned long DetectionTrackerNode::prepareFacePositionMessage(cob_people_detection_msgs::DetectionArray& face_position_msg_out, ros::Time image_recording_time)
{
	// publish face positions
	std::vector<cob_people_detection_msgs::Detection> faces_to_publish;
	for (int i=0; i<(int)face_position_accumulator_.size(); i++)
	{
		if (debug_)
			std::cout << "'UnknownHead' score: " << face_identification_votes_[i]["UnknownHead"] << " label '" << face_position_accumulator_[i].label << "' score: " << face_identification_votes_[i][face_position_accumulator_[i].label] << " - ";
		if ((face_identification_votes_[i][face_position_accumulator_[i].label]>min_face_identification_score_to_publish_ || face_identification_votes_[i]["UnknownHead"]>min_face_identification_score_to_publish_) && ((image_recording_time-face_position_accumulator_[i].header.stamp) < publish_currently_not_visible_detections_timespan_))
		{
			faces_to_publish.push_back(face_position_accumulator_[i]);
			if (debug_) std::cout << "published\n";
		}
		else
			if (debug_)
				std::cout << "not published\n";
	}
	face_position_msg_out.detections = faces_to_publish;
	// hack: for WimiCare replace 'Unknown' by '0000'
	//        for (int i=0; i<(int)face_position_msg_out.detections.size(); i++)
	//        {
	//      	  if (face_position_msg_out.detections[i].label=="Unknown")
	//      		  face_position_msg_out.detections[i].label = "0000";
	//        }
	face_position_msg_out.header.stamp = ros::Time::now();

	return ipa_Utils::RET_OK;
}

/// checks the detected faces from the input topic against the people segmentation and outputs faces if both are positive
void DetectionTrackerNode::inputCallback(const cob_people_detection_msgs::DetectionArray::ConstPtr& face_position_msg_in, const sensor_msgs::Image::ConstPtr& people_segmentation_image_msg, const cob_people_detection_msgs::ColorDepthImageArray::ConstPtr& face_image_msg_in)
{
	// todo? make update rates time dependent!
	// NOT USEFUL, as true detections occur with same rate as decay -> so computing power only affects response time to a changed situation

//	Timer tim;
//	tim.start();

	// convert segmentation image to cv::Mat
	cv_bridge::CvImageConstPtr people_segmentation_image_ptr;
	cv::Mat people_segmentation_image;
	if (use_people_segmentation_ == true)
		convertColorImageMessageToMat(people_segmentation_image_msg, people_segmentation_image_ptr, people_segmentation_image);

	if (debug_)
		std::cout << "incoming detections: " << face_position_msg_in->detections.size() << "\n";

	// delete old face positions in list, i.e. those that were not updated for a long time
	ros::Duration timeSpan(face_redetection_time_);
	// do not delete  when bag file is played back
	if(!rosbag_mode_)
	{
		for (int i=(int)face_position_accumulator_.size()-1; i>=0; i--)
		{
			if ((ros::Time::now()-face_position_accumulator_[i].header.stamp) > timeSpan)
			{
				face_position_accumulator_.erase(face_position_accumulator_.begin()+i);
				face_identification_votes_.erase(face_identification_votes_.begin()+i);
				face_image_accumulator_.erase(face_image_accumulator_.begin()+i);
				//std::cout << "image accu size: " << face_image_accumulator_.size() << " position accu size: " << face_position_accumulator_.size() << "\n";
			}
		}
		if (debug_)
			std::cout << "Old face positions deleted.\n";
	}

	// verify face detections with people segmentation if enabled -> only accept detected faces if a person is segmented at that position
	std::vector<int> face_detection_indices; // index set for face_position_msg_in: contains those indices of detected faces which are supported by a person segmentation at the same location
	if (use_people_segmentation_ == true)
	{
		for(int i=0; i<(int)face_position_msg_in->detections.size(); i++)
		{
			const cob_people_detection_msgs::Detection& det_in = face_position_msg_in->detections[i];
			cv::Rect face;
			face.x = det_in.mask.roi.x;
			face.y = det_in.mask.roi.y;
			face.width = det_in.mask.roi.width;
			face.height = det_in.mask.roi.height;
			//std::cout << "face: " << face.x << " " << face.y << " " << face.width << " " << face.height << "\n";

			int numberBlackPixels = 0;
			for (int v=face.y; v<face.y+face.height; v++)
			{
				uchar* data = people_segmentation_image.ptr(v);
				for (int u=face.x; u<face.x+face.width; u++)
				{
					int index = 3*u;
					if (data[index]==0 && data[index+1]==0 && data[index+2]==0)
						numberBlackPixels++;
				}
			}
			int faceArea = face.height * face.width;
			double segmentedPeopleRatio = (double)(faceArea-numberBlackPixels)/(double)faceArea;

			// if (debug_) std::cout << "ratio: " << segmentedPeopleRatio << "\n";
			if ((det_in.detector=="face" && segmentedPeopleRatio < min_segmented_people_ratio_face_) || (det_in.detector=="head" && segmentedPeopleRatio < min_segmented_people_ratio_head_))
			{
				// False detection
				if (debug_)
					std::cout << "False detection\n";
			}
			else
			{
				// Push index of this detection into the list
				face_detection_indices.push_back(i);
			}
		}
		if (debug_) std::cout << "Verification with people segmentation done.\n";
	}
	else
	{
		for (unsigned int i=0; i<face_position_msg_in->detections.size(); i++)
			face_detection_indices.push_back(i);
	}

	// match current detections with previous detections
	std::vector<bool> current_detection_has_matching(face_detection_indices.size(), false); // index set for face_detection_indices: contains the indices of face_detection_indices and whether these faces could be matched to previously found faces
	// Option 1: greedy procedure
	if (false)
	{
		// build distance matrix
		std::map<int, std::map<int, double> > distance_matrix; // 1. index = face_position_accumulator_ index of previous detections, 2. index = index of current detections, content = spatial distance between the indexed faces
		std::map<int, std::map<int, double> >::iterator distance_matrix_it;
		for (unsigned int previous_det=0; previous_det<face_position_accumulator_.size(); previous_det++)
		{
			std::map<int, double> distance_row;
			for (unsigned int i=0; i<face_detection_indices.size(); i++)
				distance_row[face_detection_indices[i]] = computeFacePositionDistanceTrackingRange(face_position_accumulator_[previous_det], face_position_msg_in->detections[face_detection_indices[i]]);
			distance_matrix[previous_det] = distance_row;
		}
		if (debug_)
			std::cout << "Distance matrix.\n";

		// find matching faces between previous and current detections
		unsigned int number_matchings = (face_position_accumulator_.size() < face_detection_indices.size()) ? face_position_accumulator_.size() : face_detection_indices.size();
		for (unsigned int m=0; m<number_matchings; m++)
		{
			// find minimum matching distance between any two elements of the distance_matrix
			double min_dist = DBL_MAX;
			int previous_min_index, current_min_index;
			for (distance_matrix_it=distance_matrix.begin(); distance_matrix_it!=distance_matrix.end(); distance_matrix_it++)
			{
				for (std::map<int, double>::iterator distance_row_it=distance_matrix_it->second.begin(); distance_row_it!=distance_matrix_it->second.end(); distance_row_it++)
				{
					if (distance_row_it->second < min_dist)
					{
						min_dist = distance_row_it->second;
						previous_min_index = distance_matrix_it->first;
						current_min_index = distance_row_it->first;
					}
				}
			}

			// todo: this is a greedy strategy that is likely to fail -> replace by global matching
			// todo: consider size relation between color and depth image face frames!

			// if there is no matching pair of detections interrupt the search for matchings at this point
			//if (min_dist == optimalAssignment[i].colDBL_MAX)
			//	break;

			// instantiate the matching
			copyDetection(face_position_msg_in->detections[current_min_index], face_position_accumulator_[previous_min_index], true, previous_min_index);
			// mark the respective entry in face_detection_indices as labeled
			for (unsigned int i=0; i<face_detection_indices.size(); i++)
				if (face_detection_indices[i] == current_min_index)
					current_detection_has_matching[i] = true;

			// delete the row and column of the found matching so that both detections cannot be involved in any further matching again
			distance_matrix.erase(previous_min_index);
			for (distance_matrix_it=distance_matrix.begin(); distance_matrix_it!=distance_matrix.end(); distance_matrix_it++)
				distance_matrix_it->second.erase(current_min_index);
		}
	}
	// Option 2: global optimum with Hungarian method
	else
	{
		// create the costs matrix (which consist of Euclidean distance in cm, a punishment for dissimilar labels and a factor from LBP comparison of images)
		std::vector< std::vector<int> > costs_matrix(face_position_accumulator_.size(), std::vector<int>(face_detection_indices.size(), 0));
		// create seperate matrix to save image similarity value
		std::vector< std::vector<int> > costs_matrix_image(face_position_accumulator_.size(), std::vector<int>(face_detection_indices.size(), 0));
		if (debug_)
			std::cout << "Costs matrix.\n";
		for (unsigned int previous_det=0; previous_det<face_position_accumulator_.size(); previous_det++)
		{
			for (unsigned int i=0; i<face_detection_indices.size(); i++)
			{
				// calculate and save image similarity separately, then add it to cost matrix.
				//std::cout << "now comparing previous " << face_position_accumulator_[previous_det].label << " and " << face_position_msg_in->detections[face_detection_indices[i]].label << "\n";

				//costs_matrix[previous_det][i] = 100*computeFacePositionDistance(face_position_accumulator_[previous_det], face_position_msg_in->detections[face_detection_indices[i]])
				//																	+ 100*tracking_range_m_ * (face_position_msg_in->detections[face_detection_indices[i]].label.compare(face_position_accumulator_[previous_det].label)==0 ? 0 : 1);

				costs_matrix_image[previous_det][i] = 100.0 - 100*computeFacePositionImageSimilarity(face_image_accumulator_[previous_det], face_image_msg_in->head_detections[i].color_image);
				costs_matrix[previous_det][i] = 100*computeFacePositionDistance(face_position_accumulator_[previous_det], face_position_msg_in->detections[face_detection_indices[i]])
																				+ 100*tracking_range_m_ * (face_position_msg_in->detections[face_detection_indices[i]].label.compare(face_position_accumulator_[previous_det].label)==0 ? 0 : 1)
																				+ costs_matrix_image[previous_det][i];

				//std::cout << "cost matrix result: "<< costs_matrix[previous_det][i] << "\n";
				if (debug_)
					std::cout << costs_matrix[previous_det][i] << "\t";
			}
			if (debug_)
				std::cout << std::endl;
		}

		if (face_position_accumulator_.size()!=0 && face_detection_indices.size()!=0)
		{
			// solve assignment problem
			munkres assignmentProblem;
			assignmentProblem.set_diag(false);
			assignmentProblem.load_weights(costs_matrix);
			int num_rows = std::min(costs_matrix.size(), costs_matrix[0].size());
			//int num_columns = std::max(costs_matrix.size(), costs_matrix[0].size());
			ordered_pair *optimalAssignment = new ordered_pair[num_rows];
			assignmentProblem.assign(optimalAssignment);
			if (debug_)
				std::cout << "Assignment problem solved.\n";

			// read out solutions, update face_position_accumulator
			for (int i = 0; i < num_rows; i++)
			{
				int current_match_index = 0, previous_match_index = 0;
				if (face_position_accumulator_.size() < face_detection_indices.size())
				{
					// results are reported with original row and column
					previous_match_index = optimalAssignment[i].row;
					current_match_index = optimalAssignment[i].col;

				}
				else
				{
					// rows and columns are switched in results
					previous_match_index = optimalAssignment[i].col;
					current_match_index = optimalAssignment[i].row;
				}

				// stop matching for poor similarity
				// TODO : find appropriate values.
				// added checks for image similarity. Values to be determined experimentally
				// if  cost < 50, do not match to previous detections.
				// -> Detection will have no matching, new detections for unmatched entries are created in the next section.
				if (costs_matrix_image[previous_match_index][current_match_index] < 25 && costs_matrix[previous_match_index][current_match_index] < 80)
				{
					// instantiate the matching
					copyDetection(face_position_msg_in->detections[current_match_index], face_position_accumulator_[previous_match_index], true, previous_match_index);
					// mark the respective entry in face_detection_indices as labeled
					for (unsigned int i=0; i<face_detection_indices.size(); i++)
						if (face_detection_indices[i] == current_match_index)
						{
							current_detection_has_matching[i] = true;
							// update face_image_accumulator_ with current match image.
							face_image_accumulator_[previous_match_index]=face_image_msg_in->head_detections[current_match_index].color_image;
						}
				}
				// TODO: not running copy detection leads to creation of new entry in accumulator.
				else
				{

					face_position_accumulator_.erase(face_position_accumulator_.begin()+previous_match_index);
					face_identification_votes_.erase(face_identification_votes_.begin()+previous_match_index);
					face_image_accumulator_.erase(face_image_accumulator_.begin()+previous_match_index);
					std::cout << "deleted entry of last entity at this position \n";

					std::cout << " Matching to previous detections was blocked because of low image similarities! ";
					/*if (costs_matrix_image[previous_match_index][current_match_index] > 83)
					{
						TODO: can we check suspected false detections again or remove them outright instead of creating new detections for them?
						std::cout << "Cost to match image is extremely high. Match may not be a true face detection. \n";
					}*/
				}

			}
			delete optimalAssignment;
		}
	}
	if (debug_)
		std::cout << "Matches found.\n";

	// create new detections for the unmatched of the current detections if they originate from the color image
	for (unsigned int i=0; i<face_detection_indices.size(); i++)
	{
		if (current_detection_has_matching[i] == false)
		{
			const cob_people_detection_msgs::Detection& det_in = face_position_msg_in->detections[face_detection_indices[i]];

			if (det_in.detector=="face")
			{
				// save in accumulator
				if (debug_)
					std::cout << "\n***** New detection *****\n\n";
				cob_people_detection_msgs::Detection det_out;
				copyDetection(det_in, det_out, false);
				det_out.pose.header.frame_id = "head_cam3d_link";
				face_position_accumulator_.push_back(det_out);

				// remember head detection images
				face_image_accumulator_.push_back(face_image_msg_in->head_detections[face_detection_indices[i]].color_image);
				//std::cout << "image accu size: " << face_image_accumulator_.size() << " position accu size: " << face_position_accumulator_.size() << "\n";

				// remember label history
				std::map<std::string, double> new_identification_data;
				//new_identification_data["Unknown"] = 0.0;
				new_identification_data["UnknownHead"] = 0.0;
				new_identification_data[det_in.label] = 1.0;
				face_identification_votes_.push_back(new_identification_data);
			}
		}
	}
	if (debug_)
		std::cout << "New detections.\n";

	// eliminate multiple instances of a label
	removeMultipleInstancesOfLabel();

	// publish face positions
	ros::Time image_recording_time = (face_position_msg_in->detections.size() > 0 ? face_position_msg_in->detections[0].header.stamp : ros::Time::now());
	cob_people_detection_msgs::DetectionArray face_position_msg_out;
	prepareFacePositionMessage(face_position_msg_out, image_recording_time);
	face_position_msg_out.header.stamp = face_position_msg_in->header.stamp;
	face_position_publisher_.publish(face_position_msg_out);
  
//  // display
//  if (debug_ == true)
//  {
//	// convert image message to cv::Mat
//	cv_bridge::CvImageConstPtr colorImagePtr;
//	cv::Mat colorImage;
//	convertColorImageMessageToMat(color_image_msg, colorImagePtr, colorImage);
//
//	// display color image
//	for(int i=0; i<(int)face_position_msg_out.detections.size(); i++)
//	{
//	  cv::Rect face;
//	  cob_people_detection_msgs::Rect& faceRect = face_position_msg_out.detections[i].mask.roi;
//	  face.x = faceRect.x; face.width = faceRect.width;
//	  face.y = faceRect.y; face.height = faceRect.height;
//
//	  if (face_position_msg_out.detections[i].detector == "range")
//		cv::rectangle(colorImage, cv::Point(face.x, face.y), cv::Point(face.x + face.width, face.y + face.height), CV_RGB(0, 0, 255), 2, 8, 0);
//	  else
//		cv::rectangle(colorImage, cv::Point(face.x, face.y), cv::Point(face.x + face.width, face.y + face.height), CV_RGB(0, 255, 0), 2, 8, 0);
//
//	  if (face_position_msg_out.detections[i].label == "Unknown")
//		// Distance to face class is too high
//		cv::putText(colorImage, "Unknown", cv::Point(face.x,face.y+face.height+25), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB( 255, 0, 0 ), 2);
//	  else if (face_position_msg_out.detections[i].label == "No face")
//		// Distance to face space is too high
//		cv::putText(colorImage, "No face", cv::Point(face.x,face.y+face.height+25), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB( 255, 0, 0 ), 2);
//	  else
//		// Face classified
//		cv::putText(colorImage, face_position_msg_out.detections[i].label.c_str(), cv::Point(face.x,face.y+face.height+25), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB( 0, 255, 0 ), 2);
//	}
//	// publish image
//	cv_bridge::CvImage cv_ptr;
//	cv_ptr.image = colorImage;
//	cv_ptr.encoding = "bgr8";
//	people_detection_image_pub_.publish(cv_ptr.toImageMsg());
//  }

	if (display_timing_ == true)
		ROS_INFO("%d DetectionTracker: Time stamp of pointcloud message: %f. Delay: %f.", face_position_msg_in->header.seq, face_position_msg_in->header.stamp.toSec(), ros::Time::now().toSec()-face_position_msg_in->header.stamp.toSec());
//	ROS_INFO("Detection Tracker took %f ms.", tim.getElapsedTimeInMilliSec());
}


//#######################
//#### main programm ####
int main(int argc, char** argv)
{
	// Initialize ROS, specify name of node
	ros::init(argc, argv, "detection_tracker");

	// Create a handle for this node, initialize node
	ros::NodeHandle nh;

	// Create FaceRecognizerNode class instance
	DetectionTrackerNode detection_tracker_node(nh);

	// Create action nodes
	//DetectObjectsAction detect_action_node(object_detection_node, nh);
	//AcquireObjectImageAction acquire_image_node(object_detection_node, nh);
	//TrainObjectAction train_object_node(object_detection_node, nh);

	ros::spin();

	return 0;
}
