#pragma once
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//
//struct MatchingPt {
//	double score;
//	cv::Vec2i pt_1;
//	cv::Vec2i pt_2;
//};
//
//struct MatchingSurfPt {
//	double score;
//	int index_1;
//	int index_2;
//};
//
//class Utils {
//public:
//	static cv::Vec3d calculate_line(const cv::Point2d& a, const cv::Point2d& b);
//	static cv::Vec3d calculate_line(const cv::Point3d& a, const cv::Point3d& b);
//	static cv::Vec3d calculate_intersection(const cv::Vec3d& a, const cv::Vec3d& b);
//	static void draw_lines(cv::Mat& img, cv::Point2d pts[], cv::Mat& output_img, std::string filename);
//
//	//static std::vector<MatchingPt> calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border);
//	//static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border,
//	//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
//
//	static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border, float threshold,
//		std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
//	static void calculate_ncc(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border, float threshold,
//		std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
//
//	static void calculate_surf_ssd(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptors_1, 
//		const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, const cv::Mat& pair_1_img_1, const cv::Mat& pair_1_img_2, float threshold, std::vector<cv::DMatch>& matches);
//	static void calculate_surf_ncc(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, const cv::Mat& input_img_1, const cv::Mat& input_img_2, float threshold, std::vector<cv::DMatch>& matches);
//
//
//
//
//	//static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border,
//	//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
//};
