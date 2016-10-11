#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

struct MatchingPt {
	double score;
	cv::Vec2i pt_1;
	cv::Vec2i pt_2;
};

struct MatchingSurfPt {
	double score;
	int index_1;
	int index_2;
};

class Utils {
public:
	static cv::Vec3d calculate_line(const cv::Point2d& a, const cv::Point2d& b);
	static cv::Vec3d calculate_line(const cv::Point3d& a, const cv::Point3d& b);
	static cv::Vec3d calculate_intersection(const cv::Vec3d& a, const cv::Vec3d& b);
	static cv::Point2d apply_custom_homography(const cv::Mat matrix, const cv::Point2d& src_pt);
	static void find_bounding_box(const std::vector<cv::Point2d>& points, double& min_x, double& max_x, double& min_y, double& max_y);
	//static void find_bounding_box(const cv::Point_<float> points[4], float& min_x, float& max_x, float& min_y, float& max_y);
	static void contruct_projected_img(cv::Point2f* img_1, cv::Mat& homography_matrix, const cv::Mat image_to_project, cv::Mat& world_img, const std::string& final_image_name);
	static void apply_distortion_correction(cv::Mat& homography_matrix, const cv::Mat& img, cv::Mat& world_img, const std::string& final_image_name);
	static void draw_lines(cv::Mat& img, cv::Point2d pts[], cv::Mat& output_img, std::string filename);

	//static std::vector<MatchingPt> calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border);
	//static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border,
	//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);

	static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border, float threshold,
		std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
	static void calculate_ncc(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border, float threshold,
		std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);

	static void calculate_surf_ssd(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptors_1, 
		const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, const cv::Mat& pair_1_img_1, const cv::Mat& pair_1_img_2, float threshold, std::vector<cv::DMatch>& matches);
	static void calculate_surf_ncc(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, const cv::Mat& input_img_1, const cv::Mat& input_img_2, float threshold, std::vector<cv::DMatch>& matches);


	static void ransac_for_homography();


	//static void calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border,
	//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches);
};
