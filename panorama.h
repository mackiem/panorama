#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

class LinearLeastSquares {
public:
	static void linear_least_squares_for_homography(const cv::Mat& A, cv::Mat& H);
};

class Ransac {
public:
	static void ransac_for_homography(const cv::Mat& A, const double epsilon_percentage_of_outliers, const int total_correspondences,
		const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2,
	const std::vector<cv::DMatch >& matches);
	
};

class Homography {
private:
	cv::Mat H;
	void construct_first_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A);
	void construct_second_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A);

public:
	Homography(const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2,
	const std::vector<cv::DMatch >& selected_matches);
	cv::Mat get_homography() const;
	
};


