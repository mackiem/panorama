#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

class EuclideanDistance {
public:
	static void match_by_euclidean_distance(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, 
		const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, double threshold, std::vector<cv::DMatch>& matches);
	
};

class LinearLeastSquares {
public:
	static void linear_least_squares_for_homography(const cv::Mat& A, cv::Mat& H);
};

class Ransac {
public:
	static void ransac_for_homography(const double epsilon_percentage_of_outliers, const int total_correspondences,
		const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2,
	const std::vector<cv::DMatch >& matches, cv::Mat& final_H, std::vector<cv::Vec3d>& final_img1_pts_vec3d, std::vector<cv::Vec3d>& final_img2_pts_vec3d);
	
};

class Homography {
private:
	cv::Mat H;
	void construct_first_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A);
	void construct_second_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A);

public:
	Homography(const std::vector<cv::Vec3d>& img_vec_array_1, const std::vector<cv::Vec3d>& img_vec_array_2);
	Homography(const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2, const std::vector<cv::DMatch>& selected_matches);
	void construct_H(const std::vector<cv::Vec3d>& img_vec_array_1, const std::vector<cv::Vec3d>& img_vec_array_2);
	cv::Mat get_homography() const;
	
};

class DogLeg {

	static void construct_first_row_of_jacobian(const cv::Vec3d& img1_pt, const cv::Vec3d& calculated_pt, int pts_i, cv::Mat& J);
	static void construct_second_row_of_jacobian(const cv::Vec3d& img1_pt, const cv::Vec3d& calculated_pt, int pts_i, cv::Mat& J);
	static void compute_jacobian(std::vector<cv::Vec3d>& img1_pts, std::vector<cv::Vec3d>& calculated_pts, cv::Mat& jacobian);
	static void compute_error_vector(std::vector<cv::Vec3d>& ground_truth, std::vector<cv::Vec3d>& calculated_pts, cv::Mat& error_vector);
	static void compute_gradient_descent_increment(const cv::Mat& J, const cv::Mat& error_vector, cv::Mat& gradient_descent_increment);
	static void compute_gauss_newton_increment(const cv::Mat& J, const cv::Mat& error_vector, const cv::Mat& identity, double mu_k, cv::Mat& gauss_newton_increment);
	static double compute_beta(const cv::Mat& gauss_newton_increment, const cv::Mat& gradient_descent_increment, double r_k);
	static void convert_to_H_k(const cv::Mat& P_k, cv::Mat& H_k);
	static void convert_to_P_k(const cv::Mat& H_k, cv::Mat& P_k);
	static double initialize_mu(const cv::Mat& J, double tau);
	static void calculate_result_pts(const std::vector<cv::Vec3d>& img1_pts, const cv::Mat& H_k, std::vector<cv::Vec3d>& calculated_pts);
public:
	static void dogleg_for_non_linear_least_squares_optimization(const cv::Mat& H, std::vector<cv::Vec3d>& img1_pts, std::vector<cv::Vec3d>& img2_pts, cv::Mat& output_H);

};

class ImgUtils {
public:
	static cv::Point2d apply_custom_homography(const cv::Mat matrix, const cv::Point2d& src_pt);
	static void find_bounding_box(const std::vector<cv::Point2d>& points, double& min_x, double& max_x, double& min_y, double& max_y);
	static void apply_distortion_correction(cv::Mat& homography_matrix, const cv::Mat& img, cv::Mat& world_img, const std::string& final_image_name);
	static void combine_transformed_imgs(const cv::Mat& img_1, const cv::Mat& img_2, cv::Mat& H, cv::Mat& transformed_img);
};


