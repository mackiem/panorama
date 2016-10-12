
#include "panorama.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>
#include "Utils.h"
#include "Corners.h"
#include <chrono>
#include <cmath>
#include <random>
#include <set>

const int NO_OF_POINTS = 4;

cv::Mat pair_1_img_1;
cv::Mat pair_1_img_2;
int sigma_i;
const int sigma_max = 10;

int ssd_threshold;
const int ssd_threshold_max = 100;

int ncc_threshold;
const int ncc_threshold_max = 100;

int surf_hessian_threshold;
const int surf_hessian_threshold_max = 1000;
std::vector<cv::Vec2i> corners_1;
std::vector<cv::Vec2i> corners_2;
float sigma;

std::string sub_dir;


void EuclideanDistance::match_by_euclidean_distance(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, 
	const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, double threshold, std::vector<cv::DMatch>& matches) {

	std::vector<MatchingSurfPt> matching_pts;
	double max_ssd = -DBL_MAX;
	double min_ssd = DBL_MAX;

	//std::cout << descriptor_1.row(0) << " " << descriptor_1.rows << " " << descriptor_1.cols << "\n";

	for (auto i1 = 0; i1 < keypoints_1.size(); ++i1) {

		cv::Mat keypt_1_descriptor = descriptor_1.row(i1);

		for (auto i2 = 0; i2 < keypoints_2.size(); ++i2) {

			double ssd = 0.0;

			cv::Mat keypt_2_descriptor = descriptor_2.row(i2);

			for (int k = 0; k < keypt_1_descriptor.cols; ++k) {
				auto des_1 = keypt_1_descriptor.at<float>(0, k);
				auto des_2 = keypt_2_descriptor.at<float>(0, k);
				ssd += std::pow((des_1 - des_2), 2);
			}

			max_ssd = std::max(max_ssd, ssd);
			min_ssd = std::min(min_ssd, ssd);

			MatchingSurfPt matching_pt;
			matching_pt.score = ssd;
			matching_pt.index_1 = i1;
			matching_pt.index_2 = i2;

			matching_pts.push_back(matching_pt);
		}
	}


	// sort values according to SSD for each point
	std::sort(matching_pts.begin(), matching_pts.end(), [&] (const MatchingSurfPt& lhs, const MatchingSurfPt& rhs)
	{
		if (lhs.index_1 == rhs.index_1) {
			return lhs.score < rhs.score;
		}
		return lhs.index_1 < rhs.index_1;
	});

	std::vector<MatchingSurfPt> filtered_matching_pts;

	int prev_index = -1;


	std::cout << "max ssd : " << max_ssd << " min ssd: " << min_ssd << " threshold : " << threshold << "\n";

	// keep only points that are T * SSD_(max)
	for (auto i = 0; i < matching_pts.size(); ++i) {
		auto matching_pt = matching_pts[i];
		if (prev_index == matching_pt.index_1) {
		} else {
			if (matching_pt.score < threshold * (min_ssd)) {
				cv::DMatch match;
				match.queryIdx = matching_pt.index_1;
				match.trainIdx = matching_pt.index_2;
				matches.push_back(match);
			} 
		}
		prev_index = matching_pt.index_1;
	}
}

void LinearLeastSquares::linear_least_squares_for_homography(const cv::Mat& A, cv::Mat& H) {

	cv::Mat U, D, Vt;
	cv::SVD::compute(A, D, U, Vt, cv::SVD::FULL_UV);

	std::cout << Vt << "\n";

	cv::Mat last_col_V = Vt.t().col(Vt.cols - 1);

	H = last_col_V;

	std::cout << H << "\n";

}

double round(double number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

void Ransac::ransac_for_homography(const cv::Mat& A, const double epsilon_percentage_of_outliers, const int n_total_correspondences,
	const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2,
	const std::vector<cv::DMatch >& matches, cv::Mat& final_H, std::vector<cv::Vec3d>& final_img1_pts_vec3d, std::vector<cv::Vec3d>& final_img2_pts_vec3d) {

	const double sigma = 5;
	// threshold for deciding whether an inlier
	const double delta = 3 * sigma;


	// N = no of trials
	const double p = 0.99;
	const int n = 6;
	const int N = std::log(1 - p) / std::log(1 - std::pow((1 - epsilon_percentage_of_outliers), n));

	// M = minimum size of the inlier set for it to be considered acceptable
	const int M = round((1 - epsilon_percentage_of_outliers) * n_total_correspondences);

	cv::Mat max_h;

	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<int> dist(0, n_total_correspondences - 1);

	std::set<int> selected_matches_index;
	std::vector<cv::DMatch> selected_matches(n);

	// max
	int max_inliers = 0;
	std::vector<cv::Vec3d> max_img1_pts_vec3d;
	std::vector<cv::Vec3d> max_img2_pts_vec3d;

	for (int i = 0; i < N; ++i) {
		// randomly select n points
		for (int j = 0; j < n || selected_matches_index.size() < n; ++j) {
			selected_matches_index.insert(dist(eng));
		}

		std::vector<cv::Vec3d> inlier_img1_pts_vec3d;
		std::vector<cv::Vec3d> inlier_img2_pts_vec3d;


		// construct Homography
		int j = 0;
		for (auto itr = selected_matches_index.begin(); itr != selected_matches_index.end(); ++itr, ++j) {
			selected_matches[j] = matches[*itr];
			cv::KeyPoint img_pt1 = key_points_1[matches[*itr].queryIdx];
			cv::KeyPoint img_pt2 = key_points_2[matches[*itr].trainIdx];
			cv::Vec3d img_pt1_vec(img_pt1.pt.x, img_pt1.pt.y, 1.0);
			cv::Vec3d img_pt2_vec(img_pt2.pt.x, img_pt2.pt.y, 1.0);
			inlier_img1_pts_vec3d.push_back(img_pt1_vec);
			inlier_img2_pts_vec3d.push_back(img_pt2_vec);
		}

		Homography homography(key_points_1, key_points_2, selected_matches);
		cv::Mat H = homography.get_homography();

		// calculate outliers and inliners
		for (j = 0; j < matches.size(); ++j) {
			// make sure it's not a randomly selected point
			if (selected_matches_index.find(j) != selected_matches_index.end()) {
				continue;
			}
			cv::KeyPoint img_pt1 = key_points_1[matches[i].queryIdx];
			cv::KeyPoint img_pt2 = key_points_2[matches[i].trainIdx];

			cv::Vec3d img_pt1_vec(img_pt1.pt.x, img_pt1.pt.y, 1.0);
			cv::Vec3d img_pt2_vec(img_pt2.pt.x, img_pt2.pt.y, 1.0);

			cv::Mat calculated_pt_mat = H * cv::Mat(img_pt2_vec);
			cv::Vec3d calculated_pt(calculated_pt_mat);

			cv::Vec2d img_pt2_vec2d(img_pt2_vec[0], img_pt2_vec[1]);
			if (calculated_pt[2] < 1e-6) {
				continue;
			}
			cv::Vec2d calculated_pt_vec2d(calculated_pt[0] / calculated_pt[2], calculated_pt[1] / calculated_pt[2]);
			double distance = cv::norm(calculated_pt_vec2d - img_pt2_vec2d);
			if (distance < delta) {
				inlier_img1_pts_vec3d.push_back(img_pt1_vec);
				inlier_img2_pts_vec3d.push_back(img_pt2_vec);
			}
		}

		if (max_inliers < inlier_img1_pts_vec3d.size()) {
			// new max inliers
			max_inliers = inlier_img1_pts_vec3d.size();
			max_img1_pts_vec3d = inlier_img1_pts_vec3d;
			max_img2_pts_vec3d = inlier_img2_pts_vec3d;
		}
	}

	// refine homography with max inlier set
	Homography max_homography(max_img1_pts_vec3d, max_img2_pts_vec3d);
	final_H = max_homography.get_homography();
	final_img1_pts_vec3d = max_img1_pts_vec3d;
	final_img2_pts_vec3d = max_img2_pts_vec3d;

}

void Homography::construct_first_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A) {
	for (int i = 0; i < 3; ++i) {
		A.at<double>(2 * pts_i, i) = 0.0;	
	}

	for (int i = 3; i < 6; ++i) {
		A.at<double>(2 * pts_i, i) = -1.0 * img2_pt[2] * img1_pt(i - 3);	
	}

	for (int i = 6; i < 9; ++i) {
		A.at<double>(2 * pts_i, i) = img2_pt[1] * img1_pt(i - 6);	
	}
}

void Homography::construct_second_row(const cv::Vec3d& img1_pt, const cv::Vec3d& img2_pt, int pts_i, cv::Mat& A) {
	for (int i = 0; i < 3; ++i) {
		A.at<double>(2 * pts_i + 1, i) = img2_pt[2] * img1_pt(i);	
	}

	for (int i = 3; i < 6; ++i) {
		A.at<double>(2 * pts_i + 1, i) = 0.0;
	}

	for (int i = 6; i < 9; ++i) {
		A.at<double>(2 * pts_i + 1, i) = -1.0 * img2_pt[0] * img1_pt(i - 6);
	}
}


Homography::Homography(const std::vector<cv::KeyPoint>& key_points_1, const std::vector<cv::KeyPoint>& key_points_2, const std::vector<cv::DMatch>& selected_matches) {

	cv::Mat A(selected_matches.size() * 2, 9, CV_64F);
	std::vector<cv::Vec3d> img_vec_array_1; 
	std::vector<cv::Vec3d> img_vec_array_2;

	for (int i = 0; i < selected_matches.size(); ++i) {
		cv::KeyPoint img_pt1 = key_points_1[selected_matches[i].queryIdx];
		cv::KeyPoint img_pt2 = key_points_2[selected_matches[i].trainIdx];

		cv::Vec3d img_vec_1(img_pt1.pt.x, img_pt1.pt.y, 1.0);
		cv::Vec3d img_vec_2(img_pt2.pt.x, img_pt2.pt.y, 1.0);

		img_vec_array_1.push_back(img_vec_1);
		img_vec_array_2.push_back(img_vec_2);

		//construct_first_row(img_vec_1, img_vec_2, i, A);
		//construct_second_row(img_vec_1, img_vec_2, i, A);
	}

	Homography(img_vec_array_1, img_vec_array_2);

	//cv::Mat H_vector;
	//LinearLeastSquares::linear_least_squares_for_homography(A, H_vector);

	//H = cv::Mat(3, 3, CV_64F);
	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		H.at<double>(i, j) = H_vector.at<double>(3 * i + j, 0);
	//	}
	//}
}

Homography::Homography(const std::vector<cv::Vec3d>& img_vec_array_1, const std::vector<cv::Vec3d>& img_vec_array_2) {
	cv::Mat A(img_vec_array_1.size() * 2, 9, CV_64F);
	for (int i = 0; i < img_vec_array_1.size(); ++i) {
		construct_first_row(img_vec_array_1[i], img_vec_array_2[i], i, A);
		construct_second_row(img_vec_array_1[i], img_vec_array_2[i], i, A);
	}
	cv::Mat H_vector;
	LinearLeastSquares::linear_least_squares_for_homography(A, H_vector);

	H = cv::Mat(3, 3, CV_64F);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			H.at<double>(i, j) = H_vector.at<double>(3 * i + j, 0);
		}
	}

}

cv::Mat Homography::get_homography() const {
	return H;
}

void DogLeg::construct_first_row_of_jacobian(const cv::Vec3d& img1_pt, const cv::Vec3d& calculated_pt, int pts_i, cv::Mat& J) const {
	for (int i = 0; i < 3; ++i) {
		J.at<double>(2 * pts_i, i) = img1_pt[i] / calculated_pt[2];
	}

	for (int i = 3; i < 6; ++i) {
		J.at<double>(2 * pts_i, i) = 0.0;
	}

	for (int i = 6; i < 9; ++i) {
		J.at<double>(2 * pts_i, i) = -1.0 * img1_pt[i - 6] * calculated_pt[0] / std::pow(calculated_pt[2], 2);
	}
}

void DogLeg::construct_second_row_of_jacobian(const cv::Vec3d& img1_pt, const cv::Vec3d& calculated_pt, int pts_i, cv::Mat& J) const {
	for (int i = 0; i < 3; ++i) {
		J.at<double>(2 * pts_i + 1, i) = 0.0;
	}
	
	for (int i = 3; i < 6; ++i) {
		J.at<double>(2 * pts_i + 1, i) = img1_pt[i - 3] / calculated_pt[2];
	}

	for (int i = 6; i < 9; ++i) {
		J.at<double>(2 * pts_i + 1, i) = -1.0 * img1_pt[i - 6] * calculated_pt[1] / std::pow(calculated_pt[2], 2);
	}
}

void DogLeg::compute_jacobian(std::vector<cv::Vec3d>& img1_pts, std::vector<cv::Vec3d>& calculated_pts, cv::Mat& jacobian) {
	for (int i = 0; i < img1_pts.size(); ++i) {
		construct_first_row_of_jacobian(img1_pts[i], calculated_pts[i], i, jacobian);
		construct_second_row_of_jacobian(img1_pts[i], calculated_pts[i], i, jacobian);
	}
}

void DogLeg::compute_error_vector(std::vector<cv::Vec3d>& ground_truth, std::vector<cv::Vec3d>& calculated_pts, cv::Mat& error_vector) const {
	for (int i = 0; i < ground_truth.size(); ++i) {
		error_vector.at<double>(2 * i, 0) = ground_truth[i][0] - (calculated_pts[i][0] / calculated_pts[i][2]);
		error_vector.at<double>(2 * i + 1, 0) = ground_truth[i][1] - (calculated_pts[i][1] / calculated_pts[i][2]);
	}
}

void DogLeg::compute_gradient_descent_increment(const cv::Mat& J, const cv::Mat& error_vector, cv::Mat& gradient_descent_increment) const {
	cv::Mat increment = J.t() * error_vector;
	double numerator = cv::norm(increment);
	double denominator = cv::norm(J * increment);

	gradient_descent_increment = (numerator / denominator) * increment;
}

void DogLeg::compute_gauss_newton_increment(const cv::Mat& J, const cv::Mat& error_vector, const cv::Mat& identity, double mu_k, cv::Mat& gauss_newton_increment) const {

	cv::Mat multiplier = J.t() * J + mu_k * identity;
	gauss_newton_increment = multiplier.inv() * J.t() * error_vector;

}

double DogLeg::compute_beta(const cv::Mat& gauss_newton_increment, const cv::Mat& gradient_descent_increment, double r_k) {
	cv::Mat difference_increment = gauss_newton_increment - gradient_descent_increment;
	double a = std::pow(cv::norm(difference_increment), 2);
	cv::Mat result_b_coeff = gradient_descent_increment.t() * difference_increment;
	double b = 2.0 * result_b_coeff.at<double>(0, 0);
	double c = std::pow(cv::norm(gradient_descent_increment), 2) - std::pow(r_k, 2);

	double determinant = std::pow(b, 2) - 4 * a * c;
	if (determinant  < 0) {
		std::cout << "Undefined condition occurred.\n";
	}

	double beta = -b + std::sqrt(determinant) / (2 * a);

	return beta;
}

void DogLeg::convert_to_H_k(const cv::Mat& P_k, cv::Mat& H_k) const {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			H_k.at<double>(i, j) = P_k.at<double>(3 * i + j, 0);
		}
	}
}

void DogLeg::convert_to_P_k(const cv::Mat& H_k, cv::Mat& P_k) const {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			 P_k.at<double>(3 * i + j, 0) = H_k.at<double>(i, j);
		}
	}
}

double DogLeg::initialize_mu(const cv::Mat& J, double tau) const {
	cv::Mat J_squared = J.t() * J;
	double max = -DBL_MAX;
	for (int i = 0; i < J_squared.rows; ++i) {
		max = std::max(max, J_squared.at<double>(i, i));
	}
	return max * tau;
}

void DogLeg::calculate_result_pts(const std::vector<cv::Vec3d>& img1_pts, const cv::Mat& H_k, std::vector<cv::Vec3d>& calculated_pts) const {
	for (int i = 0; i < img1_pts.size(); ++i) {
		cv::Mat img_1_pts_mat = cv::Mat(img1_pts[i]);
		cv::Mat calculated_pts_mat = H_k * img_1_pts_mat;
		calculated_pts[i] = calculated_pts_mat;
	}
}

void DogLeg::dogleg_for_non_linear_least_squares_optimization(const cv::Mat& H, std::vector<cv::Vec3d>& img1_pts, std::vector<cv::Vec3d>& img2_pts, cv::Mat& output_H) {

	cv::Mat P_k(9, 1, CV_64F);

	cv::Mat H_k = H;
	cv::Mat H_k_plus_1 = H;

	convert_to_P_k(H_k, P_k);

	int some_iterations = 10000;
	double error_threshold = 0.01;


	std::vector<cv::Vec3d> calculated_pts(img1_pts.size());
	std::vector<cv::Vec3d> calculated_k_plus_1_pts(img1_pts.size());
	cv::Mat J(img1_pts.size() * 2, 9, CV_64F);
	cv::Mat error_vector(img1_pts.size() * 2, 0, CV_64F);
	cv::Mat error_vector_k_plus_1(img1_pts.size() * 2, 0, CV_64F);

	cv::Mat gradient_descent_increment(9, 1, CV_64F);
	cv::Mat gauss_newton_increment(9, 1, CV_64F);

	calculate_result_pts(img1_pts, H_k, calculated_pts);
	compute_jacobian(img1_pts, calculated_pts, J);
	compute_error_vector(img2_pts, calculated_pts, error_vector);

	double calculated_error = cv::norm(error_vector);

	double tau = 0.001;
	double mu_k = initialize_mu(J, tau);
	double r_k = 0.5;

	cv::Mat identity(9, 9, CV_64F);
	cv::setIdentity(identity);

	cv::Mat last_increment;

	for (int n = 0; n < some_iterations || calculated_error > error_threshold; ++n) {

		convert_to_H_k(P_k, H_k);
		calculate_result_pts(img1_pts, H_k, calculated_pts);

		compute_jacobian(img1_pts, calculated_pts, J);
		compute_error_vector(img2_pts, calculated_pts, error_vector);

		compute_gradient_descent_increment(J, error_vector, gradient_descent_increment);
		compute_gauss_newton_increment(J, error_vector, identity, mu_k, gauss_newton_increment);

		double gauss_newton_norm = cv::norm(gauss_newton_increment);
		double gradient_descent_norm = cv::norm(gradient_descent_norm);

		
		cv::Mat increment;
		if (gauss_newton_norm < r_k) {
			increment = gauss_newton_increment;
		} else if (gradient_descent_norm < r_k && r_k < gauss_newton_norm) {
			double beta = compute_beta(gauss_newton_increment, gradient_descent_increment, r_k);
			increment = gradient_descent_increment + beta * (gauss_newton_increment - gradient_descent_increment);
		} else {
			increment = (r_k / gradient_descent_norm) * gradient_descent_increment;
		}

		cv::Mat temp_P_k_plus_1 = P_k + increment;

		// calculate of C_p(k) and C_p(k+1)
		convert_to_H_k(temp_P_k_plus_1, H_k_plus_1);
		calculate_result_pts(img1_pts, H_k_plus_1, calculated_k_plus_1_pts);
		compute_error_vector(img2_pts, calculated_k_plus_1_pts, error_vector_k_plus_1);

		cv::Mat C_p_mat = error_vector.t() * error_vector;
		cv::Mat C_p_plus_1_mat = error_vector_k_plus_1.t() * error_vector_k_plus_1;

		// calculate row_LM
		double row_lm_numerator = C_p_mat.at<double>(0, 0) - C_p_plus_1_mat.at<double>(0, 0);
		cv::Mat row_lm_denominator_mat = increment.t() * J.t() * error_vector + increment.t() * mu_k * identity * increment;

		double row_LM = row_lm_numerator / row_lm_denominator_mat.at<double>(0, 0);


		// calculate row_DL
		double row_DL_numerator = row_lm_numerator;
		cv::Mat row_DL_denominator_mat = 2.0 * increment * J.t() * error_vector - increment.t() * J.t() * J * increment;

		double row_DL = row_DL_numerator / row_DL_denominator_mat.at<double>(0, 0);

		if (row_DL <= 0.0) {
			// we've jumped too far, revert
			r_k = r_k / 2.0;
			mu_k = 2 * mu_k;
		} else {
			calculated_error = row_lm_numerator;
			mu_k =  mu_k * std::max(1.0/3.0, 1 - std::pow(2 * row_LM - 1, 3));
			if (row_DL < 0.25) {
				r_k = r_k / 4.0;
			} else if (0.25 <= row_DL && row_DL <= 0.75) {
				// just for the if
				r_k = r_k;
			} else {
				r_k = r_k * 2.0;
			}
			P_k = temp_P_k_plus_1;
		}


	}


}

void ssd(int threshold, void*) {
	std::vector<cv::KeyPoint> key_points_1;
	std::vector<cv::KeyPoint> key_points_2;

	std::vector<cv::DMatch > matches;

	auto t1 = std::chrono::high_resolution_clock::now();

	Utils::calculate_ssd(corners_1, corners_2, pair_1_img_1, pair_1_img_2, 10 * sigma, threshold, key_points_1, key_points_2, matches);
	
	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Time taken for ssd : " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << "\n";

	cv::Mat output_img;
	cv::drawMatches(pair_1_img_1, key_points_1, pair_1_img_2, key_points_2, matches, output_img);
	cv::imshow("ssd", output_img);

	std::string sigma_str = "sigma_" + std::to_string(sigma) + "_";
	std::string filename = sub_dir + sigma_str + "corner_ssd.jpg";
	cv::imwrite(filename, output_img);
}

cv::Mat descriptors_1, descriptors_2;
std::vector<cv::KeyPoint> surf_keypoints_1, surf_keypoints_2;

void surf(int threshold, void*) {
	cv::SurfFeatureDetector detector(threshold);


	detector.detect( pair_1_img_1, surf_keypoints_1 );
	detector.detect( pair_1_img_2, surf_keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SurfDescriptorExtractor extractor;

	extractor.compute( pair_1_img_1, surf_keypoints_1, descriptors_1 );
	extractor.compute( pair_1_img_2, surf_keypoints_2, descriptors_2 );

}

int main(int argc, char** argv)  {

	std::string dir = "set_1/";

	std::vector<cv::Mat> imgs;
	for (int i = 0; i < 2; ++i) {
		std::stringstream ss;
		ss << dir;
		ss << (i+1) << ".jpg";
		cv::Mat img = cv::imread(ss.str());
		imgs.push_back(img);
	}

	// init
	std::vector<std::vector<cv::KeyPoint>> key_points_in_imgs;
	cv::SiftFeatureDetector sift_feature_detector;
	sift_feature_detector.detect(imgs, key_points_in_imgs);

	std::vector<cv::Mat> descriptors;
	cv::SiftDescriptorExtractor sift_descriptor_extractor;
	sift_descriptor_extractor.compute()

	cv::createTrackbar("T", "corners_1", &sigma_i, sigma_max, corner_detect);

	cv::waitKey(0);

}
