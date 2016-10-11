
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
	const std::vector<cv::DMatch >& matches) {

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
	int max_inliers = 0;

	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<int> dist(0, n_total_correspondences - 1);

	std::set<int> selected_matches_index;
	std::vector<cv::DMatch> selected_matches(n);

	for (int i = 0; i < N; ++i) {
		// randomly select n points
		for (int j = 0; j < n || selected_matches_index.size() <= n; ++j) {
			selected_matches_index.insert(dist(eng));
		}

		// construct A
		int j = 0;
		for (auto itr = selected_matches_index.begin(); itr != selected_matches_index.end(); ++itr, ++j) {
			selected_matches[j] = matches[*itr];
		}

		// calculate h

		//  c

	}
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
	for (int i = 0; i < selected_matches.size(); ++i) {
		cv::KeyPoint img_pt1 = key_points_1[selected_matches[i].queryIdx];
		cv::KeyPoint img_pt2 = key_points_2[selected_matches[i].trainIdx];

		cv::Vec3d img_vec_1(img_pt1.pt.x, img_pt1.pt.y, 1.0);
		cv::Vec3d img_vec_2(img_pt2.pt.x, img_pt2.pt.y, 1.0);

		construct_first_row(img_vec_1, img_vec_2, i, A);
		construct_second_row(img_vec_1, img_vec_2, i, A);
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

void ncc(int threshold, void*) {
	std::vector<cv::KeyPoint> key_points_1;
	std::vector<cv::KeyPoint> key_points_2;

	std::vector<cv::DMatch > matches;

	auto t1 = std::chrono::high_resolution_clock::now();

	Utils::calculate_ncc(corners_1, corners_2, pair_1_img_1, pair_1_img_2, 10 * sigma, threshold / (float) ncc_threshold_max, key_points_1, key_points_2, matches);

	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Time taken for ncc: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << "\n";

	cv::Mat output_img;
	cv::drawMatches(pair_1_img_1, key_points_1, pair_1_img_2, key_points_2, matches, output_img);
	cv::imshow("ncc", output_img);


	std::string sigma_str = "sigma_" + std::to_string(sigma) + "_";
	std::string filename = sub_dir + sigma_str + "corner_ncc.jpg";
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


void surf_ssd(int threshold, void*) {

	std::vector<cv::DMatch > matches;

	Utils::calculate_surf_ssd(surf_keypoints_1, descriptors_1, surf_keypoints_2, descriptors_2, pair_1_img_1, pair_1_img_2, threshold, matches);

	cv::Mat output_img;
	cv::drawMatches(pair_1_img_1, surf_keypoints_1, pair_1_img_2, surf_keypoints_2, matches, output_img);
	cv::imshow("surf_ssd", output_img);

	//std::string sigma_str = "sigma_" + std::to_string(sigma);
	std::string filename = sub_dir + "surf_ssd.jpg";
	cv::imwrite(filename, output_img);
}

void surf_ncc(int threshold, void*) {

	std::vector<cv::DMatch > matches;

	Utils::calculate_surf_ncc(surf_keypoints_1, descriptors_1, surf_keypoints_2, descriptors_2, pair_1_img_1, pair_1_img_2, threshold / (float) ncc_threshold_max, matches);

	cv::Mat output_img;
	cv::drawMatches(pair_1_img_1, surf_keypoints_1, pair_1_img_2, surf_keypoints_2, matches, output_img);
	cv::imshow("surf_ncc", output_img);
	std::string filename = sub_dir + "surf_ncc.jpg";
	cv::imwrite(filename, output_img);
}

int main(int argc, char** argv)  {
	// init
	sub_dir = "pair2/";
#ifndef DEBUG
	pair_1_img_1 = cv::imread(sub_dir + "1.jpg");
	pair_1_img_2 = cv::imread(sub_dir + "2.jpg");
#else
	pair_1_img_1 = cv::imread("test.png");
	pair_1_img_2 = cv::imread("test2.png");
#endif
	cv::namedWindow("ncc", 1);
	cv::namedWindow("ssd", 1);
	cv::namedWindow("surf_ssd", 1);
	cv::namedWindow("surf_ncc", 1);
	cv::namedWindow("surf", 1);
	cv::namedWindow("corners_1", 1);
	cv::namedWindow("corners_2", 1);
	sigma_i = 1;
	// corner detector
	ssd_threshold = 20;
	ncc_threshold = 90;
	//ssd_threshold = 10;
	//ncc_threshold = 995;
	surf_hessian_threshold  = 500;


	cv::createTrackbar("T", "corners_1", &sigma_i, sigma_max, corner_detect);
	cv::createTrackbar("T", "ncc", &ncc_threshold, ncc_threshold_max, ncc);
	cv::createTrackbar("T", "ssd", &ssd_threshold, ssd_threshold_max, ssd);
	cv::createTrackbar("T", "surf", &surf_hessian_threshold, surf_hessian_threshold_max, surf);
	cv::createTrackbar("T", "surf_ssd", &ssd_threshold, ssd_threshold_max, ssd);
	cv::createTrackbar("T", "surf_ncc", &ncc_threshold, ncc_threshold_max, ncc);

	int sigma_i_values[] = {0, 1, 2, 3};
	for (auto& i : sigma_i_values) {
		corner_detect(i, 0);
		ncc(ncc_threshold, 0);
		ssd(ssd_threshold, 0);
		
	}

	//corner_detect(sigma_i, 0);
	//ncc(ncc_threshold, 0);
	//ssd(ssd_threshold, 0);
	//surf(surf_hessian_threshold, 0);
	//surf_ssd(ssd_threshold, 0);
	//surf_ncc(ncc_threshold, 0);

	cv::waitKey(0);

}
